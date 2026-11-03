import copy
from functools import partial
import json
import logging
from pathlib import Path
import uuid

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.utils
from comfy.ldm.minimax.model import FRAME_RESCALE, PackedLayout
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_minimax_h3 import FPS, video_latent_t
import folder_paths

from .regional_attention import H3RegionalGuider


STATE_FORMAT = "h3_experiments_continuation_v1"


def audio_tick(frame):
    return round(frame * 40 / FPS)


def av_streams(latent):
    samples = latent["samples"]
    if not isinstance(samples, comfy.nested_tensor.NestedTensor) or len(samples.tensors) != 2:
        raise ValueError("H3 continuation needs a native joint video/audio latent.")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[:2] != (1, 24) or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
        raise ValueError("H3 continuation expects video [1,24,T,H,W] and audio [1,32,2,T].")
    return video, audio


def frame_count(video):
    steps = video.shape[2]
    if steps < 2 or (steps - 2) % 5:
        raise ValueError("Use a full H3 generation window: 5 + 17*n frames.")
    return 5 + (steps - 2) // 5 * 17


def fit_context(requested, window):
    cap = min(requested, window - 1)
    if cap < 39:
        raise ValueError("Continuation needs a window of at least 56 frames and at least 39 context frames.")
    return 39 + (cap - 39) // 51 * 51


def offset_positions(executor, x, timestep, context, transformer_options, minimax_payload=None, *, start_frame, **kwargs):
    layout = (minimax_payload or {}).get("layout")
    if not isinstance(layout, PackedLayout):
        raise ValueError("H3 continuation requires the native H3 conditioning layout.")
    shifted = copy.copy(layout)
    shifted.position_ids = layout.position_ids.clone()
    for start, stop, kind in layout.segments:
        if kind == "video":
            shifted.position_ids[start:stop, 0] += start_frame * FRAME_RESCALE
        elif kind == "audio":
            shifted.position_ids[start:stop, 0] += audio_tick(start_frame)
    return executor(x, timestep, context, transformer_options.copy(),
                    minimax_payload={**minimax_payload, "layout": shifted}, **kwargs)


def resize_source_masks(sources, video, window, mask_space):
    count = window if mask_space == "frames" else video.shape[2]
    prepared = []
    for source in sources:
        source = source.copy()
        mask = source.get("mask")
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = F.interpolate(mask.float().unsqueeze(1), size=video.shape[-2:],
                                 mode="bilinear", align_corners=False)[:, 0]
            if 1 < mask.shape[0] < count:
                mask = torch.cat((mask, mask[-1:].expand(count - mask.shape[0], -1, -1)))
            source["mask"] = mask
        prepared.append(source)
    return prepared


def window_sources(sources, start_frame, window, mask_space, playback):
    offset = start_frame if mask_space == "frames" else start_frame // 17 * 5
    count = window if mask_space == "frames" else video_latent_t(window)
    result = []
    for source in sources:
        source = source.copy()
        mask = source.get("mask")
        if mask is not None and mask.shape[0] > 1:
            indices = torch.arange(count, device=mask.device) + offset
            indices = indices.remainder(mask.shape[0]) if playback == "loop" else indices.clamp_max(mask.shape[0] - 1)
            source["mask"] = mask.index_select(0, indices)
        result.append(source)
    return result


def carry_prefix(template, history, start_frame, window, context):
    video, audio = av_streams(template)
    previous_video, previous_audio = av_streams(history)
    video = torch.zeros_like(video)
    audio = audio.new_zeros((*audio.shape[:-1], audio_tick(start_frame + window) - audio_tick(start_frame)))
    video_steps, audio_steps = video_latent_t(context), audio_tick(context)
    video[:, :, :video_steps] = previous_video[:, :, -video_steps:].to(video)
    audio[..., :audio_steps] = previous_audio[..., -audio_steps:].to(audio)
    video_mask = video.new_ones((1, 1, *video.shape[2:]), dtype=torch.float32)
    audio_mask = audio.new_ones((1, 1, *audio.shape[2:]), dtype=torch.float32)
    video_mask[:, :, :video_steps] = 0
    audio_mask[..., :audio_steps] = 0
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask))}


def append_future(history, result, context):
    past_video, past_audio = av_streams(history)
    video, audio = av_streams(result)
    vt, at = video_latent_t(context), audio_tick(context)
    video_error = (video[:, :, :vt] - past_video[:, :, -vt:].to(video)).abs().max().item()
    audio_error = (audio[..., :at] - past_audio[..., -at:].to(audio)).abs().max().item()
    if not torch.allclose(video[:, :, :vt], past_video[:, :, -vt:].to(video), rtol=1e-5, atol=1e-5):
        raise RuntimeError("H3 continuation changed the protected video prefix; checkpoint not advanced.")
    if not torch.allclose(audio[..., :at], past_audio[..., -at:].to(audio), rtol=1e-5, atol=1e-5):
        raise RuntimeError("H3 continuation changed the protected audio prefix; checkpoint not advanced.")
    combined = comfy.nested_tensor.NestedTensor((
        torch.cat((past_video.to(video), video[:, :, vt:]), dim=2),
        torch.cat((past_audio.to(audio), audio[..., at:]), dim=-1)))
    return {"samples": combined}, video_error, audio_error


def output_path(path):
    root = Path(folder_paths.get_output_directory()).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("H3 continuation checkpoints must stay inside the ComfyUI output folder.")
    return resolved


def validate_state(state):
    video, audio = av_streams(state["latent"])
    if frame_count(video) != state["frames"] or audio.shape[-1] != audio_tick(state["frames"]):
        raise ValueError("H3 checkpoint tensor lengths do not match its saved timeline.")
    expected = state["window_frames"] + (state["completed_cycles"] - 1) * (state["window_frames"] - state["context_frames"])
    if state["completed_cycles"] < 1 or expected != state["frames"]:
        raise ValueError("H3 checkpoint cycle count does not match its saved timeline.")


def save_state(state, prefix, prompt=None, extra_pnginfo=None):
    validate_state(state)
    target = output_path(prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    path = target.with_name(f"{target.name}_{uuid.uuid4().hex}_cycle{state['completed_cycles']:04d}.safetensors")
    temporary = path.with_suffix(".tmp")
    metadata = {"format": STATE_FORMAT,
                "state": json.dumps({key: value for key, value in state.items() if key != "latent"})}
    if prompt is not None:
        metadata["prompt"] = json.dumps(prompt)
    if extra_pnginfo is not None and "workflow" in extra_pnginfo:
        metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
    video, audio = av_streams(state["latent"])
    try:
        save_file({"video": video.cpu().contiguous(), "audio": audio.cpu().contiguous()}, str(temporary), metadata=metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


class H3LoadContinuationState:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint_path": ("STRING", {"default": "",
            "tooltip": "Explicit checkpoint path from the continuation report, inside ComfyUI/output. Never automatically selects the latest attempt."})}}

    RETURN_TYPES = ("H3_CONTINUATION_STATE",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 Experiments/continuation"

    @classmethod
    def IS_CHANGED(cls, checkpoint_path):
        path = output_path(checkpoint_path)
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def load(self, checkpoint_path):
        path = output_path(checkpoint_path)
        with safe_open(str(path), framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
            if metadata.get("format") != STATE_FORMAT:
                raise ValueError("Choose a checkpoint saved by H3 Continuation Sampler (Beta).")
            state = json.loads(metadata["state"])
            state["latent"] = {"samples": comfy.nested_tensor.NestedTensor((file.get_tensor("video"), file.get_tensor("audio")))}
        validate_state(state)
        return (state,)


class H3ContinuationSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
            "cycles": ("INT", {"default": 2, "min": 1, "max": 1000,
                "tooltip": "Total cycles, including any loaded checkpoint cycles. 1 matches the first normal generation. Later cycles append window minus context frames. This sampler requests no previews or decoding; bypass separate preview patches for a decode-free loop."}),
            "context_frames": ("INT", {"default": 39, "min": 39, "max": 9999, "step": 51,
                "tooltip": "Protected video/audio history: 39, 90, 141, ... frames. Snaps down to fit. Larger context leaves fewer new frames per cycle."}),
            "mask_playback": (["loop", "hold"], {"default": "loop",
                "tooltip": "Advance masks on the full sequence timeline. Short masks first hold to fill the initial window. Loop repeats that sequence; hold retains its final frame."}),
            "save_checkpoints": ("BOOLEAN", {"default": True}),
            "checkpoint_prefix": ("STRING", {"default": "h3_continuation/run"}),
        }, "optional": {"resume_state": ("H3_CONTINUATION_STATE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("LATENT", "H3_CONTINUATION_STATE", "STRING")
    RETURN_NAMES = ("full_sequence", "continuation_state", "report")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3 Experiments/continuation"
    DESCRIPTION = "Separate beta: internal H3 regional continuation cycles. Carries real AV latent history, advances masks and generated positions, and saves checkpoints. Connect final decoders downstream; bypass separate preview patches to avoid intermediate decoding. Resume uses the connected model/conditioning."

    def sample(self, noise, guider, sampler, sigmas, latent_image, cycles=2, context_frames=39,
               mask_playback="loop", save_checkpoints=True, checkpoint_prefix="h3_continuation/run",
               resume_state=None, prompt=None, extra_pnginfo=None):
        if not isinstance(guider, H3RegionalGuider):
            raise ValueError("Connect H3 Regional Attention Guider or H3 Regional Interaction Guider to this sampler.")
        if cycles < 1:
            raise ValueError("H3 continuation needs at least one cycle.")
        if mask_playback not in ("loop", "hold"):
            raise ValueError("mask_playback must be loop or hold.")
        if sigmas.numel() < 2 or sigmas[-1].item() != 0:
            raise ValueError("H3 continuation requires a schedule ending at zero to save a clean continuation state.")
        if save_checkpoints:
            output_path(checkpoint_prefix)
        video, audio = av_streams(latent_image)
        window = frame_count(video)
        if audio.shape[-1] != audio_tick(window):
            raise ValueError("Use the normal H3 AV latent for the initial generation window.")
        context = fit_context(context_frames, window) if window >= 56 or cycles > 1 or resume_state else context_frames
        state = None
        if resume_state is not None:
            validate_state(resume_state)
            previous_video, _ = av_streams(resume_state["latent"])
            if previous_video.shape[:2] + previous_video.shape[3:] != video.shape[:2] + video.shape[3:]:
                raise ValueError("Keep the checkpoint's canvas size when continuing.")
            if (resume_state["window_frames"], resume_state["context_frames"], resume_state["mask_playback"]) != (window, context, mask_playback):
                raise ValueError("Keep the saved window, context length and mask playback to preserve the continuation timeline.")
            state = resume_state.copy()
        completed = state["completed_cycles"] if state else 0
        sources = resize_source_masks(guider.sources, video, window, guider.mask_space) if cycles > max(1, completed) else None
        report = [f"Window {window} frames; context {context}; cycles {completed} -> {max(cycles, completed)}.",
                  "Direct AV carry; sampler requests no decode or explicit model unload. External preview patches still apply. Resume uses connected model/conditioning."]
        steps = sigmas.numel() - 1
        progress = comfy.utils.ProgressBar(max(0, cycles - completed) * steps)
        for index in range(completed, cycles):
            comfy.model_management.throw_exception_if_processing_interrupted()
            start = state["frames"] - context if state else 0
            target = carry_prefix(latent_image, state["latent"], start, window, context) if state else latent_image
            current_guider = copy.copy(guider)
            current_guider.model_patcher = guider.model_patcher.clone()
            current_guider.model_patcher.model_options = comfy.model_patcher.create_model_options_clone(guider.model_options)
            current_guider.model_options = current_guider.model_patcher.model_options
            if start:
                current_guider.sources = window_sources(sources, start, window, guider.mask_space, mask_playback)
                current_guider.model_patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "h3_continuation_clock",
                    partial(offset_positions, start_frame=start))
            current_noise = copy.copy(noise)
            current_noise.seed = (noise.seed + index) % (2**64)

            def callback(step, x0, x, total_steps):
                progress.update_absolute((index - completed) * steps + step + 1)

            samples = current_guider.sample(current_noise.generate_noise(target), target["samples"], sampler, sigmas,
                denoise_mask=target.get("noise_mask"), callback=callback, disable_pbar=True, seed=current_noise.seed)
            result = {"samples": samples.to(comfy.model_management.intermediate_device())}
            errors = ""
            if state:
                result, video_error, audio_error = append_future(state["latent"], result, context)
                errors = f"; preserved-prefix max error video={video_error:.3g}, audio={audio_error:.3g}"
            state = {"latent": result, "frames": start + window, "completed_cycles": index + 1,
                     "window_frames": window, "context_frames": context, "mask_playback": mask_playback,
                     "last_seed": current_noise.seed}
            validate_state(state)
            line = f"Cycle {index + 1}: window starts at frame {start}; total {state['frames']} frames / {state['frames']/FPS:.3f}s{errors}"
            if save_checkpoints:
                line += "\n" + save_state(state, checkpoint_prefix, prompt, extra_pnginfo)
            report.append(line)
            logging.info("[H3 continuation] %s", line)
        return (state["latent"], state, "\n".join(report))
