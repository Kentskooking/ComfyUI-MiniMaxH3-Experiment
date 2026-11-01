import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN


def video_stream(latent):
    samples = latent["samples"]
    video = samples.tensors[0] if isinstance(samples, comfy.nested_tensor.NestedTensor) else samples
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("H3 Latent Composite Masked needs H3 video latents [B,24,T,H,W] or joint H3 AV latents.")
    return video


def composite_mask(mask, video, mask_space):
    if mask is None:
        return video.new_ones((1, 1, 1, 1, 1), dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] == 0:
        raise ValueError("Composite mask must be [frames,height,width], or one static [height,width] mask.")
    frames = F.interpolate(mask.to(device=video.device, dtype=torch.float32).unsqueeze(1),
                           size=video.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    if frames.shape[0] == 1:
        return frames.unsqueeze(0).unsqueeze(0)
    rows = []
    offset = 0
    for i in range(video.shape[2]):
        span = FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] if mask_space == "frames" else 1
        chunk = frames[offset:offset + span]
        if chunk.shape[0] < span:
            chunk = torch.cat((chunk, frames[-1:].expand(span - chunk.shape[0], -1, -1)))
        rows.append(chunk.mean(dim=0))
        offset += span
    return torch.stack(rows).unsqueeze(0).unsqueeze(0)


def video_noise_mask(latent, video):
    mask = latent.get("noise_mask")
    if isinstance(mask, comfy.nested_tensor.NestedTensor):
        mask = mask.tensors[0]
    shape = (video.shape[0], 1, *video.shape[2:])
    if mask is None:
        return video.new_ones(shape, dtype=torch.float32)
    return comfy.utils.reshape_mask(mask.to(device=video.device, dtype=torch.float32), shape)


class H3LatentCompositeMasked:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("LATENT", {"tooltip": "H3 destination video or AV latent. Its audio and other metadata are retained."}),
                "source": ("LATENT", {"tooltip": "H3 source video or AV latent on the same canvas and timeline. Only its video is composited."}),
                "mask_space": (["frames", "latent"], {"default": "frames", "tooltip": "frames: moving masks use H3's repeating 1,4,4,4,4 frame groups. latent: one mask per video latent step. A single mask is static."}),
                "noise_mask_mode": (["blend", "destination", "none"], {"default": "blend", "tooltip": "blend: composite the input video denoise masks too, retaining destination audio protection. destination: retain its existing mask. none: remove denoise masks."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "White selects source, black selects destination, gray blends. This selects content; it is separate from a denoise mask. Disconnected selects all source video."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "composite"
    CATEGORY = "MiniMax H3 Experiments/latent"
    DESCRIPTION = "Composite H3 video latents with static or moving soft masks. Keeps destination audio. Does not combine conditioning or encode the images supplied to Image to Video nodes."

    def composite(self, destination, source, mask_space="frames", noise_mask_mode="blend", mask=None):
        dest_video = video_stream(destination)
        source_video = video_stream(source)
        if any(s != d and s != 1 for s, d in zip(source_video.shape, dest_video.shape)):
            raise ValueError("Match the source and destination canvas and frame count before H3 latent compositing.")
        weight = composite_mask(mask, dest_video, mask_space)
        video = torch.lerp(dest_video, source_video.to(dest_video), weight.to(dtype=dest_video.dtype))
        output = destination.copy()
        dest_samples = destination["samples"]
        joint = isinstance(dest_samples, comfy.nested_tensor.NestedTensor)
        output["samples"] = comfy.nested_tensor.NestedTensor((video, dest_samples.tensors[1])) if joint else video

        if noise_mask_mode == "none":
            output.pop("noise_mask", None)
        elif noise_mask_mode == "blend" and ("noise_mask" in destination or "noise_mask" in source):
            dest_mask = video_noise_mask(destination, dest_video)
            source_mask = video_noise_mask(source, source_video)
            video_mask = torch.lerp(dest_mask, source_mask.to(dest_mask), weight)
            if joint:
                prior_mask = destination.get("noise_mask")
                if isinstance(prior_mask, comfy.nested_tensor.NestedTensor):
                    audio_mask = prior_mask.tensors[1]
                else:
                    audio = dest_samples.tensors[1]
                    audio_mask = audio.new_ones((audio.shape[0], 1, *audio.shape[2:]), dtype=torch.float32)
                output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
            else:
                output["noise_mask"] = video_mask
        return (output,)
