from functools import partial

import torch
import torch.nn.functional as F

import comfy.model_patcher
import comfy.samplers
from comfy.ldm.minimax.model import Attention, MiniMaxH3Model
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

from .latent_composite import composite_mask


def join_conditioning(conditioning):
    if not conditioning:
        raise ValueError("Regional attention needs at least one conditioning entry. Use Conditioning Combine after setting the source masks.")
    lengths = tuple(c[0].shape[1] for c in conditioning)
    tags, keyframes, owners = [], [], []
    common = {}
    per_source = {"mask", "mask_strength", "strength", "set_area_to_bounds", "minimax_token_tags", "minimax_keyframes"}
    for bank, (context, metadata) in enumerate(conditioning):
        if context.shape[0] != 1:
            raise ValueError("H3 regional attention requires batch size one.")
        if metadata.get("minimax_refs"):
            raise ValueError("This regional attention beta supports Image to Video conditioning; Omni references are not implemented yet.")
        if any(key in metadata for key in ("area", "hooks", "control", "gligen", "start_percent", "end_percent")):
            raise ValueError("Use plain H3 I2V conditioning with Conditioning Set Mask for this regional attention beta.")
        for key, value in metadata.items():
            if key in per_source or key == "minimax_refs":
                continue
            if key in common:
                same = torch.equal(value, common[key]) if isinstance(value, torch.Tensor) else value == common[key]
                if not same:
                    raise ValueError(f"Regional attention requires matching shared conditioning setting: {key}.")
            common[key] = value
        token_tags = metadata.get("minimax_token_tags")
        if token_tags is None:
            token_tags = torch.ones(context.shape[1], device=context.device, dtype=torch.long)
        if token_tags.numel() != context.shape[1]:
            raise ValueError("H3 modality tags must match each source's token count.")
        tags.append(token_tags.reshape(-1))
        for frame in metadata.get("minimax_keyframes", []):
            keyframes.append(frame.copy())
            if frame.get("latent") is not None:
                owners.append(("cond", bank))
            if frame.get("audio_latent") is not None:
                owners.append(("cond_audio", bank))
    first = conditioning[0][0]
    common["minimax_token_tags"] = torch.cat([tag.to(tags[0]) for tag in tags])
    common["minimax_keyframes"] = keyframes
    return [[torch.cat([c[0].to(first) for c in conditioning], dim=1), common]], lengths, owners


def refine_sources(model, text_states, lengths):
    # Keep source identities intact until they enter the shared DiT blocks.
    return torch.cat([MiniMaxH3Model.preprocess_text_embeds(model, source)
                      for source in text_states.split(lengths, dim=1)], dim=1)


class RegionalRouting:
    """Small, sampling-scoped routing weights; never retains a full attention matrix."""

    def __init__(self, sources, lengths, owners, video_shape, device, mask_space, query_chunk):
        video = torch.empty((), device=device).expand(video_shape)
        weights, strengths = [], []
        for metadata in sources:
            strength = max(0.0, metadata.get("strength", 1.0) * metadata.get("mask_strength", 1.0))
            strengths.append(strength)
            weight = composite_mask(metadata.get("mask"), video, mask_space)
            weight = weight.expand(1, 1, *video_shape[2:])[0, 0]
            h, w = weight.shape[-2:]
            weight = F.pad(weight, (0, -w % 2, 0, -h % 2), mode="replicate")
            weights.append(F.avg_pool2d(weight.unsqueeze(1), 2).flatten().clamp_min(0) * strength)
        weights = torch.stack(weights, dim=1)
        self.video_log_weights = (weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-20)).log()
        strengths = torch.tensor(strengths, device=device, dtype=torch.float32)
        self.audio_log_weights = (strengths / strengths.sum().clamp_min(1e-20)).log()
        self.lengths = lengths
        self.owners = owners
        self.query_chunk = query_chunk
        self.spans = None

    def bind(self, layout):
        if self.spans is not None:
            return
        self.spans = []
        offset = 0
        for bank, length in enumerate(self.lengths):
            self.spans.append((offset, offset + length, bank))
            offset += length
        guides = [(start, stop, kind) for start, stop, kind in layout.segments if kind in ("cond", "cond_audio")]
        if len(guides) != len(self.owners) or layout.signature[0] != offset:
            raise ValueError("H3 regional routing does not match the native conditioning layout.")
        for (start, stop, kind), (expected_kind, bank) in zip(guides, self.owners):
            if kind != expected_kind:
                raise ValueError("H3 regional guide order does not match the native conditioning layout.")
            self.spans.append((start, stop, bank))
        self.video_start, self.video_stop, _ = next(s for s in layout.segments if s[2] == "video")
        self.audio_start, self.audio_stop, _ = next(s for s in layout.segments if s[2] == "audio")
        if self.video_stop - self.video_start != self.video_log_weights.shape[0]:
            raise ValueError("H3 regional mask token count does not match the generated video.")
        self.seq_len = layout.seq_len

    def bias(self, start, stop, device, dtype):
        bias = torch.zeros((stop - start, self.seq_len), device=device, dtype=dtype)
        for key_start, key_stop, bank in self.spans:
            # Source tokens keep their identity; interaction occurs through the shared generated tokens.
            for query_start, query_stop, query_bank in self.spans:
                left, right = max(start, query_start), min(stop, query_stop)
                if left < right and bank != query_bank:
                    bias[left - start:right - start, key_start:key_stop] = -torch.inf
            left, right = max(start, self.audio_start), min(stop, self.audio_stop)
            if left < right:
                bias[left - start:right - start, key_start:key_stop] = self.audio_log_weights[bank]
            left, right = max(start, self.video_start), min(stop, self.video_stop)
            if left < right:
                bias[left - start:right - start, key_start:key_stop] = self.video_log_weights[
                    left - self.video_start:right - self.video_start, bank, None]
        # All generated video/audio key columns stay open, including across mask boundaries.
        return bias

    def attention(self, original, q, k, v, heads, *, preferred, transformer_options, **kwargs):
        self.bind(transformer_options["minimax_h3_layout"])
        options = transformer_options.copy()
        options.pop("optimized_attention_override", None)
        kwargs.pop("_inside_attn_wrapper", None)
        kwargs.pop("mask", None)
        output = None
        for start in range(0, self.seq_len, self.query_chunk):
            stop = min(start + self.query_chunk, self.seq_len)
            result = optimized_attention(
                AttentionTensorContainer(q[:, :, start:stop]), AttentionTensorContainer(k), AttentionTensorContainer(v),
                heads, mask=self.bias(start, stop, q.device, q.dtype), preferred_attention=preferred,
                transformer_options=options, **kwargs)
            if output is None:
                output = result.new_empty((result.shape[0], self.seq_len, result.shape[-1]))
            output[:, start:stop] = result
        return output


def regional_forward(attention, x, rope_freqs=None, transformer_options={}):
    routing = transformer_options.get("h3_regional_attention")
    if routing is None:
        raise ValueError("Regional H3 attention must run through H3 Regional Attention Guider.")
    options = transformer_options.copy()
    options["optimized_attention_override"] = partial(routing.attention, preferred=attention.comfy_attention)
    return Attention.forward(attention, x, rope_freqs, options)


class H3RegionalGuider(comfy.samplers.CFGGuider):
    def set_regions(self, conditioning, lengths, owners, mask_space, query_chunk):
        self.sources = [c[1] for c in conditioning]
        self.lengths = lengths
        self.owners = owners
        self.mask_space = mask_space
        self.query_chunk = query_chunk

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if latent_shapes is None or len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or latent_shapes[0][1] != 24:
            raise ValueError("H3 Regional Attention Guider needs joint H3 video/audio latents.")
        routing = RegionalRouting(self.sources, self.lengths, self.owners, latent_shapes[0], device,
                                  self.mask_space, self.query_chunk)
        original_options = self.model_options
        self.model_options = comfy.model_patcher.create_model_options_clone(original_options)
        self.model_options.setdefault("transformer_options", {})["h3_regional_attention"] = routing
        try:
            return super().inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed,
                                        latent_shapes=latent_shapes)
        finally:
            self.model_options = original_options


class H3RegionalAttentionGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING", {"tooltip": "I2V branches: Conditioning Set Mask on each, then Conditioning Combine. All prompts and image guides enter one shared model evaluation; supports two, five, or more sources."}),
            "mask_space": (["frames", "latent"], {"default": "frames"}),
            "query_chunk": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64,
                                      "tooltip": "Attention query rows per chunk. All keys remain visible. Smaller uses less temporary mask memory; larger reduces kernel launches."}),
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "MiniMax H3 Experiments/sampling"
    DESCRIPTION = "Beta: joint regional attention at CFG 1. Routes all source contexts and image guides inside every DiT attention block; generated video/audio attention remains global. Replaces unmasked attention patches on this model clone. No separate regional predictions or latent overlays."

    def get_guider(self, model, conditioning, mask_space="frames", query_chunk=512):
        native = model.get_model_object("diffusion_model")
        if not isinstance(native, MiniMaxH3Model):
            raise ValueError("H3 Regional Attention Guider requires the native ComfyUI MiniMax H3 model.")
        joined, lengths, owners = join_conditioning(conditioning)
        patched = model.clone()
        patched.add_object_patch("diffusion_model.preprocess_text_embeds", partial(refine_sources, native, lengths=lengths))
        for i, block in enumerate(native.blocks):
            patched.add_object_patch(f"diffusion_model.blocks.{i}.attn.forward", partial(regional_forward, block.attn))
        guider = H3RegionalGuider(patched)
        guider.set_regions(conditioning, lengths, owners, mask_space, query_chunk)
        guider.inner_set_conds({"positive": joined})
        return (guider,)
