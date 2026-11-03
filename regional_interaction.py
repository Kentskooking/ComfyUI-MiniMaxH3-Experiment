from functools import partial
import math

import torch
import torch.nn.functional as F

import comfy.model_patcher
import comfy.samplers
from comfy.ldm.minimax.model import MiniMaxH3Model

from .regional_attention import (
    H3RegionalAttentionGuider, H3RegionalGuider, RegionalRouting,
    join_conditioning, refine_sources, regional_forward,
)


class InteractionRouting(RegionalRouting):
    def __init__(self, sources, lengths, owners, video_shape, device, mask_space, query_chunk, interaction):
        super().__init__(sources, lengths, owners, video_shape, device, mask_space, query_chunk)
        self.interaction = interaction
        if interaction < 1:
            weights = self.video_log_weights.exp()
            # Uncovered tokens form their own group instead of bypassing separation.
            uncovered = (weights.sum(dim=1, keepdim=True) == 0).to(weights.dtype)
            self.membership = F.normalize(torch.cat((weights, uncovered), dim=1), dim=1)
            self.source_video_bias = self.membership[:, :len(lengths)].mul(1 - interaction).add(interaction).log()
            self.audio_readback_bias = math.log(interaction) if interaction > 0 else -math.inf

    def bias(self, start, stop, device, dtype):
        bias = super().bias(start, stop, device, dtype)
        if self.interaction == 1:
            return bias
        for query_start, query_stop, bank in self.spans:
            left, right = max(start, query_start), min(stop, query_stop)
            if left < right:
                bias[left - start:right - start, self.video_start:self.video_stop] = self.source_video_bias[:, bank]
                bias[left - start:right - start, self.audio_start:self.audio_stop] = self.audio_readback_bias
        left, right = max(start, self.video_start), min(stop, self.video_stop)
        if left < right:
            membership = self.membership[left - self.video_start:right - self.video_start]
            affinity = (membership @ self.membership.T).clamp_(0, 1)
            affinity.mul_(1 - self.interaction).add_(self.interaction).log_()
            bias[left - start:right - start, self.video_start:self.video_stop] = affinity
            bias[left - start:right - start, self.audio_start:self.audio_stop] = self.audio_readback_bias
        # Audio still reads the whole scene; its return path into video/sources is gated above.
        return bias


class H3InteractionGuider(H3RegionalGuider):
    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if latent_shapes is None or len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or latent_shapes[0][1] != 24:
            raise ValueError("H3 Regional Interaction Guider needs joint H3 video/audio latents.")
        routing = InteractionRouting(self.sources, self.lengths, self.owners, latent_shapes[0], device,
                                     self.mask_space, self.query_chunk, self.interaction)
        original_options = self.model_options
        self.model_options = comfy.model_patcher.create_model_options_clone(original_options)
        self.model_options.setdefault("transformer_options", {})["h3_regional_attention"] = routing
        try:
            return comfy.samplers.CFGGuider.inner_sample(
                self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed,
                latent_shapes=latent_shapes)
        finally:
            self.model_options = original_options


class H3RegionalInteractionGuider(H3RegionalAttentionGuider):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["interaction"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
            "tooltip": "1 = original regional attention. Lower values reduce cross-region video attention and shared-context readback. 0 blocks direct communication between disjoint regions; overlapping masks can still connect them. Not the old prediction blend.",
        })
        return inputs

    DESCRIPTION = "Separate beta: one shared H3 denoise with adjustable interaction between mask regions. 1 matches H3 Regional Attention Guider; lower values preserve stronger regional separation. Animated masks still apply at each latent time position. Existing guiders are unchanged."

    def get_guider(self, model, conditioning, mask_space="frames", query_chunk=512, interaction=1.0):
        if not 0 <= interaction <= 1:
            raise ValueError("interaction must be between 0 and 1 for valid attention weights.")
        native = model.get_model_object("diffusion_model")
        if not isinstance(native, MiniMaxH3Model):
            raise ValueError("H3 Regional Interaction Guider requires the native ComfyUI MiniMax H3 model.")
        joined, lengths, owners = join_conditioning(conditioning)
        patched = model.clone()
        patched.add_object_patch("diffusion_model.preprocess_text_embeds", partial(refine_sources, native, lengths=lengths))
        for i, block in enumerate(native.blocks):
            patched.add_object_patch(f"diffusion_model.blocks.{i}.attn.forward", partial(regional_forward, block.attn))
        guider = H3InteractionGuider(patched)
        guider.set_regions(conditioning, lengths, owners, mask_space, query_chunk)
        guider.interaction = interaction
        guider.inner_set_conds({"positive": joined})
        return (guider,)
