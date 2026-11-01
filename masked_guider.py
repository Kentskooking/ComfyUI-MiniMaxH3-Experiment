import itertools
import math

import torch

import comfy.samplers
import comfy.utils

from .latent_composite import composite_mask
from .overlap_conditioning import average_conditioning


class H3MaskedGuider(comfy.samplers.CFGGuider):
    def set_conds(self, conditioning, mask_space, blend_mode="predictions", overlap_mix=1.0):
        self.mask_space = mask_space
        self.blend_mode = blend_mode
        self.overlap_mix = overlap_mix
        self.condition_count = len(conditioning)
        self.overlap_pairs = []
        if blend_mode == "embedding_overlap":
            if len(conditioning) < 2:
                raise ValueError("Embedding overlap needs at least two separate conditioning entries. Use Conditioning Combine.")
            self.overlap_pairs = list(itertools.combinations(range(self.condition_count), 2))
            conditioning = [*conditioning, *(average_conditioning(conditioning[i], conditioning[j]) for i, j in self.overlap_pairs)]
        self.inner_set_conds({"positive": conditioning})

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if latent_shapes is None or len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or latent_shapes[0][1] != 24:
            raise ValueError("H3 Masked Conditioning Guider needs joint H3 video/audio latents.")
        original = self.conds["positive"]
        self.video_elements = math.prod(latent_shapes[0][1:])
        prepared = []
        # Only mask weights are packed here; ComfyUI owns the latent layout.
        video = noise.new_empty(1).expand(latent_shapes[0])
        audio = noise.new_ones(1).expand(latent_shapes[1])
        for cond in original:
            if "area" in cond:
                raise ValueError("Use Conditioning Set Mask instead of Conditioning Set Area with the H3 masked guider.")
            cond = cond.copy()
            if "mask" in cond:
                weight = composite_mask(cond["mask"], video, self.mask_space).expand_as(video)
                weight, _ = comfy.utils.pack_latents((weight, audio))
                cond["mask"] = weight[:, 0]
            # H3 references need the full canvas, including outside the mask.
            cond.pop("set_area_to_bounds", None)
            prepared.append(cond)
        self.conds["positive"] = prepared
        try:
            return super().inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            self.conds["positive"] = original

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        total = torch.zeros_like(x)
        count = torch.zeros_like(x)
        conditions = self.conds["positive"]
        if self.blend_mode == "embedding_overlap":
            regions = [comfy.samplers.get_area_and_mult(c, x, timestep) for c in conditions[:self.condition_count]]
            weights = [r.mult if r is not None else torch.zeros_like(x) for r in regions]
            if self.condition_count == 2:
                shared = torch.minimum(*weights) * self.overlap_mix
                contributions = zip(conditions, (weights[0] - shared, weights[1] - shared, 2 * shared))
            else:
                weight_sum = sum(weights)
                remaining = [w.clone() for w in weights]
                pair_weights = []
                for i, j in self.overlap_pairs:
                    denominator = weight_sum - torch.minimum(weights[i], weights[j])
                    shared = weights[i] * weights[j] / torch.where(denominator != 0, denominator, 1) * self.overlap_mix
                    # Multi-region audio keeps the original branch average; it must not activate unrelated image pairs.
                    shared[..., self.video_elements:] = 0
                    remaining[i].sub_(shared)
                    remaining[j].sub_(shared)
                    pair_weights.append(2 * shared)
                contributions = zip(conditions, [*(w.clamp_min_(0) for w in remaining), *pair_weights])
        else:
            regions = [comfy.samplers.get_area_and_mult(c, x, timestep) for c in conditions]
            contributions = ((c, r.mult) for c, r in zip(conditions, regions) if r is not None)
        for cond, weight in contributions:
            if self.blend_mode == "embedding_overlap" and not bool(weight.any()):
                continue
            full = cond.copy()
            full.pop("mask", None)
            full.pop("mask_strength", None)
            full.pop("strength", None)
            # Separate calls keep H3 at batch size one, even for identical refs.
            prediction = comfy.samplers.calc_cond_batch(self.inner_model, [[full]], x, timestep, model_options)[0]
            total.addcmul_(prediction, weight)
            count.add_(weight)
        covered = count != 0
        prediction = torch.where(covered, total / torch.where(covered, count, 1), x)
        conds = [self.conds["positive"], None]
        out = [prediction, torch.zeros_like(prediction)]
        for fn in model_options.get("sampler_pre_cfg_function", []):
            out = fn({"conds": conds, "conds_out": out, "cond_scale": 1.0, "timestep": timestep,
                      "input": x, "sigma": timestep, "model": self.inner_model, "model_options": model_options})
        return comfy.samplers.cfg_function(self.inner_model, out[0], out[1], 1.0, x, timestep,
                                          model_options=model_options, cond=conds[0], uncond=None)


class H3MaskedConditioningGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING", {"tooltip": "Combine H3 branches after Conditioning Set Mask. Embedding overlap adds pairwise averaged-conditioning predictions where masks overlap. Audio uses global branch strengths."}),
            "mask_space": (["frames", "latent"], {"default": "frames", "tooltip": "frames: moving masks use H3's 1,4,4,4,4 frame groups. latent: one mask per video latent step. Single masks are static."}),
        }, "optional": {
            "blend_mode": (["predictions", "embedding_overlap"], {"default": "predictions", "tooltip": "predictions: original mask-weighted prediction average. embedding_overlap: mixes pairwise averaged embeddings and visual latents in overlaps. Supports text, matching first/last frames, or matching image-reference slots. More overlapping pairs require more model evaluations."}),
            "overlap_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "In embedding_overlap mode: 0 uses the original prediction blend, 1 uses full embedding overlap. Intermediate values mix the two behaviors. Affects generated audio with two branches; three or more keep the original audio average. Ignored in predictions mode."}),
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "MiniMax H3 Experiments/sampling"
    DESCRIPTION = "Regional H3 blending at CFG 1. Embedding overlap adds averaged-conditioning predictions for pairs of overlapping masks. This is not spatial attention routing. Uncovered video regions receive no denoising update."

    def get_guider(self, model, conditioning, mask_space="frames", blend_mode="predictions", overlap_mix=1.0):
        guider = H3MaskedGuider(model)
        guider.set_conds(conditioning, mask_space, blend_mode, overlap_mix)
        return (guider,)
