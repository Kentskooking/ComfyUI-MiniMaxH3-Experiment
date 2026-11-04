import math

import torch
import torch.nn.functional as F

from .reference_curve import CURVES, temporal_curve_weights


def rms_distance(a, b):
    return torch.linalg.vector_norm(a.float() - b.float()).item() / math.sqrt(a.numel())


class H3OmniReferenceBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_a": ("CONDITIONING",),
            "conditioning_b": ("CONDITIONING",),
            "context_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "0 = A, 1 = B. Mixes Qwen text and visual embeddings before H3 text refinement. B is truncated or zero-padded to A's token count; A's token tags are retained."}),
            "reference_latent_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Constant reference ratio: 0 = A, 1 = B. Used when reference_curve is constant; animated curves use curve_start/end instead."}),
        }, "optional": {
            "reference_curve": (list(CURVES), {"default": "constant", "tooltip": "Mix over the reference video's duration. Sine/triangle go start -> end -> start per cycle. Linear and exponential easing go start -> end once. This does not schedule denoising steps or pin output frames."}),
            "curve_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Starting B weight (0=A, 1=B). Waves return to this value. Swap start/end to invert a curve."}),
            "curve_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ending B weight for ramps; peak B weight for sine/triangle waves."}),
            "curve_cycles": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 32.0, "step": 0.25, "tooltip": "Cycles across the whole reference clip, for sine/triangle only."}),
            "curve_phase": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Phase in cycles for sine/triangle only; 0.5 starts at the opposite end."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "mix_report")
    FUNCTION = "blend"
    CATEGORY = "MiniMax H3 Experiments/conditioning"
    DESCRIPTION = "Blend two omni conditioning packages before generation. Uses A's token layout, with no semantic alignment. Video references must have matching dimensions and frame counts. Audio references are not mixed."

    def blend(self, conditioning_a, conditioning_b, context_mix=0.5, reference_latent_mix=0.5,
              reference_curve="constant", curve_start=0.0, curve_end=1.0, curve_cycles=1.0, curve_phase=0.0):
        if len(conditioning_a) != 1 or len(conditioning_b) != 1:
            raise ValueError("H3 Omni Reference Blend needs one conditioning entry per input. Connect each Reference to Video node directly.")
        a, ma = conditioning_a[0]
        b, mb = conditioning_b[0]
        for key in ("control", "hooks", "gligen", "model_conds", "minimax_keyframes"):
            if ma.get(key) is not None or mb.get(key) is not None:
                raise ValueError(f"H3 Omni Reference Blend cannot mix {key}; connect unmodified omni conditioning.")

        length_a, length_b = a.shape[1], b.shape[1]
        b = b[:, :length_a].to(a)
        if length_b < length_a:
            b = F.pad(b, (0, 0, 0, length_a - length_b))
        context = torch.lerp(a, b, context_mix)
        metadata = ma.copy()
        for key in ("mask", "mask_strength", "strength", "area", "set_area_to_bounds", "start_percent", "end_percent"):
            metadata.pop(key, None)
        metadata.pop("pooled_output", None)
        if ma.get("pooled_output") is not None and mb.get("pooled_output") is not None:
            metadata["pooled_output"] = torch.lerp(ma["pooled_output"], mb["pooled_output"].to(ma["pooled_output"]), context_mix)

        tags = []
        for tensor, source in ((a, ma), (conditioning_b[0][0], mb)):
            tag = source.get("minimax_token_tags")
            if tag is None:
                tag = torch.ones(tensor.shape[1], dtype=torch.long, device=a.device)
            tag = tag.flatten().to(device=a.device)
            if tag.numel() != tensor.shape[1]:
                raise ValueError("H3 token tags must match their conditioning token count.")
            tags.append(F.pad(tag[:length_a], (0, max(0, length_a - tag.numel())), value=1))
        metadata["minimax_token_tags"] = tags[0]
        mismatches = (tags[0] != tags[1]).sum().item()
        reference_summary = (f"reference mix: A={1-reference_latent_mix:.3f}, B={reference_latent_mix:.3f}" if reference_curve == "constant"
                             else f"reference curve: {reference_curve}, start={curve_start:.3f}, end/peak={curve_end:.3f}")
        if reference_curve in ("sine", "triangle"):
            reference_summary += f", cycles={curve_cycles:g}, phase={curve_phase:g}"
        report = [
            f"Context mix: A={1-context_mix:.3f}, B={context_mix:.3f}; {reference_summary}",
            f"Tokens: A={length_a}, B={length_b}, output={length_a}; B truncated={max(0, length_b-length_a)}, zero-padded={max(0, length_a-length_b)}",
            f"A token layout retained; text={tags[0].eq(1).sum().item()}, visual={tags[0].eq(0).sum().item()}; B tag mismatches={mismatches} (no alignment)",
            f"Context RMS distance: A-B={rms_distance(a, b):.6g}, output-A={rms_distance(context, a):.6g}, output-B={rms_distance(context, b):.6g}",
        ]

        refs_a, refs_b = ma.get("minimax_refs", []), mb.get("minimax_refs", [])
        if not refs_a or len(refs_a) != len(refs_b):
            raise ValueError("H3 Omni Reference Blend needs matching reference slots with the video VAE connected to both reference nodes.")
        refs = []
        for index, (ra, rb) in enumerate(zip(refs_a, refs_b), 1):
            if ra["kind"] not in ("image", "video") or ra["kind"] != rb["kind"]:
                raise ValueError("Pair matching image/video references. Disconnect reference audio for this experiment.")
            za, zb = ra["latent"], rb["latent"]
            if za.shape != zb.shape or any(ra.get(key) != rb.get(key) for key in ("latent_t", "latent_h", "latent_w")):
                raise ValueError("Reference latent sizes differ. Resize both videos and fit them to the same frame count before the reference nodes.")
            zb = zb.to(za)
            mixed = ra.copy()
            weight = reference_latent_mix
            if reference_curve != "constant":
                weights, frames = temporal_curve_weights(za.shape[2], reference_curve, reference_latent_mix,
                                                         curve_start, curve_end, curve_cycles, curve_phase)
                weight = za.new_tensor(weights).view(1, 1, -1, 1, 1)
                report.append(f"Reference {index} curve: {frames} reference frames -> {len(weights)} latent times; B weights first/middle/last={weights[0]:.6g}/{weights[len(weights)//2]:.6g}/{weights[-1]:.6g}; min/max={min(weights):.6g}/{max(weights):.6g}")
            mixed["latent"] = torch.lerp(za, zb, weight)
            refs.append(mixed)
            report.append(f"Reference {index} ({ra['kind']}) {list(za.shape)} RMS distance: A-B={rms_distance(za, zb):.6g}, output-A={rms_distance(mixed['latent'], za):.6g}, output-B={rms_distance(mixed['latent'], zb):.6g}")
        metadata["minimax_refs"] = refs
        if reference_curve != "constant":
            report.append("Curve weights are averaged over H3's 1,4,4,4,4 reference-frame groups. Context mix stays constant; generated-frame timing is not pinned.")
        report.append("One blended conditioning entry. References are not starting latents or pinned frames. RMS measures tensor mixing, not semantic quality.")
        return ([[context, metadata]], "\n".join(report))
