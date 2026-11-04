import numpy as np
from scipy import ndimage
import torch


def ramp(value, curve):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value) if curve == "smoothstep" else value


class H3MaskProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "mode": (["original", "gaussian", "interior_gradient", "average_inside"], {"default": "interior_gradient", "tooltip": "original: use the input weights. gaussian: blur them. interior_gradient: pure A core with inward and outward fades. average_inside: averaged A+B inside white, B outside (use embedding_overlap with overlap_mix=1)."}),
            "transition_width": ("FLOAT", {"default": 96.0, "min": 0.0, "max": 4096.0, "step": 1.0, "tooltip": "Interior gradient only. Distance inward from the white boundary to pure A, in input-mask pixels. Increase to shrink the pure-A core; zero disables the inward fade."}),
            "outer_fade": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 4096.0, "step": 1.0, "tooltip": "Interior gradient only. Fade outside the boundary from A weight 0.25 to zero over this many input-mask pixels. At zero, the inward gradient reaches zero at the boundary instead."}),
            "curve": (["smoothstep", "linear"], {"default": "smoothstep", "tooltip": "Interior gradient only. Smoothstep eases the transitions; linear gives a straight ramp."}),
            "blur": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1, "tooltip": "Gaussian mode only. Same kernel-size meaning as Essentials Mask Fix blur, using a fast separable Gaussian."}),
        }}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_A", "mask_B")
    FUNCTION = "profile"
    CATEGORY = "MiniMax H3 Experiments/mask"
    DESCRIPTION = "Shape static or animated masks for the H3 masked guider. Outputs complementary A/B weights. Interior gradients use each frame's 0.5 contour, with no temporal filtering or per-frame width normalization. Distances are computed on CPU; outputs retain the input device."

    def profile(self, mask, mode="interior_gradient", transition_width=96.0, outer_fade=24.0, curve="smoothstep", blur=32):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("H3 Mask Profile needs [frames,height,width] or a single [height,width] mask.")
        a = mask.float().clamp(0, 1)
        if mode == "average_inside":
            a = a * .5
        elif mode in ("gaussian", "interior_gradient"):
            frames = a.cpu().numpy()
            result = np.empty_like(frames)
            for i, frame in enumerate(frames):
                if mode == "gaussian":
                    kernel = blur + 1 if blur % 2 == 0 else blur
                    result[i] = ndimage.gaussian_filter(frame, sigma=kernel * .15 + .35,
                                                       radius=kernel // 2, mode="mirror") if blur > 0 else frame
                    continue
                inside = frame >= .5
                if not inside.any() or inside.all():
                    result[i] = inside
                    continue
                edge_weight = .25 if outer_fade > 0 else 0.0
                if transition_width > 0:
                    distance = ndimage.distance_transform_edt(inside).astype(np.float32) - .5
                    interior = edge_weight + (1 - edge_weight) * ramp(distance / transition_width, curve)
                else:
                    interior = 1.0
                if outer_fade > 0:
                    distance = ndimage.distance_transform_edt(~inside).astype(np.float32) - .5
                    exterior = edge_weight * ramp(1 - distance / outer_fade, curve)
                else:
                    exterior = 0.0
                result[i] = np.where(inside, interior, exterior)
            a = torch.from_numpy(result).to(mask.device)
        return (a, 1 - a)
