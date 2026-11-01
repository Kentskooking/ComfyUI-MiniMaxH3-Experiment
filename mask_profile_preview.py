import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import nodes
from .mask_profile import H3MaskProfile


class H3MaskProfilePreview(nodes.PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        profile_inputs = H3MaskProfile.INPUT_TYPES()["required"]
        mask_input = profile_inputs.pop("mask")
        return {"required": {
            "mask": mask_input,
            "frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1, "tooltip": "Zero-based index in the incoming mask batch. Values beyond the clip use its last frame. Only this frame is processed."}),
            **profile_inputs,
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_NAMES = ("comparison",)
    FUNCTION = "preview"
    CATEGORY = "MiniMax H3 Experiments/mask"
    DESCRIPTION = "Preview one mask frame with the exact H3 Mask Profile calculation. Shows original, mask A and mask B. Run this output alone, or use a standalone preview workflow with auto queue on change; no H3 model is needed."

    def preview(self, mask, frame_index=0, mode="interior_gradient", transition_width=96.0,
                outer_fade=24.0, curve="smoothstep", blur=32, prompt=None, extra_pnginfo=None):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] == 0:
            raise ValueError("H3 Mask Profile Preview needs a nonempty [frames,height,width] or [height,width] mask.")
        index = min(max(frame_index, 0), mask.shape[0] - 1)
        frame = mask[index:index + 1].float().clamp(0, 1).cpu()
        a, b = H3MaskProfile().profile(frame, mode, transition_width, outer_fade, curve, blur)
        height, width = frame.shape[-2:]
        comparison = Image.new("RGB", (width * 3, height + 64), "#202020")
        draw = ImageDraw.Draw(comparison)
        font = ImageFont.load_default(size=16)
        for column, (title, values) in enumerate((("Original", frame), ("A - altered", a), ("B - complement", b))):
            pixels = (values[0].numpy() * 255).round().astype(np.uint8)
            comparison.paste(Image.fromarray(pixels).convert("RGB"), (column * width, 32))
            draw.text((column * width + 8, 6), title, font=font, fill="white")
        draw.text((8, height + 40), f"Frame {index + 1}/{mask.shape[0]} (index {index}) | {width} x {height} | {mode} | Black = 0, white = 1", font=font, fill="white")
        image = torch.from_numpy(np.asarray(comparison).copy()).float().unsqueeze(0) / 255
        return self.save_images(image, filename_prefix="H3_Mask_Profile", prompt=prompt, extra_pnginfo=extra_pnginfo)
