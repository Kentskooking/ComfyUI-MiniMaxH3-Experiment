import copy
from functools import partial

from comfy.ldm.minimax.model import FRAME_RESCALE, PackedLayout
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_minimax_h3 import FPS


def offset_generated_timeline(executor, x, timestep, context, transformer_options, minimax_payload=None, *, offset, **kwargs):
    layout = (minimax_payload or {}).get("layout")
    if not isinstance(layout, PackedLayout):
        raise ValueError("H3 Generated Timeline Forward Offset needs a native H3 model with prepared audio/video conditioning.")

    shifted = copy.copy(layout)
    shifted.position_ids = layout.position_ids.clone()
    for start, stop, kind in layout.segments:
        if kind in ("video", "audio"):
            shifted.position_ids[start:stop, 0] += offset

    return executor(x, timestep, context, transformer_options.copy(),
                    minimax_payload={**minimax_payload, "layout": shifted}, **kwargs)


class H3GeneratedTimelineOffset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Connect after LoRAs and attention patches, before the guider or sampler."}),
            "forward_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 15.0, "step": 0.25, "display": "number",
                "tooltip": "Move generated video/audio positions forward relative to fixed text, visual context, and image anchors. Zero disables this patch. With Virtual Head Start, the two offsets add to the first-image separation. Duration and masks stay unchanged."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3 Experiments/model"
    DESCRIPTION = "Beta: offset only generated video/audio temporal positions. Keeps text, Omni references, and all keyframes fixed. No extra frames or skipped denoising steps. Can combine with H3 Virtual Head Start. If chained, the last Forward Offset node sets the offset."

    def execute(self, model, forward_seconds=5.0):
        return {"ui": {"h3_offset_seconds": [forward_seconds]}, "result": self.patch(model, forward_seconds)}

    def patch(self, model, forward_seconds=5.0):
        patched = model.clone()
        key = "h3_experiments_generated_timeline_offset"
        patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, key)
        if forward_seconds != 0:
            wrapper = partial(offset_generated_timeline, offset=forward_seconds * FPS * FRAME_RESCALE)
            patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, wrapper)
        return (patched,)
