from comfy_extras.nodes_minimax_h3 import FPS


class H3VirtualHeadStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING", {"tooltip": "First-frame H3 conditioning, or combined masked branches. Connect before the guider so every branch uses the same head start."}),
            "head_start_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 15.0, "step": 0.25, "display": "slider",
                "tooltip": "Move frame-zero image anchors this many seconds before the generated clip. Zero preserves the input. Output duration, masks, and later keyframes keep their existing timing. This does not generate or skip real frames."}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 Experiments/conditioning"
    DESCRIPTION = "Experimental virtual pre-roll for H3 first-frame conditioning. Changes anchor positions without adding video/audio frames. Place after Conditioning Combine and before the guider. Omni references have no first-frame anchor."

    def apply(self, conditioning, head_start_seconds=5.0):
        if head_start_seconds == 0:
            return (conditioning,)
        offset = head_start_seconds * FPS
        output = []
        shifted = False
        for embedding, metadata in conditioning:
            keyframes = metadata.get("minimax_keyframes", [])
            frames = []
            for frame in keyframes:
                if frame["resolved_frame_index"] == 0 and frame.get("latent") is not None:
                    frame = {**frame, "resolved_frame_index": -offset}
                    shifted = True
                frames.append(frame)
            output.append([embedding, {**metadata, "minimax_keyframes": frames}] if keyframes else [embedding, metadata])
        if not shifted:
            raise ValueError("H3 Virtual Head Start needs a frame-zero image anchor. Connect MiniMax H3 Image to Video with a first frame; Omni references and text-only conditioning have no starting anchor.")
        return (output,)
