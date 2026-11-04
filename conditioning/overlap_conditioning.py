import torch
import torch.nn.functional as F


def average_conditioning(a, b):
    """Midpoint of H3 embeddings and matching keyframe or image-reference latents."""
    x, ma = a
    y, mb = b
    for key in ("control", "hooks", "gligen", "model_conds"):
        if ma.get(key) is not None or mb.get(key) is not None:
            raise ValueError(f"Embedding overlap cannot merge branch-specific {key}; use predictions mode.")

    length = max(x.shape[1], y.shape[1])
    tags = []
    for embedding, metadata in ((x, ma), (y, mb)):
        tag = metadata.get("minimax_token_tags")
        if tag is None:
            tag = torch.ones(embedding.shape[1], dtype=torch.long, device=x.device)
        tag = tag.flatten().to(device=x.device)
        if tag.numel() != embedding.shape[1]:
            raise ValueError("H3 token tags must match their conditioning token count.")
        tags.append(F.pad(tag, (0, length - tag.numel()), value=1))
    if not torch.equal(tags[0], tags[1]):
        raise ValueError("Embedding overlap needs matching image-token positions in A and B. Use the same number and dimensions of reference/keyframe images, or text-only conditioning.")

    y = y.to(x)
    if x.shape[1] < length:
        x = F.pad(x, (0, 0, 0, length - x.shape[1]))
    if y.shape[1] < length:
        y = F.pad(y, (0, 0, 0, length - y.shape[1]))
    metadata = ma.copy()
    for key in ("mask", "mask_strength", "strength", "area", "set_area_to_bounds", "start_percent", "end_percent"):
        metadata.pop(key, None)
    metadata["minimax_token_tags"] = tags[0]
    metadata.pop("pooled_output", None)
    if ma.get("pooled_output") is not None and mb.get("pooled_output") is not None:
        metadata["pooled_output"] = torch.lerp(ma["pooled_output"], mb["pooled_output"].to(ma["pooled_output"]), .5)

    refs_a = ma.get("minimax_refs", [])
    refs_b = mb.get("minimax_refs", [])
    if len(refs_a) != len(refs_b):
        raise ValueError("Embedding overlap needs matching image-reference slots in A and B.")
    blended_refs = []
    for ra, rb in zip(refs_a, refs_b):
        if ra["kind"] != "image" or rb["kind"] != "image":
            raise ValueError("Embedding overlap supports image references only; use predictions mode for video/audio references.")
        za, zb = ra["latent"], rb["latent"]
        if za.shape != zb.shape or (ra["latent_h"], ra["latent_w"]) != (rb["latent_h"], rb["latent_w"]):
            raise ValueError("Embedding overlap needs matching image-reference sizes. Resize both references to the same dimensions before encoding.")
        blended = ra.copy()
        blended["latent"] = torch.lerp(za, zb.to(za), .5)
        blended_refs.append(blended)
    if blended_refs:
        metadata["minimax_refs"] = blended_refs

    keyframes_a = ma.get("minimax_keyframes", [])
    keyframes_b = mb.get("minimax_keyframes", [])
    if len(keyframes_a) != len(keyframes_b):
        raise ValueError("Embedding overlap needs the same keyframe positions in A and B.")
    blended_frames = []
    for ka, kb in zip(keyframes_a, keyframes_b):
        if ka["resolved_frame_index"] != kb["resolved_frame_index"]:
            raise ValueError("Embedding overlap needs the same keyframe positions in A and B.")
        blended = ka.copy()
        for key in ("latent", "audio_latent"):
            za, zb = ka.get(key), kb.get(key)
            if za is None and zb is None:
                continue
            if za is None or zb is None or za.shape != zb.shape:
                raise ValueError(f"Embedding overlap needs matching {key} shapes at each keyframe.")
            blended[key] = torch.lerp(za, zb.to(za), .5)
        blended_frames.append(blended)
    if blended_frames:
        metadata["minimax_keyframes"] = blended_frames
    return [torch.lerp(x, y, .5), metadata]
