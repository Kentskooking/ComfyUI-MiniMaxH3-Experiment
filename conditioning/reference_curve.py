import math

from comfy.ldm.minimax.model import FRAME_PER_TOKEN


CURVES = ("constant", "sine", "triangle", "linear", "ease_in", "ease_out", "ease_in_out")


def curve_value(time, mode, constant=0.5, start=0.0, end=1.0, cycles=1.0, phase=0.0):
    if mode == "constant":
        return constant
    if mode == "sine":
        value = (1 - math.cos(math.tau * (time * cycles + phase))) / 2
    elif mode == "triangle":
        value = 1 - abs(2 * ((time * cycles + phase) % 1) - 1)
    elif mode == "linear":
        value = time
    elif mode == "ease_in":
        value = (2 ** (10 * time) - 1) / 1023
    elif mode == "ease_out":
        value = 1 - (2 ** (10 * (1 - time)) - 1) / 1023
    elif mode == "ease_in_out":
        value = ((2 ** (20 * time) - 1) / 2046 if time <= 0.5
                 else 1 - (2 ** (20 * (1 - time)) - 1) / 2046)
    else:
        raise ValueError(f"Unknown reference curve: {mode}")
    return start + (end - start) * value


def temporal_curve_weights(latent_t, mode, constant=0.5, start=0.0, end=1.0, cycles=1.0, phase=0.0):
    spans = [FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(latent_t)]
    frames = sum(spans)
    values = [curve_value(i / max(1, frames - 1), mode, constant, start, end, cycles, phase) for i in range(frames)]
    weights = []
    offset = 0
    for span in spans:
        weights.append(sum(values[offset:offset + span]) / span)
        offset += span
    return weights, frames
