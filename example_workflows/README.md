# Example workflows

Install the pack and the example dependencies listed in the main README. Import
the JSON into ComfyUI, select your media, then review the model selections. No
model weights or personal reference images/videos are included.

## Concentric red/green canvas

`MiniMax_H3_Concentric_RedGreen_Canvas_8step.json` contains three image branches:
a global canvas and two regions extracted from the included red/green clip.
Upload `assets/Concentric_Center_Circle_RedGreen_150f.mp4` in the video loader,
or copy that file into `ComfyUI/input/` and select it there. Select three images
and edit the three basic prompts.

The clip is 1024 x 1024, 150 frames, 24 fps. The loader processes it at 512 x 512;
the workflow holds its last frame for eight frames to match the 158-frame H3
output. It skips no frames. The source clip is included byte-for-byte.

Canvas strength is 0.65. Both color profiles start at `average_inside`, using
their A outputs (0.5 inside, zero outside); the global canvas supplies context
throughout the video. This is not the same as wiring a single profile's
complementary A/B outputs to two equal-strength branches.

Use `interior_gradient` for an adjustable pure-color core with inward and
outward fades. Widths are measured in the 512 x 512 mask, independently of the
output resolution. `average_inside` ignores transition width, outer fade,
smoothstep and blur; `gaussian` uses the blur control.

## Mask preview

`MiniMax_H3_Concentric_Mask_Preview.json` needs no models. Upload the same mask,
choose `frame_index`, adjust the profile controls and Run. Red is selected by
default; change Color to Mask's RGB to `0, 255, 0` to inspect green.

The preview starts in `interior_gradient` to demonstrate the width controls.
Select `average_inside` to match the generation example. Copy the chosen
settings to its profiles; the two workflows do not synchronize automatically.

## Omni video blending

The two Omni examples have empty A/B upload controls and neutral editable
prompts. Both reference videos are resized to 768 x 768, start from frame zero,
and use up to 124 frames at 24 fps. Short clips hold their last frame. The output
canvas is 1024 x 1024, 124 frames. No reference audio is connected.

- `MiniMax_H3_Omni_Video_Blend_NVFP4_8step.json` starts at constant 0.5 context
  and 0.5 reference ratios.
- `MiniMax_H3_Omni_Video_Blend_NVFP4_8step-reference-curve.json` starts at 0.5
  context with a reference sine curve from A to B and back to A over one cycle.

Reference-time curves change the supplied reference tensor along time. They do
not guarantee that generated events occur at the same times. Neither example
performs semantic token alignment or motion matching between input clips.

## Model selections

These are the filenames saved in the examples. Select equivalent files from
your installation when names differ. Keep task families and acceleration
LoRAs matched; the examples do not download missing models.

| Use | ComfyUI folder | Selected filename |
| --- | --- | --- |
| Concentric FL2VA model | `models/diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Concentric eight-step LoRA | `models/loras` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| Omni Ref2VA model | `models/diffusion_models` | `minimax_h3_ref2va_pruned_nvfp4.safetensors` |
| Omni eight-step LoRA | `models/loras` | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| H3 Qwen3-VL encoder | `models/text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Audio VAE | `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| Optional tiny VAE | `models/vae_approx` | `taeh3.safetensors` |

Generation uses Euler/simple, eight steps, sigma shifts 12/3 and CFG 1.
SageAttention/Triton and the tiny-VAE preview need compatible installations;
the main README explains how to bypass those optional model patches.
