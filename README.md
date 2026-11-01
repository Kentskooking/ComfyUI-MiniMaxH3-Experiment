# ComfyUI MiniMax H3 Experiments

Experimental nodes for masked conditioning, H3 latent compositing, mask-profile
previews, and two-reference blending with reference-time curves.

The node IDs and existing blend defaults are preserved. This pack does not
install models, modify ComfyUI core, or require the separate Motion Context
MultiRef pack. Blending is experimental: interpolation can produce double
exposures or distortion rather than a coherent combination of concepts.

## Installation

1. Use a ComfyUI installation with native MiniMax H3 support: the Image to Video,
   Reference to Video, Sigma Shift, and joint audio/video latent implementation.
   The checks for this release used ComfyUI 0.42.0 with those APIs present.
2. Extract this folder directly into `ComfyUI/custom_nodes/`. The layout should be
   `custom_nodes/ComfyUI-MiniMaxH3-Experiments/__init__.py`, with `web/` beside it.
3. Use the same Python environment as ComfyUI. The dependencies are NumPy, SciPy,
   Pillow and ComfyUI's existing PyTorch. They are normally already installed;
   `requirements.txt` lists the additional package requirements for installers.
4. Restart ComfyUI, then refresh the browser. Search for **H3** or browse the
   **MiniMax H3 Experiments** node category.

Alternatively, run this from your `ComfyUI/custom_nodes` directory:

```sh
git clone https://github.com/Kentskooking/ComfyUI-MiniMaxH3-Experiment.git ComfyUI-MiniMaxH3-Experiments
```

If dependencies are missing, install `requirements.txt` using your ComfyUI
environment's Python (`-m pip install -r <pack>/requirements.txt`). Do not create
a second environment or replace your working PyTorch/CUDA build for this pack.

## Start with the concentric example

Open [the concentric red/green canvas workflow](example_workflows/MiniMax_H3_Concentric_RedGreen_Canvas_8step.json).

1. Upload the included `example_workflows/assets/Concentric_Center_Circle_RedGreen_150f.mp4`
   using the mask video's **choose video to upload** button. Alternatively, copy
   it into your ComfyUI `input` folder and select it in the video dropdown.
2. Select your **Canvas**, **Red**, and **Green** images. These three inputs are
   intentionally empty. Each branch has a basic editable prompt.
3. Select locally installed model files if their names differ from the example.
   Keep the FL2VA model and its matching FL2V eight-step LoRA together.
4. Run with a fixed seed when comparing profile modes and overlap settings.

This example uses a global canvas at strength 0.65 plus two animated color
regions. It retains `average_inside`, `embedding_overlap`, and `overlap_mix=1`.
The 512 x 512 output uses all 150 mask frames at 24 fps, then holds the last mask
for eight frames to reach H3's 158-frame grid. There is no upscaling.

For inward/outward fades, select `interior_gradient` on each color's profile.
The width/fade/curve controls affect that mode; blur affects `gaussian`.
`average_inside` ignores those controls. The two color profiles use their A
outputs; the global canvas supplies shared context everywhere.

## Examples and dependencies

| Example | Purpose |
| --- | --- |
| [Concentric canvas](example_workflows/MiniMax_H3_Concentric_RedGreen_Canvas_8step.json) | Base image plus red and green masked image conditioning. |
| [Mask profile preview](example_workflows/MiniMax_H3_Concentric_Mask_Preview.json) | Inspect any frame of the included mask, without loading an H3 model. |
| [Omni video blend](example_workflows/MiniMax_H3_Omni_Video_Blend_NVFP4_8step.json) | Mix two videos' context embeddings and reference latents. Supply both clips. |
| [Omni reference curve](example_workflows/MiniMax_H3_Omni_Video_Blend_NVFP4_8step-reference-curve.json) | The same blend with a sine curve over reference time and a live curve preview. |

Generation examples also use **ComfyUI-VideoHelperSuite** (video loading),
**comfyui_essentials** (Image Expand Batch), and **ComfyUI-KJNodes** (SageAttention,
tiny-VAE preview, and color extraction). The mask preview needs VideoHelperSuite
and KJNodes. These are example dependencies, not dependencies of the five nodes
in this pack. FFmpeg must be available to VideoHelperSuite.

The generation examples enable KJNodes' **MiniMax H3 Memory Efficient Sage
Attention Patch**. This requires a working SageAttention/Triton setup. If your
installation does not support it, connect Sigma Shift's model output directly
to Model Preview Override, or bypass both and feed the model to the scheduler
and guider. Model Preview Override can also be bypassed if the tiny VAE is absent.

Models are not bundled. The concentric example selects an FL2VA INT8 model;
the Omni examples select a Ref2VA NVFP4 model. Both use the H3 video/audio VAEs,
the H3 Qwen3-VL text encoder, the matching eight-step LoRA, and optionally
`taeh3.safetensors` for previews. NVFP4 requires compatible hardware and ComfyUI
support. See [the example guide](example_workflows/README.md) for file selections.

## Nodes

| Node | Role |
| --- | --- |
| H3 Masked Conditioning Guider | Mask-weighted predictions or pairwise embedding-overlap blending. |
| H3 Mask Profile | Original, Gaussian, interior-gradient, or average-inside mask weights. |
| H3 Mask Profile Preview | Preview one mask frame with the exact profile calculation. |
| H3 Omni Reference Blend | Mix text/visual embeddings and image/video reference latents. |
| H3 Latent Composite Masked | Composite encoded or sampled H3 video latents while retaining destination audio. |

See the [node reference](docs/NODES.md) for mask semantics, blend math, audio behavior,
reference curves, and supported inputs.

## Development and verification

Version 0.1.0: 77 CPU tests passed; all four examples passed native prompt
validation with temporary media substitutions. The bundled mask was loaded and
the standalone profile preview executed. No pretrained generation was run.

Run the CPU checks from your ComfyUI root with its Python executable:

```powershell
# Windows example for a .venv installation
.\.venv\Scripts\python.exe custom_nodes/ComfyUI-MiniMaxH3-Experiments/tests/run_tests.py
```

```sh
# Linux example for a .venv installation
./.venv/bin/python custom_nodes/ComfyUI-MiniMaxH3-Experiments/tests/run_tests.py
```

The checks cover native H3 latent layout and sampler integration with synthetic
model outputs, audio retention, mask timing, profile math, reference blending,
and curve weights. They require the pack to be installed under `custom_nodes`
in an H3-capable ComfyUI checkout. They do not download weights or run a
pretrained generation. Only Windows CPU execution has been checked for this
sharing bundle; other platforms are unverified.

Native workflow validation uses temporary input substitutions because the
distributed image/video selections are intentionally empty. Passing the tests
and graph checks establishes implementation behavior, not generation quality
or universal model/backend compatibility.

## License

[MIT](LICENSE), copyright 2026 kentskooking. ComfyUI, other custom-node packs,
and model weights retain their own licenses and are not included in this pack.
