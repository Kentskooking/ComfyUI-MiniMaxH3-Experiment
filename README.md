# MiniMax H3 Experiments

Masked conditioning, latent blending, mask previews, and animated reference mixing for ComfyUI.

Requires native MiniMax H3 support. Clone into `ComfyUI/custom_nodes`:

```sh
git clone https://github.com/Kentskooking/ComfyUI-MiniMaxH3-Experiment.git ComfyUI-MiniMaxH3-Experiments
```

Install `requirements.txt` in ComfyUI's Python environment if needed, then restart ComfyUI and refresh the browser.

Start with the [concentric canvas workflow](example_workflows/MiniMax_H3_Concentric_RedGreen_Canvas_8step.json): upload the included mask, choose Canvas/Red/Green images, and edit the prompts.

Examples require VideoHelperSuite, KJNodes, and Essentials. Models are not included.

[Example setup](example_workflows/README.md) · [Node reference](docs/NODES.md) · [MIT license](LICENSE)
