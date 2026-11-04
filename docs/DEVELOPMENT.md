# Code map

Like ComfyUI's feature-based node modules, implementations are grouped by
function. This smaller pack uses five packages and one explicit registration
entry point. Node schemas and their nearby implementation stay together.

| Location | Responsibility |
| --- | --- |
| [`__init__.py`](../__init__.py) | Stable node IDs, display names, and `WEB_DIRECTORY`. |
| [`masks/`](../masks/) | Spatial mask profiles and preview images. Preserves the incoming frame count. |
| [`conditioning/`](../conditioning/) | Embedding averages, Omni reference blending, and temporal reference curves. |
| [`latents/`](../latents/) | Native AV/video latent compositing and frame-mask conversion to latent groups. |
| [`sampling/`](../sampling/) | Guiders, attention routing/cache, and continuation/checkpoint handling. |
| [`timeline/`](../timeline/) | Virtual head start and generated video/audio position offsets. |
| [`web/`](../web/) | Curve previews and synchronized offset widgets. |
| [`tests/`](../tests/) | Routing, tensor, sampler, persistence, and workflow-compatibility checks. |
| [`example_workflows/`](../example_workflows/) | Shareable workflow JSONs and their bundled mask asset. |

## Sampling experiments

- [`masked_guider.py`](../sampling/masked_guider.py): original prediction blend
  and embedding-overlap branches; averages use `conditioning/overlap_conditioning.py`.
- [`regional_attention.py`](../sampling/regional_attention.py): one joint
  evaluation, source routing, and the per-cycle attention-mask cache.
- [`regional_interaction.py`](../sampling/regional_interaction.py): adds the
  interaction slider to regional routing.
- [`continuation.py`](../sampling/continuation.py): advances mask windows and
  timeline positions, protects the previous AV prefix, samples the new suffix,
  and optionally saves/restores latent checkpoints.

Masks flow from `masks/` through ComfyUI's Conditioning Set Mask into the guider.
The continuation sampler selects each cycle's mask window. The `composite_mask`
helper in `latents/latent_composite.py` converts frame masks into H3 latent
groups; regional routing then pools them to attention patches. The mask cache
reuses routing biases, not model activations.

## Validation

From the ComfyUI checkout, using its verified project interpreter on Windows:

```powershell
& .\.venv\Scripts\python.exe .\custom_nodes\ComfyUI-MiniMaxH3-Experiments\tests\run_tests.py
```

The suite defaults to CPU and uses small native models. For approved GPU tests,
set `H3_EXPERIMENT_TEST_DEVICE` to the intended CUDA device. No model downloads
or full-size renders are needed for import/layout changes.

When moving code, update relative imports and test module paths together. Keep
root registration, node schemas, browser asset paths, and example workflows
stable. Backend changes take effect after ComfyUI restarts; coordinate that
restart with the user after active jobs finish.
