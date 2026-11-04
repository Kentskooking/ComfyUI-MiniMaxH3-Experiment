# Working on this pack

Start with [the code map](docs/DEVELOPMENT.md) and [node behavior](docs/NODES.md).
The parent ComfyUI repository's guidance also applies.

- Root `__init__.py` owns registration. Preserve node IDs, display names, input
  order/defaults, and output contracts unless the requested change requires it.
- Keep prediction blending, regional attention, and regional interaction as
  separate experiments so older saved workflows retain their behavior.
- Keep caches scoped to sampling; release them on completion or failure.
  Model changes go through ComfyUI's model patcher.
- Use the verified ComfyUI Python environment for the tests in the code map.
  Do not restart a running ComfyUI instance or queue full-model renders for a
  source-layout change.
- Keep README edits concise. Put implementation guidance in `docs/DEVELOPMENT.md`.
