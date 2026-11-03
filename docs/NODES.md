# Node reference

[Installation and examples](../README.md)

## H3 Regional Attention Guider (Beta)

Use **MiniMax H3 Image to Video** branches, each with **Conditioning Set
Mask**, followed by **Conditioning Combine -> H3 Regional Attention Guider ->
SamplerCustomAdvanced**. Supply one branch's empty AV latent to the sampler.
Supports two, five, or more sources in the same shared evaluation.

- One joint model evaluation at CFG 1 per sampler evaluation. All prompts,
  Qwen visual tokens and image guides enter the same DiT sequence; no separate
  regional predictions or image-latent averaging.
- Masks are normalized across sources at each generated video token. Their log
  weights bias attention to the corresponding source tokens. Zero blocks direct
  source access; gray gives weighted access inside the same softmax. Generated
  video/audio tokens can attend globally, including across region boundaries.
- Each source is refined separately once, then its tokens can read its own
  source bank and the shared generated tokens. Other sources remain
  reachable through that shared evolving video/audio context.
- Audio uses global source strengths. Uncovered video still denoises through
  shared video/audio attention, with no source directly visible there.
- `mask_space=frames` uses the existing 1,4,4,4,4 grouping; `latent` uses one mask
  per latent time step. Masks are also averaged over each spatial 2x2 DiT patch.
  Existing Mask Profile controls, including interior gradients, still apply.
  With multiple colors, connect only each profile's `mask_A` to its own source;
  overlapping fades normalize together. No background branch is required.
- `query_chunk` bounds temporary attention-mask memory; it does not divide the
  video into separate contexts. Every chunk reads the full shared key sequence.
- `mask_cache_mb` reuses unchanged attention-bias chunks across blocks/steps
  within each cycle. Default 256 MiB; 0 disables it for timing comparisons.
  The cache also stays below a quarter of free memory above ComfyUI's inference
  reserve. Chunks beyond the budget are recomputed normally. It clears after
  each cycle or error; moving masks update next cycle. Model attention results
  are never cached, and `query_chunk`/interaction math are unchanged.

The cloned model uses native H3 QKV/norm/RoPE and ComfyUI's mask-capable attention
dispatch. It replaces unmasked attention-forward patches, including KJ's H3
memory-efficient Sage patch. The backend may fall back to PyTorch attention;
one joint forward is not a guarantee of faster sampling. LoRA and preview
patches remain in place. This beta supports I2V/FLF and text-only branches,
not Omni references or independently scheduled/control-hook conditionings.

Verified with routing tests and a tiny native H3 sampler on CPU/4090; full-model
visual quality is unverified. Restart ComfyUI after active jobs finish and
refresh the browser to load the new node.

## H3 Regional Interaction Guider (Beta)

A separate node with the same wiring as Regional Attention Guider, plus a
native `interaction` slider. Existing guider nodes and saved workflows keep
their behavior.

- **1:** exactly the original regional attention routing (default).
- **Lower values:** reduce attention between different mask memberships while
  keeping matching memberships connected across space and time. Try 0.5, then
  0.25 and 0.75 with the same seed.
- **0:** blocks direct attention between disjoint regions. Soft-mask overlaps
  can still bridge them; this is not the old prediction-blending algorithm.

All sources still denoise together in one model evaluation. After the existing
mask conversion, video-to-video attention uses cosine similarity of normalized
source memberships, weighted as `interaction + (1-interaction)*similarity`.
Source tokens use the same rule to read generated video. Uncovered positions
form their own group. Generated audio still reads the whole scene, but its
readback into video/source tokens is multiplied by `interaction` so it cannot
bypass the zero endpoint. Lower settings may also change audio/video coupling.

This follows mask membership, not tracked object identity; harder separation
may bring visible boundaries back. The same bounded `mask_cache_mb` reuse covers
the interaction-bias calculations. No full-model quality guarantee.
Restart ComfyUI after active jobs finish, then refresh to load the new node.

## H3 Continuation Sampler (Beta)

Replace `SamplerCustomAdvanced` after either regional guider. Connect
`full_sequence` to the existing video/audio decoders. Old nodes are unchanged.

- `cycles` counts the initial generation too. Each later cycle protects the
  previous AV latent tail and generates a new suffix. No VAE decode, MP4 save,
  or explicit model unload is requested by the sampler; normal VRAM offloading still applies.
  Bypass separate preview patches to prevent them from decoding during sampling.
- The regular length control sets each window. `context_frames` accepts
  39, 90, 141, ... and snaps down to fit. Final frames are
  `window + (cycles-1)*(window-context)`. Example: 209 / 39 / 2 gives 379 frames.
- Both regions stay in the same joint denoise. Original prompts/images remain
  connected; generated positions advance while original image anchors stay put.
- Masks follow the full timeline: `loop` repeats, `hold` keeps the last frame.
  Short sequences first hold to fill the initial window, matching existing routing.
- Seed is the supplied noise seed plus the zero-based cycle index. The sampler
  reports step progress; final decoding happens downstream.
- One execution reuses its model wrapper, resident-model preparation and fixed
  source text refinement. Moving masks, noise, AV layout and time offsets still
  update each cycle. Evicted models or changed weight patches trigger preparation
  again; extra models retain the normal path. Temporary cached tensors clear on exit/error.
- The report includes sampling, carry/merge and checkpoint times per cycle,
  plus preparation/refinement counts. Their host timings are not GPU profiling.
- With `save_checkpoints`, each completed cycle writes its cumulative raw AV
  tensors and workflow metadata to a unique safetensors file under
  `output/h3_continuation`. Saving remains synchronous. Disk/RAM use grows with sequence length.

To resume, add **H3 Load Continuation State (Beta)**, paste the explicit checkpoint
path from the report/log, and connect `resume_state`. Set `cycles` to the desired
**total**, keeping the model, conditioning, seed, window, masks and context settings
unchanged. A completed count at or above `cycles` only returns the saved sequence.
The checkpoint contains latents, not model weights or encoded conditioning.

Requires native joint H3 AV latents and a schedule ending at zero. One cycle
matches the normal sampler. Continuation needs at least 56 window frames.
This is protected-prefix continuation, not replay of every previous denoising step.
The small-model tests verify carryover, joint attention, save/resume and clock
alignment. Full-model motion/audio seam quality needs a render test; decoding
the combined sequence may also change pixels near joins despite unchanged latents.
Restart after active jobs finish to load these separate beta nodes.

## H3 Virtual Head Start

Connect `Conditioning Combine -> H3 Virtual Head Start -> H3 Masked Conditioning
Guider`. A single I2V branch can use it before Basic Guider. The float field ranges
from 0 to 15 seconds, starts at 5, and uses H3's native 24 fps timebase.
Type a value or drag the native slider; both edit the same setting. You can
also convert the field to a FLOAT input. While connected, the input controls
the value and the slider shows the last executed value without allowing edits.

It moves each frame-zero image anchor into the virtual past: 5 seconds places
it at frame -120 relative to the generated clip. Fractional positions are
retained. Zero is an exact pass-through. Later keyframes keep their positions;
audio attached to a shifted image anchor moves with that anchor.
Only the VAE anchor's temporal position changes; Qwen text/visual embeddings
retain their existing positions.

The existing latent/length controls still determine generated duration. Masks,
prompts, image contents, and generated audio/video length are unchanged. No
extra frames are allocated or sampled, and the node does not crop actual
frames. Use the video loader's skip setting separately if you want to advance
an animated mask.

Use first-frame I2V/FLF conditioning, not Omni references. Apply the node once,
after combining branches so their anchor positions still match for embedding
overlap. Unanchored branches in a mixed list pass through; a nonzero head start
with no frame-zero image anchor reports an error.

This tests whether temporal distance from the starting image changes the
result. It does not reproduce the tail of a longer generation, guarantee weaker
image influence, or reduce sampling time at the same output length.

## H3 Generated Timeline Forward Offset (Beta)

Connect `Model / LoRAs / attention patches -> H3 Generated Timeline Forward
Offset (Beta) -> guider or sampler MODEL input`. The float field is 0-15 seconds,
default 5. The number field and native draggable slider stay synchronized.
You can also convert the field to a FLOAT input; while connected, the slider
shows the last executed value without allowing edits. Zero disables the
offset; if chained, the last node sets the value.

Only generated video/audio temporal coordinates move forward. Text/visual
context, Omni references, and all keyframes (including last frames) stay fixed.
The existing length controls determine output duration. Masks, frame indexing,
latent contents, and denoising steps stay unchanged. Works with native H3
text-to-video, first-frame/FLF, and Omni conditioning.

Use alongside `H3 Virtual Head Start` on the conditioning branch. With both at
5 seconds, the first image sits at -5 and the generated clip begins at +5
relative to the original timeline: a 10-second separation. To move generation
5 seconds past both the original image and text context, use Forward Offset 5
and Virtual Head Start 0.

This uses a cloned model patcher and a private copy of the position coordinates
for each forward call. It adds no frames and does not generate a hidden lead-in
or reproduce the tail of a longer render. It is an experimental positional
change, with no quality guarantee. Restart ComfyUI after active jobs finish,
then refresh the browser to load the new node.

## H3 Omni Reference Blend

Connect two native **MiniMax H3 Reference to Video** conditionings directly to
this node, then connect its conditioning to **Basic Guider**. Each branch has
its own video and prompt, with the video VAE connected. This produces one
conditioning entry before H3 text refinement and inference.

- `context_mix`: blends Qwen's combined text/visual embeddings; 0=A, 1=B.
- `reference_latent_mix`: constant VAE reference ratio; 0=A, 1=B.
- A owns the output token count and modality tags. B is truncated or zero-padded
  to that count. No semantic token or motion alignment is performed. At mix 1,
  context is the length-adjusted B, not necessarily the original full B prompt.
- Match reference slots, dimensions and frame counts before encoding. The supplied
  workflow resizes both clips and fits each to the same length. Audio references
  are disconnected; H3 still generates output audio from the blended context.
- `mix_report` connects to **Preview as Text**. It reports lengths, truncation,
  padding, tag mismatches and actual RMS distances between inputs and output.
  With different inputs, a midpoint should have roughly equal distances to A/B;
  identical inputs have zero distance. This verifies tensor interpolation, not
  whether the model combines concepts coherently.

Use a fixed seed. Compare (context, reference) settings (0,0), (1,1), (.5,0),
(0,.5), and (.5,.5). Both native reference encoders still run at slider endpoints.

### Reference-time curves

The blend node has a live curve preview. `reference_curve=constant`
preserves older workflows and uses `reference_latent_mix`. Animated modes use
`curve_start` and `curve_end` as B weights instead:

- `sine` / `triangle`: start -> end -> start per cycle. `curve_cycles` sets the
  number of cycles across each reference; `curve_phase` shifts them in cycles.
- `linear`: one start-to-end ramp.
- `ease_in`, `ease_out`, `ease_in_out`: normalized exponential start-to-end ramps.
  Cycles and phase apply only to sine/triangle. Swap start/end to reverse a curve.

The preview redraws as controls change without running H3. It shows the ideal
curve over reference time, 0-100%. Execution averages frame values over H3's
repeating 1,4,4,4,4 groups and broadcasts each weight across its latent channels
and spatial positions. The report gives actual first/middle/last and min/max
weights. A one-frame image reference uses the curve's initial value.

Context mixing stays fixed. Reference timing does not pin generated-frame timing;
these curves are not denoising-step schedules. Restart ComfyUI and refresh the
browser after updating to load the controls and inline preview.

## H3 Latent Composite Masked

Search for **H3 Latent Composite Masked**, under **MiniMax H3 Experiments / latent**.
Connect `destination`, `source`, and a `mask`; connect the result to the sampler's latent input.

- Accepts H3 video tensors `[B,24,T,H,W]` and ComfyUI's joint video/audio latents.
- White selects source video, black selects destination, and gray blends them. An unconnected mask selects the whole source video.
- Keeps destination audio and metadata. Inputs are not modified.
- Inputs use the same canvas and timeline. Singleton source axes can broadcast; this node does not resize or retime source videos.
- `mask_space=frames` averages moving masks over H3's repeating `1,4,4,4,4` frame groups. `latent` accepts one mask per latent step. A single mask repeats across the clip. Short mask sequences hold their last frame; excess frames are cropped. Spatial masks resize to the latent canvas without cropping.
- `noise_mask_mode=blend` also composites the input video denoise masks, treating an absent mask as all-generate. Destination audio protection is retained. `destination` keeps its existing mask, while `none` removes denoise masks.

The composite mask selects **content**. Denoise masks control **regeneration**, where zero preserves and one generates. These are independent controls.

`MiniMax H3 Image to Video` produces an empty starting latent; its frame images travel through conditioning. To composite image/video content, supply encoded or sampled latents. For regional conditioning, use the guider below.

## H3 Masked Conditioning Guider

Replace **Basic Guider** with **H3 Masked Conditioning Guider**, under **MiniMax H3 Experiments / sampling**. Keep the existing `Conditioning Set Mask -> Conditioning Combine -> conditioning` connections, and connect its `GUIDER` output to `SamplerCustomAdvanced`.

- Converts conditioning masks to H3's packed video/audio sampler layout, fixing the `tuple index out of range` error in `resolve_areas_and_cond_masks_multidim`.
- Evaluates each conditioning entry separately against the same full-scene latent at each step, then blends the video predictions with normalized mask weights. This preserves H3's batch-size-one requirement and each branch's references/keyframes.
- Uses CFG 1, like Basic Guider. Two active entries require two model evaluations per sampler evaluation.
- Soft masks blend overlapping regions. An unmasked entry applies globally. Regions with no active mask coverage receive no denoising update; use complementary masks or a global branch to cover the canvas.
- Audio predictions are averaged by conditioning strength and mask strength, independently of the spatial mask. An all-black spatial mask still contributes to audio. To preserve existing audio, use the separate AV denoise mask.
- `mask_space` uses the same static/moving-mask rules as the composite node above. Conditioning start/end percentages remain supported.
- Always evaluates the full canvas, including when `set_cond_area=mask bounds` is selected. Explicit `Conditioning Set Area` crops are unsupported; use masks instead.

This is experimental blending of model predictions, not spatial attention routing inside H3. References can influence a branch's whole scene before its output is masked. Visual quality and temporal consistency require generation tests.

### Embedding overlap mode

Set `blend_mode=embedding_overlap` on the same guider, keeping the separate
`Conditioning Set Mask -> Conditioning Combine` branches. The default
`predictions` mode retains the original behavior for existing workflows.

- Builds a 50/50 embedding conditioning **before H3 text refinement and inference**. Supports text-only branches, matching first/last-frame branches, and matching omni image-reference slots. Matching keyframe video/audio latents and image-reference latents are averaged too. Video/audio reference lists require `predictions` mode.
- `overlap_mix` controls the overlap behavior from 0 to 1: 0 uses the original prediction blend, 1 uses full embedding overlap, and intermediate values interpolate between them. Default 1 preserves existing embedding-overlap workflows. It affects generated audio with two branches; three or more retain the original strength-weighted audio average. It is ignored in `predictions` mode.
- With two effective mask weights `a` and `b`, let `s=overlap_mix*min(a,b)`. It uses prediction weights `a-s` for A, `b-s` for B, and `2s` for the pre-blended branch, then divides by `a+b`. One-sided coverage retains full strength; at `overlap_mix=1`, equal overlap uses only the embedding-blended prediction. Uncovered video regions still receive no denoising update.
- Unequal soft weights interpolate the three predictions. This is a three-branch approximation, not a different embedding per video pixel inside a single attention pass. A hard boundary with no shared coverage has no embedding-blended region; feather or overlap the masks to create one.
- Moving masks keep the same H3 temporal mapping. With two branches, audio uses the same branch calculation with global branch strengths, independently of spatial coverage. Separate AV denoise masks still protect existing latent content.
- Keeps all tokens by zero-padding the shorter embedding on its own device/dtype; modality tags must match. No semantic token alignment is performed. Keyframe positions and latent shapes must match.
- Two branches require up to three full-canvas model evaluations per sampler evaluation. Zero-weight contributions are skipped. Original conditionings and masks are not modified.

For omni image references, use a Ref2VA model and its matching acceleration LoRA.
Each branch must supply the same number of image references at matching sizes,
with the video VAE connected. Reference slots are averaged in order, producing
one midpoint reference per pair of slots, rather than concatenating two lists.
This is experimental interpolation of reference conditioning, not a trained
concept-merging feature. No keyframe is added. The original A/B references remain
intact outside the overlap; `overlap_mix=0` compares ordinary prediction blending.

For three or more entries, the guider prepares pairwise averaged conditionings.
At each video location, total weight `W` allocates a shared amount
`s_ij=overlap_mix*w_i*w_j/(W-min(w_i,w_j))` to pair `i,j` (zero when the denominator
is zero). Each pair gets prediction weight `2*s_ij`, and that shared amount is
subtracted from both original branches. Nonnegative mask weights retain
nonnegative residuals and preserve total coverage. If only two regions are
active, this reduces to the existing two-region formula. Three-way intersections
combine pairwise predictions; they do not use one three-way averaged embedding.

Only pairs with nonzero video coverage run inference. Multi-region audio uses
the original branches so it does not activate unrelated pairs. Worst-case cost
is `N + N*(N-1)/2` model evaluations per sampler evaluation; pairwise conditioning
preprocessing still occurs for all pairs. Five non-overlapping regions require
five evaluations; soft boundaries add the pairs that overlap anywhere in the clip.

## H3 Mask Profile

Connect a static or animated `MASK` to **H3 Mask Profile**, then its `mask_A` and
`mask_B` outputs to the corresponding Conditioning Set Mask nodes. The outputs
sum to one. Replace the old blur/invert pair with this node to switch modes
without rewiring:

- `original`: preserves the input mask weights.
- `gaussian`: Essentials-style blur, using a separable Gaussian.
- `interior_gradient`: a pure-A core with a distance-based transition inside the
  white region and an optional tail outside it. `transition_width` is the inward
  distance to pure A; `outer_fade` is the outward distance to zero. Both use input
  mask pixels, not output-video pixels. The boundary weight is 0.25 when an outer
  fade is enabled; with no outer fade, the inward ramp reaches zero there.
- `average_inside`: white becomes A=0.5/B=0.5, black becomes A=0/B=1. With equal
  branch strengths, `embedding_overlap`, and `overlap_mix=1`, this selects the
  averaged-conditioning prediction inside the band and B outside.

Interior gradients threshold the input at 0.5 and use `smoothstep` or `linear`
falloff. Fixed distances avoid changing the gradient scale as a moving band
enters or leaves the canvas. Entirely white/black frames remain white/black;
canvas edges do not introduce artificial fading. Processing is spatial per
frame; the guider retains its normal H3 temporal mapping. Distance transforms
and Gaussian filtering run on CPU, returning float32 masks on the input device.

## H3 Mask Profile Preview

Connect the mask **before** H3 Mask Profile to **H3 Mask Profile Preview**. Choose
`frame_index` (zero is the first incoming frame) and adjust the same profile
controls. The node processes only that frame and displays original, altered A,
and complementary B side by side, at the input mask resolution. An index past
the end holds the last frame. Preview images are temporary; the `comparison`
image output can connect to Save Image if desired.

Use the standalone preview workflow and enable ComfyUI's auto queue on change
for updates after edits, or click Run. It needs no H3 models or sampler. Keep
the loader resolution and frame sampling identical to the generation workflow
when choosing widths. Copy the settings to H3 Mask Profile afterward. The
preview shows spatial mask weights before the guider's latent resizing, temporal
grouping, and conditioning-strength normalization.
