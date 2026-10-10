# Original Google E4B multimodal baseline

## Scope

Complete `google/gemma-4-E4B-it` uncompressed BF16 inference in vllm.cpp for text, tool calls, images, audio input, and video frames. Audio/video generation is not implied. Preserve PLE, KV sharing, partial RoPE, sliding/full attention, softcaps, tokenizer behavior, and trained clipping semantics. Do not substitute another runtime, community checkpoint, quantized checkpoint, or preprojected embeddings for acceptance. No LittleBit-2, training, or compression implementation belongs in this milestone. Later LittleBit-2 must preserve every modality and tool behavior.

## Ownership and integration

Owner: `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`.
Existing canonical local record: `.agents/issues/MODEL-MM-gemma4-mm-gemma4-for-conditional-generation/ISSUE-GH-2173.md`. It is `State: UNKNOWN`, `Availability: METADATA_ONLY`, not proof of an open actionable issue. It covers vision, not this complete milestone.
Canonical milestone issue: [ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8](../issues/MODEL-MM-gemma4-mm-gemma4-for-conditional-generation/ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8.md), OPEN. The historical record remains UNKNOWN/METADATA_ONLY. No remote state is inferred.
Historical designs: [Gemma-4 multimodal](gemma4-multimodal.md) and [vision reachability](gemma4-vision-reachability.md). Their captures remain historical evidence, not acceptance for the original Google checkpoint.

Shared seam owner: `ENG-MM-INPUT-PIPELINE`. Historical `ISSUE-GH-2300` remains UNKNOWN/METADATA_ONLY. Commit `c0649889d` added the tower and merge runner seam. Reuse and test that seam. Do not create a parallel runner or take ownership of shared changes without coordination.

## Now

T00 establishes the local issue and canonical design before product implementation. Independent review and the operator rerun remain pending. No original-checkpoint inference or product implementation is claimed.

The owning matrix lifecycle remains READY. The coordination audit dated 2026-08-06 retired the old Gemma claims and moved this row from ACTIVE to READY. A stale ACTIVE token in its historical evidence made the derived live view report ACTIVE. The scoped metadata correction removes that ambiguity without changing historical test results. The local helper role owns only this bounded records task on `row/MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`.

## Git integration

One PR is the repository default. No PR, push, merge, or remote write is authorized for T00. Commit this canonical design before product code. Planning artifacts and run state remain untracked.

## Upstream chain

Inspected local base: `6c43232260b300e6992f9375811f782fe24c172a`.
Google revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`, independently confirmed by the public HF revision API. API reports ungated, 7,996,156,490 BF16 parameters, one `model.safetensors` file. Weight bytes were not downloaded or hashed; theoretical parameter storage is 15,992,312,980 bytes, not measured runtime residency.
The planning audit captured small public metadata files. The source is [the pinned Google checkpoint](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2). Config SHA256 `33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4`; processor SHA256 `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c`.

Text config: 42 layers, width 2560, PLE width 256, 18 KV-shared layers, 8 query heads, 2 KV heads, local/global dimensions 256/512, vocabulary 262144, sliding window 512. Vision: 16 layers, width 768, patch 16, pooling 3, 280 image soft tokens. Audio: 12 layers, width 1024, projection 1536, trained clipped linears. Frontend: 16 kHz, 128 features, FFT 512, frame 320, hop 160, mel floor 0.001. Video processor declares 32 frames and 70 maximum soft tokens, distinct from image defaults. Do not collapse video processing into repeated default image processing.

Primary oracle: vLLM `a7c23ac96d7806e7c7e7d862eadbce5a33529b94` from `.agents/upstream-sync.md:101`; runtime/distribution `0.3.0.dev267+ga7c23ac96d`, FlashInfer `0.6.15.post1`. Fetched `vllm/model_executor/models/gemma4_mm.py`: `Gemma4ProcessingInfo` at 196, `Gemma4MultiModalProcessor` at 580, `_get_mm_fields_config` at 784, `Gemma4MultimodalEmbedder` at 950, `Gemma4ForConditionalGeneration` at 1015, `_process_image_input` at 1299, `_process_audio_input` at 1581, `embed_multimodal` at 1617. These anchors identify the source to port, not proof that the oracle runs this exact checkpoint.
Secondary preprocessing oracle: transformers 5.14.1 per `.agents/oracles/transformers.md:39-40`. Its record refers to an older vLLM environment. The public v5.14.1 annotated tag resolves to commit `a08ace4bbd97e721c98751deec37d87b026acadc` (tag object `ec710763db3fd53921a54c6eccf693619abbe866`), verified with git ls-remote. The processor source at that commit is captured in the source manifest. T01 must confirm compatibility with the current primary before fixture capture. Do not silently update the global oracle pin. Exact-Google gateability remains PENDING.

## Our baseline

`src/vllm/model_executor/models/gemma4_registry.cpp::kGemma4Factory` at 171-178 lacks both MM hooks. `src/vllm/model_executor/models/model_registry.cpp::ModelRegistry::SupportsMmInputs` at 781-783 requires both. Existing registered `ForwardMm` is not an input pipeline. `include/vllm/model_executor/models/gemma4_audio.h:37-42` explicitly declares host-F32 reference math and leaves production BF16/device execution owed.
The audit child scanned 1110 production files and found no production tower invocation or Gemma processor registration at this SHA. Root independently verified only the three critical source files: gemma4_registry.cpp, model_registry.cpp, and gemma4_audio.h. These are static findings, not executed tests. Direct Gemma MatmulBT calls remain; inventory their dtypes and shared-seam obligations without changing model math or starting a performance rewrite.

Design: model loader retains all checkpoint tower/projector weights; model processor expands modality placeholders and emits validated typed fields; factory encode/embed hooks invoke actual towers and merge in original sequence order through the existing runner. Public ABI and server route raw inputs into that chain. Bound encoder-cache lifetime, request ownership, cancellation, batch isolation, and placeholder count. Reject unsupported inputs explicitly, never fall back to text or silently drop a modality. Preserve text-only behavior and PLE token identities at MM positions.

## Gates

Implement serially: metadata and strict validation, loader accounting, chat and tools, image processing, audio processing, video processing, vision encoding, audio encoding, hook and runner wiring, public serving, strict gate harness, exact-checkpoint reference validation, then production reachability. Separate three evidence levels:
1. CPU/offline: config parsing, original template/tool tokens, processor numerics, masks, counts, loader accounting, synthetic-shape tower math, and request routing. No runtime inference claim follows from this level.
2. Production reachability: raw image/audio/video-frame requests through include/vllm.h and serving, invoking the real processor, tower, hook, runner, and decode path. Independently delete each production call site in scratch; the gate must fail. Tiny fixtures can establish routing but not exact full-checkpoint inference.
3. Exact-checkpoint reference: both actual engines load the pinned original Google artifact, verify every tensor and artifact hash, run identical prompts/media/sampling, and compare intermediate stages plus generated token IDs. Require text/tool/image/audio/video and mixed-input cases, prefill/decode, streaming/nonstreaming tools, batched isolation, and text regression. Require zero token divergence by default. Any stochastic/distributional exception needs separate prior ratification, not the legacy permissive image predicate.

Before collecting results, freeze processor/tower tolerances from the pinned upstream tests or an independently measured oracle repeatability envelope. Record exact values and justification. Missing thresholds block capture acceptance; do not choose thresholds after seeing C++ errors. Integer tokens, masks, lengths, placeholder order, and positions require exact identity. Require finite numeric outputs and explicit per-stage max-absolute and relative error bounds.

Every required gate fails on absent assets, identity mismatch, zero executed cases, skipped cases, nonfinite outputs, or mismatch. Existing Unsloth captures and skip-on-no-checkpoint tests remain regression evidence only. Report per modality executed/passed/failed/skipped counts; a process exit 0 alone is insufficient.

## Dependencies

No hardware or download authority is granted here. Root obtains approved checkpoint staging, fixture provenance, environment values, sufficient disk/RAM/VRAM, an idle-device lease, and required mutex before full acceptance. No host names or secrets belong in deliverables. Record build/runtime identities and resolved dtypes. CPU builds require CMake >=3.24, C++20, Python3, and available third-party headers. Network-disabled configure must fail rather than fetch dependencies silently.
Missing environment does not block records or CPU unit design. Oracle pin compatibility, float normalization, video timestamps, audio padding, device memory residency, and stale historical ownership are explicit risks. No claim of speed parity or complete port is allowed while applicable repository performance axes remain pending. This milestone is prerequisite evidence for LittleBit-2, not a waiver for quantized arms: preserve existing arms and record new compression work as deferred.

## Risks/decisions and review

Root dispatches one bounded task at a time. A fresh independent reviewer checks each immutable implementation head and mutates critical guarantees in scratch. Root reruns each Verify command, then requires a separate independent final reviewer for the whole baseline. Findings return to a fresh fixer; never lower a gate or discard correct work. Current evidence: static inspection and metadata audit only. No product implementation, original-checkpoint inference, or accepted runtime evidence exists for this milestone.

## T00 verification and pending gates

Run after the canonical spec commit:

```sh
python3 scripts/check-agent-record.py && python3 scripts/check-symbol-anchors.py && git cat-file -e HEAD:.agents/specs/gemma4-e4b-google-baseline.md
```

The original setup preflight (bg-51) exited 1 with `tools suites` failing. The later pre-edit run (bg-82) also failed: `test_check_symbol_anchors`, `test_tower_skip_rss_arm`, `tools suites`, and `check-windows-portability.py`. All three runs reported five argument-dependent skips: `check-arm-isa-build.py`, `check-cpu-isa-build.py`, `check-cuda-fat-gencode.py`, `check-pr-size.py`, and `check-triton-aot-multiarch.py`. The staged run (bg-97) exited 1 with three failures: `test_check_symbol_anchors`, `tools suites`, and `check-windows-portability.py`. Its tower test did not fail, and `check-now-current.py --staged` passed. These attempts do not establish a green full preflight.

Diagnosis 395d3ad5 confirmed symbol-test Git discovery contamination through a one-variable A/B. Future scratch runs inside the enclosing checkout need `GIT_CEILING_DIRECTORIES` set to the scratch parent. Temporary directories must sit outside the E4B source tree. Source inspection attributes the Windows failure to generated, then deleted temporary headers inside that tree; no Windows rerun confirmed this diagnosis. The focused tower test passed, but its original failure cause remains UNKNOWN. These findings do not prove a general concurrent-contamination explanation.

The oracle guard requires 20 GiB free; the operator measured 18.75 GiB, so that blocker remains. Diagnostic visibility-variable mismatches came from setting device variables empty rather than leaving them unset. They do not establish the original tools failure's cause. Full preflight remains PENDING, not waived. No further full preflight runs are authorized for this follow-up. Do not alter checkers or delete files to bypass the guard. A preflight attempt is not runtime inference evidence. No translation units in scope means no C++ build was proved.

## Owed

The OPEN milestone issue owns the original-checkpoint baseline and its remaining gates. Numeric processor and tower tolerances are PENDING upstream-test enumeration or independent oracle repeatability measurement. Freeze justified values before accepting C++ comparisons. Do not invent numeric bars. Exact integer identity and zero token divergence remain the default requirements.

Original checkpoint hashes, complete tensor inventory, primary-oracle gateability, processor fixtures, runtime dtypes, device execution, and production-call-site mutations remain PENDING. Environment-dependent gates remain PENDING until explicit resource and environment authority exists. Missing values are not defaults.

GGUF and other quantized arms remain obligations of the broader model port. This BF16 milestone does not complete or waive them. Preserve existing arms and explicitly refuse unsupported paths. LittleBit-2, compression, training, and performance rewrites are outside this milestone. All unmeasured performance axes remain open.

## Port map

| Upstream surface | Local destination and obligation |
|---|---|
| Gemma4ProcessingInfo and Gemma4MultiModalProcessor | Register model processing through the shared processor factory. Preserve image, audio, and video defaults separately. |
| Gemma4MultimodalEmbedder | Load original tower and projector tensors. Preserve model BF16 and trained clipping. |
| Gemma4ForConditionalGeneration.embed_multimodal | Fill both factory hooks and reuse the existing encoder and merge runner. |
| Gemma-4 text backbone and chat template | Preserve PLE, KV sharing, rotary positions, text regression, and tool streaming behavior. |
| Public raw-media requests | Route through include/vllm.h and serving to the same model hooks. |

## Tests to port

Before implementation, enumerate tests at the pinned upstream revisions and preserve their parameters, fixtures, modes, tolerances, and failure cases. Exact upstream test symbols and measured float bars remain PENDING, not inferred from historical tests.

Required coverage includes original config and tensor accounting, canonical chat and tools, image patches and positions, audio features and masks, and video sampling and timestamps. Tower comparisons require independent stage references. Hook and runner tests require batching, cancellation, cache lifetime, count mismatch refusal, and call-site deletion mutations. Synthetic tests establish routing or math only. Final reference and reachability gates require original Google weights and all modalities, with no skipped cases.

## Work breakdown

T00 creates the OPEN issue and commits this design. T01 pins metadata, validates evidence, enumerates upstream tests, and freezes justified numeric bars. T02 accounts for config and weights. T03 preserves chat and tools. T04, T05, and T06 implement image, audio, and video processing. T07 and T08 wire the vision and audio encoders. T09 wires hooks and the existing runner. T10 exposes raw inputs through the public ABI and serving. T11 creates the strict acceptance harness. T12 measures original-checkpoint reference equivalence. T13 proves production reachability and completes independent review. Tasks remain serial except for the root-approved T01 prerequisite split below. T00 authorizes none of the product edits.

## T01 metadata contract and remaining prerequisites

T01 is PARTIAL. The standard-library validator and tracked original metadata
fixtures implement the executable CPU scope. Independent SHA-256 pins reject
self-attested replacement bytes. Strict report validation rejects missing cases,
skips, nonfinite metrics, identity changes, and integer or token mismatches.
Offline success explicitly says metadata-only and leaves original-model
gateability PENDING. Runtime acceptance fails closed even for zero errors.
No caller-supplied threshold can override this barrier.

The fixture [README](../../tests/fixtures/gemma4_e4b_google/README.md) records
license, attribution, reproduction, and the report schema. The
[test inventory](../../tests/fixtures/gemma4_e4b_google/upstream-tests.json)
enumerates actual declared test symbols and decorators at both immutable pins.
It includes hashes and source URLs for all Gemma4-named Python test files in the
pinned repository trees. This is source inspection, not test execution.

Relevant vLLM processing tests include
`test_gemma4_image_schema_accepts_variable_patch_counts`,
`test_gemma4_image_batching_keeps_variable_patch_counts_unstacked`,
`test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens`,
`test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens`,
`test_get_mm_max_tokens_per_item_respects_configured_video_num_frames`,
`test_get_prompt_updates_respects_nested_max_soft_tokens`,
`test_video_cache_is_independent_of_image_kwargs`, `test_limit_mm_per_prompt`,
and the five `test_encoder_chunk_*` cases. Their source is
`tests/models/multimodal/processing/test_gemma4.py` at the pinned vLLM commit.
Its `GEMMA4_MODEL_ID` is Google E2B, not E4B. These test declarations do not
establish exact-E4B runtime compatibility.

At the Transformers pin, `tests/models/gemma4/test_image_processing_gemma4.py`
checks defaults, resize dimensions, soft-token budgets, positions, and padding.
`test_processing_gemma4.py` checks vision-token counts, image tokens, truncation,
multimodal counts, and audio-token counts. It conditionally skips unsupported
counting capabilities and explicitly skips video-frame sampling.
`test_modeling_gemma4.py` contains text, image, multi-image, cache-sharing, and
sliding-window integration tests. Its integration checkpoint is E2B. Its tester
skips audio and video feature tests because input dictionaries lack those media.
The inventory preserves the decorators, including skip reasons. Unified and
assistant model suites are separate variants, not replacement E4B evidence.

Static compatibility evidence: pinned vLLM `requirements/common.txt` requires
`transformers >= 5.10.4`. Version 5.14.1 satisfies that declared lower bound.
Pinned vLLM `gemma4_mm.py` imports `Gemma4AudioFeatureExtractor`, `Gemma4Config`,
`Gemma4Processor`, and `Gemma4VisionConfig`. Pinned Transformers
`src/transformers/models/gemma4/__init__.py` exposes configuration, feature
extraction, image, modeling, processing, and video modules through lazy imports.
These source declarations are not an import smoke test or proof that both
engines load and execute the original checkpoint together. The checkpoint's
`transformers_version` is `5.5.0.dev0`, not an installed environment record.

Numeric processor and tower bars remain PENDING. No inspected test supplies a
complete, justified E4B cross-engine per-stage error envelope. A tolerance for
another model, backend, or test is not a valid substitute. Before enabling
runtime acceptance, obtain an authorized pinned environment, verify imports and
original checkpoint loading, and measure independent oracle repeatability for
each required stage and dtype. Ratify explicit absolute and relative bounds
before comparison capture. Keep exact integer identity and zero token divergence.
T01B cannot be marked done until these prerequisites are resolved and reviewed.

The first focused test run failed because the new validator did not exist.
After implementation, 18 CPU tests passed without skips. They use original
metadata and synthetic reports only. No C++ runtime, GPU, weights, or upstream
runtime tests were executed. Full preflight remains PENDING under the existing
T00 environmental blockers and the prohibition on another full-suite attempt.

## Root-approved T01 dependency split

Root accepted T01A metadata tooling at `e1759f0c0637a5be4df0735be17b2668c170d2cf`.
Root supplied passing verification `bg19728tests` and independent review
`reviewer9fce`, with all findings fixed. T01A is DONE for metadata tooling only.

T01B retains the PENDING measured oracle compatibility and justified frozen
numerical bars. T02 synthetic CPU loader/config work depends on T01A only.
T01B is a HARD dependency for every numeric/runtime acceptance, including T12
and T13. Synthetic fixtures never establish original-Google inference.
This scheduling clarification changes no runtime barrier or acceptance criterion.
Full preflight remains PENDING under the recorded failures. No rerun is authorized.

## T02 bounded output-head slice

The first executable slice validates tied/untied output-head loading through
real safetensors and Gemma loader sources. The original Google config is tied.
Pinned vLLM `vllm/model_executor/models/gemma4.py` at
`a7c23ac96d7806e7c7e7d862eadbce5a33529b94` constructs `ParallelLMHead` with
vocabulary and hidden dimensions, and ties it only when configured.
The downloaded source SHA-256 is
`205e9c543b263b2cebf0a8567de629597e179713f02e505cc4f94fb5168f566a`.
This is source grounding, not oracle execution.

CPU tests reproduce missing and malformed untied heads before the fix.
The expanded red run executed 8 cases with 8 failed assertions and zero skips.
The fix requires the untied head and its exact configured shape, and rejects
nonboolean nonnull tie flags. Absent/null defaults and explicit tying remain
unchanged. Positive controls check BF16 transposition and both loader overloads.
The green run executed 8 cases and 38 assertions with zero failures or skips.

T02 remains PARTIAL pending tower/projector inventory, clipping-scalar loading,
modality-config validation, and original-shape tower fixtures. Reduced text
fixtures and original-config refusal tests do not prove those guarantees.
PLE, KV topology, text math, tower execution, and runtime gates are unchanged.
Root rerun and independent review of this immutable slice remain required.
The owning milestone issue remains OPEN.

## T02 modality projector slice contract

The next bounded slice retains both modality projectors in the actual decoder
weight bundle. Full tower loading is deferred: the existing vision type stores
BF16 vectors, while the audio type is an F32 reference, not a production BF16
encoder representation. Do not populate that reference and claim production audio.

Source: pinned vLLM `vllm/model_executor/models/gemma4_mm.py`,
`Gemma4MultimodalEmbedder.__init__` (lines 964–997) and
`Gemma4ForConditionalGeneration.hf_to_vllm_mapper` (1045–1058), at
`a7c23ac96d7806e7c7e7d862eadbce5a33529b94`. Local source SHA-256:
`ee9b33f5f3d76bfec371dfc65292a915cc5b4741367ba0b28a450a5af8c40f9f`.
The executing constructor uses weightless pre-projection RMSNorm and a biasless
linear, despite its stale class docstring saying post-projection norm.
Checkpoint names are `model.embed_vision.embedding_projection.weight` and
`model.embed_audio.embedding_projection.weight`. Source shapes are
[text hidden, vision hidden] and [text hidden, audio output_proj_dims]; the
constructor falls back to audio hidden for absent/null/zero output_proj_dims.
The original metadata supplies 2560, 768, and 1536 respectively.

For this staged loader contract, absent/null modality config preserves legacy
text-only loading; false and other nonobject values are errors, not absence.
An object must supply a positive integer hidden_size and positive finite
rms_norm_eps. Audio output_proj_dims may be absent/null/zero (hidden fallback)
or a positive integer. Reject malformed values before projector allocation.
Retain the full modality config alongside an owned BF16 raw [out,in] projector,
including clipping flags and encoder fields for subsequent tower work. Validate
exact source shape and use the existing dense materializer, preserving its
scaled F8_E4M3 support rather than adding a global BF16-only guard.

Tests enter both decoder loader overloads using small real safetensors.
Cover config refusal, required tensor accounting, rank/orientation errors,
BF16 byte preservation, F16 refusal, scaled FP8 preservation, ownership after
shard destruction, and original-config shape rejection without full allocation.
This is projector accounting, not tower accounting or numerical equivalence.
T07–T09 must connect the retained projectors to encoder execution and hooks;
MODEL-MM-gemma4-mm-gemma4-for-conditional-generation and OPEN
ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8 own that wiring. Complete tower inventory,
clipping-scalar loading, full modality validation, original-shape tower fixtures,
assistant/tools/image/audio/video acceptance, and T12/T13 remain owed.
Full preflight remains PENDING; another full run is prohibited for this slice.

### Projector slice verification

Both real decoder loader overloads now retain validated config and owned BF16
projectors in raw checkpoint [out,in] layout (no transpose). Scaled F8_E4M3
passes through the unchanged dense materializer. The red-before run failed
84 assertions across three new cases; the original nine cases passed.
Green passes 16 cases / 192 assertions, zero skips, using the minimal CPU
target. CTest first rejected the stale nine-case expected count; updating it
to exactly sixteen restored its zero-skip gate. The 28 metadata regressions
and offline CLI pass. Evidence: control-room
`assets/work/2026-10-10-e4b-projectors`.

Root supplied rootbg271 and review24858d PASS for the prior output-head slice
(9 cases / 44 assertions plus 28 metadata cases). This projector slice awaits
independent immutable-head review and root verification. T02 remains PARTIAL:
tower inventory, clipping, full modality validation and encoder/hook wiring
are owed. No full preflight, GPU, weights download or original-model inference
ran. Assistant/tools/image/audio/video acceptance is unchanged.

## T02 vision patch-embedder retention slice

Inspection found no production vision weight loader: `Gemma4VisionWeights`
exists only in forward code and synthetic/reference tests. Bound this slice to
its two patch-embedder tensors; encoder blocks, typed full tower configuration,
clipping scalars and audio tower remain pending. An empty blocks vector is not
an executable tower and must not be interpreted as encoder readiness.

Pinned Transformers `modeling_gemma4.py` at
`a08ace4bbd97e721c98751deec37d87b026acadc`, lines 579–588, defines
`patch_embedder.input_proj.weight` [hidden, 3*patch_size^2] and
`patch_embedder.position_embedding_table` [2, position_embedding_size, hidden].
The vLLM prefix mapper cited above maps `model.vision_tower` unchanged apart
from the model prefix. Retain these in the existing vision representation,
with raw BF16 input projection and the representation's documented F32 position
lookup exception (exact widening of materialized BF16, not a new F32 policy).
Require positive integer patch_size and position_embedding_size in present
vision configs, checked multiplication, exact tensor shapes, and the unchanged
dense BF16/scaled-FP8 materializer. Absent/null vision stays optional. The full
JSON config remains retained by the projector slice; no invented defaults.

Tiny original-structure tests exercise both actual decoder loader overloads,
missing tensors, malformed shapes/dtypes/config, ownership and raw layout.
Preserve text/projector regressions, update the exact CPU case count, compile
affected shared-header consumers, and run the 28 metadata tests. Full preflight
remains pending and prohibited by the current task. No inference claim follows.

The parent reports projector rootbg296 PASS (16 cases / 192 assertions and 28
metadata tests) and independent reviewer7892d92a PASS (13 mutations, copy-mode
and header checks). That slice is reviewed; T02 remains PARTIAL.

### Patch inventory verification

Both decoder loader overloads retain owned input-projection BF16 bits and
exactly widened position-table values in `vision_patch_weights`. Config remains
in `vision.config`. This partial representation deliberately leaves blocks and
embed_projection empty; the projector remains in its existing owned field.
The dense materializer is unchanged: scaled FP8 matrices work; rank-three FP8
position tables remain explicitly rejected, not newly supported.

Red: 18 cases, 48 failed assertions, original 16 cases passing. Green: 20 cases,
356 assertions, zero skips, exact-count CTest PASS. Both overloads cover missing
inventory, malformed config/shapes/dtypes, overflow, absent modality, ownership
past shard destruction, raw layout and nonunit FP8 scaling. Seven direct shared
header consumers passed syntax-only compilation. All 28 metadata tests, record,
anchor and whitespace checks pass. Evidence: control-room
`assets/work/2026-10-10-e4b-vision`; captured pinned Transformers source SHA-256
`ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916`.

T02 remains PARTIAL. Independent review and root verification of this patch
slice are pending. Encoder blocks, full typed tower config, clipping scalars,
audio tower and executable encoder/hook wiring remain owed. No inference,
original-weight, GPU or full-preflight claim follows from retention.

## T02 vision encoder-block retention

The parent reports rootbg363 and reviewf3c57a74 PASS for the patch slice:
20 cases, 358 assertions, 28 metadata tests; overflow mutation closed.
T02 remains PARTIAL. Retain all seven matrices and six learned norms per
encoder layer through both decoder loader overloads. Retention does not
establish encoder execution, clipping, audio, or runtime acceptance.

Pinned Transformers a08ace4bbd97e721c98751deec37d87b026acadc
`modeling_gemma4.py:168-215,692-701,927-954,990-1002,1036-1038`
defines `model.vision_tower.encoder.layers.{i}.` with attention
`self_attn.{q,k,v,o}_proj.linear.weight` and MLP
`mlp.{gate,up,down}_proj.linear.weight`. Shapes are respectively
[heads*head_dim,H], [kv_heads*head_dim,H] (K and V),
[H,heads*head_dim], [I,H] (gate and up), and [H,I].
The four layer norms have shape [H]; `self_attn.{q,k}_norm.weight`
have shape [head_dim]. V normalization has no learned scale. No biases,
V norm, rotary buffers, or projector norm tensors are invented. The pinned
vLLM `gemma4_mm.py:1046-1055` mapper preserves this tower structure.

Keep the existing raw BF16 block representation, including BF16 learned
norm scales from the BF16 checkpoint (the forward widens for RMS arithmetic).
Preserve dense matrix scaled-FP8 conversion. Norms require BF16, as the
existing representation requires; no new F32 checkpoint policy is introduced.
Validate positive integer layer count, intermediate size, head count,
KV head count and head dimension, head divisibility, and checked tensor and
block-vector byte sizes before allocation. Use source dimensions rather than
assuming H equals heads*head_dim. Full execution config remains pending.

Tiny two-layer original-structure fixtures must distinguish each tensor's
layout and each layer, survive shard destruction, and reject missing,
malformed, unsupported-dtype and overflow inputs through both overloads.
No original weights or large allocations are needed. Preserve patch/projector
behavior and absent vision. Full preflight remains pending and prohibited.

### Block retention verification

Both real decoder loader overloads now retain all 13 learned tensors per layer
in `vision_patch_weights.blocks`, using original `.linear.weight` names.
Validated local integer dimensions determine shapes, including asymmetric
query/KV widths; the full JSON config remains owned. This does not make the
existing MHA-only forward execute GQA, or supply clipping scalars. Full typed
execution config and encoder/hook wiring remain pending.

The initial red run failed 97 assertions; a malformed fixture initially failed
in the safetensors reader rather than the loader. After correcting its payload,
the complete suite against the original loader failed 251 assertions (23 cases,
19 passing). Restored implementation passes 23 cases / 1492 assertions, no
skips, through the exact-count CTest gate. The two-layer fixture distinguishes
every matrix and norm, raw layout, BF16 and nonunit scaled FP8 matrices,
shard-independent lifetime, missing inventory, dtype, config and byte overflow.
Nine direct header consumers pass syntax-only compilation with actual
compile_commands flags. All 28 metadata tests and offline CLI pass; runtime
acceptance remains false. Record and symbol-anchor checks pass.

Evidence: `assets/work/2026-10-10-e4b-vision-blocks`. Only the existing minimal
CPU target was built with two jobs. Full preflight remains prohibited/pending;
no GPU, original weights, install, remote operation or inference ran. The
implementation awaits independent review and the parent rerun. T02 is PARTIAL.

## T02 clipping and typed vision configuration

Prior block slice at 54e1150c1cebf1ac4aaecb60b9f085965b5f1576 is reviewed:
parent reports rootbg422 and review0d43 PASS, 23 cases/1492 assertions and
28 metadata tests. T02 remains PARTIAL.

Pinned Transformers a08ace4bbd97e721c98751deec37d87b026acadc
`modeling_gemma4.py::Gemma4ClippableLinear` (168-194) registers scalar
input_min, input_max, output_min, output_max on each of seven block linears
when use_clipped_linears is true; defaults are negative/positive infinity.
`configuration_gemma4.py::Gemma4VisionConfig` supplies false as the clipping
default, pooling 3, RMS epsilon 1e-6, default RoPE theta 100, tanh GELU,
no attention bias and no standardization. Google explicitly enables clipping.
Pinned vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94
`gemma4_mm.py::Gemma4ForConditionalGeneration` constructs the Transformers
vision tower, replaces inner linears and maps the model.vision_tower prefix.

Header-only HTTP ranges from Google's immutable model.safetensors confirm
448 bounds (16 layers x 7 linears x 4), all BF16 scalar shape []. Names are
model.vision_tower.encoder.layers.{i}.self_attn.{q,k,v,o}_proj.{bound} and
model.vision_tower.encoder.layers.{i}.mlp.{gate,up,down}_proj.{bound}.
No bounds belong to patch/projector plain linears. The 281040-byte header's
SHA256 is 3d3bfac0fbdf7bd65981df4033efb7690240a8fe6f92eb2f3e3846b289a6f994;
metadata evidence resides in assets/work/2026-10-10-e4b-vision-clipping.
No tensor payload was fetched. This is inventory, not numerical acceptance.

Both real loaders shall require all bounds when clipping is enabled and
retain each as a float scalar for existing Clip consumers. BF16 widening is
exact and costs four bytes per bound, not matrix widening. Disabled clipping
requires no bounds and uses true infinite no-op defaults. Reject wrong shape,
wrong dtype, NaN or reversed bound pairs; permit infinities, including upstream
initialization defaults, and equal endpoints. The NaN/order checks are local
fail-closed policy against undefined/unintended clamp behavior.

Retain optional typed vision configuration alongside weights: all existing
forward dimensions, text width, epsilon, pooling, theta, clipping flag and
standardization/activation/bias/dropout semantics. Validate typed scalar inputs,
float epsilon conversion and pooling-square overflow. Reject unsupported
activation, bias, standardization and RoPE type rather than silently ignoring
them. Preserve GQA dimensions for inventory without claiming the MHA forward
executes them. No new encoder hook or execution support is in this slice.

Real-path tiny fixtures shall prove both overloads, layer-specific bounds,
shard-independent lifetime, defaults, shape/dtype/NaN/order/config/overflow
refusal and preserved BF16/scaled-FP8 matrix layout. Red precedes product
changes. Run only the minimal CPU target, actual-flags header syntax checks,
28 metadata regressions and focused record/style/trailer checks. Full preflight
and audio, hooks, original-checkpoint numeric/runtime gates remain pending.

### Clipping/config verification

Both real loader overloads now retain source-named BF16 scalar bounds and an
optional typed vision config. Disabled/absent clipping leaves infinite no-op
bounds; enabled clipping requires all four scalars for every linear. NaN,
reversed order, non-scalar shapes and non-BF16 bounds fail closed. Existing
matrix loading, including scaled FP8 conversion, is unchanged. Typed config
retains GQA dimensions but does not extend the existing MHA execution path.
Activation, bias and standardization are restricted to the existing forward's
semantics; dropout is retained for inventory but inference does not apply it.

Red before product changes: 27 cases, 23 passing, 922 failed assertions out of
2416; original 23 cases passed. Final green: 29 cases, 2648 assertions, zero
skips, plus exact-count CTest 1/1 PASS. Fixtures cover every bound on both
layers through both loaders, lifetime, infinite/equal endpoints, explicit false,
missing configuration, overrides, malformed types, NaN/Inf config and overflow.
Existing BF16/scaled-FP8 layout tests remain green. Nine direct header consumers
pass fresh syntax compilation using actual compile_commands flags. All 28
metadata tests and offline metadata CLI pass; runtime_accepted remains false.
Record and symbol-anchor checks pass. Evidence is in the clipping scratch
folder above (under 1 MiB); only the reused minimal CPU target was built, -j2.

No preflight was launched on resume. Process inspection found no remaining
process from the earlier mistaken, timed-out preflight; none was killed.
Full preflight is pending, not green. T02 is PARTIAL: audio inventory and
encoder/hook wiring remain owed. Original-checkpoint inference, numerical
acceptance, GQA execution and runtime prerequisites are not established.
Root verification and independent review of this slice remain pending.
