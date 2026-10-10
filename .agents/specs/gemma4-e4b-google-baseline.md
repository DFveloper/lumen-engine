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
