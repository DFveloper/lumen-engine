# Pascal and Volta enablement

Row: `BACKEND-CUDA-SM061`
Related rows: `BACKEND-CUDA-SM060`, `BACKEND-CUDA-SM070`.
Issue: `ISSUE-GH-146`.

## Now

SPIKE. User approved implementation on 29 October 2026. Base: `db1d71c9b7352a07983b408b2e755e853b4534f0`.

## Scope

Implement production CUDA execution for Pascal 6.0, Pascal 6.1, and Volta 7.0. Preserve existing public interfaces and newer GPU dispatch. Optimize Pascal using the local Quadro P620, 2048 MiB, sm_61, driver 580.178.04. CUDA 12.4.131 is installed. Hardware access is approved. Volta hardware validation and optimization remain TODO until the GPU arrives. P100 runtime validation remains pending without that hardware.

## Upstream chain

Use `.agents/upstream-sync.md` for the primary pin and `.agents/oracles/llama-cpp.md` for the registered secondary pin. vLLM has no Pascal or Volta execution at its architecture floor. Read the matching pinned llama.cpp CUDA source, including capability dispatch, BF16 conversions, float GEMM, quantized GEMV, and attention. Re-anchor the superseded `237ad9b96` citations in `cuda-arch-breadth-fp16.md` before relying on them. The executing chain must cite actual source paths and lines.

## Our baseline

`CMakeLists.txt:202` selects architecture 121a by default. `cmake/CudaArchFeatures.cmake` resolves per-source architecture features. `src/vt/cuda/cuda_paged_attn.cu` contains guarded BF16 WMMA bodies and scalar fallbacks. `src/vt/cuda/cuda_matmul.cu` routes BF16 work to cuBLASLt, which requires a real legacy execution path. Existing historical breadth work is scoped in `cuda-arch-breadth-fp16.md`; `ISSUE-GH-146` remains open.

## Design and port map

Use capability-based dispatch behind the existing CUDA backend. Add architecture-specific files for legacy hardware where new kernels are required. Route all model execution through the existing C API, loader, ModelRegistry::Forward, attention block, fusion, merged GEMM, and sampler. Preserve existing tensor storage formats unless an explicit documented adaptation is unavoidable. Keep all current `vllm_*` symbols, flags, configuration keys, filenames, and defaults on newer GPUs.

Audit and fix compile and execution blockers in the portable CUDA translation units. CUDA 12.x accepts the requested legacy targets; diagnose incompatible newer targets or instructions explicitly. Do not claim runtime support from successful compilation alone. Add narrow refusals naming missing quantized operations where a model arm cannot run.

## Tests to port

Port the applicable pinned-oracle capability, conversion, matrix, quantization, and attention cases with revision anchors. Add production-entry-point regression coverage that reaches the selected old-architecture implementation. Host-only dispatch tests supplement actual GPU tests. Record failing tests before code, focused green, emitted SASS, and production-call-site mutation failures.

## Gates

Full scoped gate: legacy CUDA configure/build with tests enabled; focused CUDA regression tests on P620; CMake architecture-selection tests; existing CPU and CUDA focused tests for changed operators; `scripts/agent-preflight.sh --quiet` for repository checks. Run builds with at most four jobs. Record exact commands and unchanged baseline failures separately.

Build sm_60, sm_61, and sm_70 with CUDA 12.x and inspect emitted device code. Execute all GPU commands under `${GPU_LOCK:-$HOME/gpu.lock}` on this non-fleet local device. Never bypass a fleet lease if the selected device changes.

Use real pinned checkpoint weights fitting 2 GiB including KV cache, activations, scratch, and runtime overhead. The initial candidate is OPT-125m if reachable through the current public entry point; a supported smaller dense checkpoint is an allowed builder choice. A quantized small Llama or Qwen checkpoint covers a compressed arm when authorized storage and downloads are available. Record artifact revisions, SHA256, memory, prompt, and all output tokens. Model downloads await explicit approval and storage choice. No synthetic fixture substitutes for this gate.

Declare workload axes before optimizing: batch 1, concurrency 1, prompt lengths 16 and 128, generated lengths 16 and 64, greedy sampling. If model limits or CLI reachability require adaptations, record the exact reason before execution. Correctness must pass first. Compare matching pinned-oracle execution on the identical device, artifact, and workload. Required performance floor: throughput ratio >=1, latency ratio <=1, memory ratio <=1 on every declared axis, reproduced on an idle device with the same binary. Unmet axes remain open, with a next traceable hypothesis. No ceiling claims.

## Dependencies and risks

Pinned source checkouts and model storage values are not configured. Source downloads needed to read the registered oracle are authorized task preparation; large checkpoint downloads still need user approval. Do not copy another developer's paths. The GPU is small, so broad model support cannot be inferred from a single fitting checkpoint. BF16, half, atomic, shared-memory, and tensor-core assumptions need separate inspection.

## Work breakdown

1. Establish failing legacy build and production execution tests.
2. Implement the minimum complete legacy operator paths and safe dispatch.
3. Build all three architectures, run Pascal gates, and optimize measured bottlenecks.
4. Document supported arms, evidence, and exact outstanding hardware TODOs.

## TODO after Volta GPU delivery

- Record GPU identity, VRAM, driver, and toolkit.
- Run real-checkpoint generation and token correctness through the public API.
- Confirm FP16 tensor-core dispatch and memory formats with source and trace evidence.
- Optimize attention, prefill GEMM, and decode GEMV on actual hardware.
- Measure throughput, latency, and peak memory against the pinned oracle on matching workloads.

Volta stays build-verified and runtime-unverified until these gates run.

## Integration and stop conditions

One issue-owned branch and one PR carry the spec before implementation. No push or merge is authorized. Independent implementer and fresh mutation reviewer are mandatory. The operator reruns the scoped gates. Missing external resources leave the affected gate PENDING, without stopping independent implementation or local verification. Stop acceptance on correctness failures or invalid GPU exclusion.

## Evidence

Implementation and review evidence belongs in `.agents/evidence/pascal-volta-enable/`.
