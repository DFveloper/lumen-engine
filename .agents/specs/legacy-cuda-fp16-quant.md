# Typed FP16 kept-quant CUDA projections

## Now

SPECIFIED. Wave2 starts from reviewed integration `2f195a4c3`.
Issue: ISSUE-LOCAL-01M64S95TXSC1F8382E8WPDTZC. Row: BACKEND-CUDA-SM061.
Parent design: [shared FP16 dense execution](legacy-cuda-fp16-dense.md).
This specification precedes implementation. It does not enable automatic model dtype selection.

## Problem and scope

Registered MatmulBTQuant and MatmulBTQuantGrouped reject FP16 output. Existing CUDA two-byte stores encode BF16.
The actual legacy model formats need typed FP16 projections before model dtype can propagate.
Implement dense primary Q4_K, Q5_0, Q6_K, Q8_0 math for FP16 input and output.
Add explicit FP16 stores to reached generic dense and grouped formats. Preserve packed weights and original block layouts.

Authority: src/vt/cuda/cuda_quant_dot.cu; a new reached cuda_fp16_quant.cuh; necessary existing typed-output helper headers;
src/vt/ops.cpp quant guards; tests/vt/test_cuda_fp16_quant.cpp; tests/CMakeLists.txt; this row's issue and evidence.
No attention, KV, model allocation, global dtype, public C ABI, or fused MoE implementation changes belong here.

## Source and design

Primary pins remain vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94 and first-party GGUF plugin d4c1f0d082fc7cd4350da56689109a01c1f29d6c.
Read the oracle records before accepting any result. Exact source snapshots exist under /tmp/lumen-plugin-source.

- quantization/linear.py:35-58 selects MMVQ at M<=6 when N<=5120, M<=2 otherwise, then MMQ.
- csrc/gguf/gguf_kernel.cu:33-83 quantizes per32 using float input, XOR max and original-sum reductions, d=amax/127 and roundf(x/d).
  It stores half d and half original sum, and pads scratch K to512. Both primary batch arms use this Q8_1 representation.
- mmvq.cuh:1-39 partitions blocks by qi/vdr and XOR-reduces float lane partials.
- mmq.cuh:1-103 stages Q8_1 and uses NVIDIA tiles of4 activation rows,32 weight rows and4 warps.
- vecdotq.cuh:104-138 preserves unsigned Q5 high bits and original-sum offset correction.
- vecdotq.cuh:162-176 contracts signed Q8 integers; it does not use original sum.
- vecdotq.cuh:328-377 uses reconstructed int8 sum times half d in Q4_K MMVQ minima, but original half sum in MMQ.
- vecdotq.cuh:447-491 preserves signed Q6 per16 scales, two such scales per32 activation block, and half weight scale.

Route through existing registered MatmulBTQuant, original compressed blocks, graph-safe per-stream scratch and stream launches.
New primary formula applies only when both activation and output are FP16 and the weight is one of the four formats.
Existing F32 and BF16 output arithmetic and selection remain unchanged. Cross-dtype FP16 output uses the existing arithmetic with typed stores.
New FP16 stores must encode half, never reinterpret BF16. Generic grouped arithmetic and expert ownership remain unchanged.
The wrapper admits FP16 only if the selected provider advertises the implemented scalar type.
Do not broaden the secondary D4 F32 path to FP16. Preserve all existing environment controls and repacked-layout policies.
A primary raw-block path must decline repacked weights to an existing typed representation or refuse the exact missing layout.
Native DP4A applies at sm61 and newer. The existing scalar byte-dot helper covers sm60; P100 runtime remains pending hardware.

## Binding coordinator decisions

1. Primary Q8_1 math only for dense FP16 input and FP16 output. Preserve old F32/BF16 outputs.
2. MatmulBTQuantGrouped guards and typed stores are included. Fused MoeGateUpSwiGLUGrouped remains unchanged and precisely refuses unsupported FP16.
3. Preserve old batch policies; add the primary6/2 threshold only for the new FP16 branch.
4. MXFP4 FP16 precisely refuses the missing CPU FP16 fallback store. Do not silently enter a CPU fallback or broaden CPU admission.

## Tests and gates

Capture the smallest registered FP16/FP16 RED before implementation. Tests enter MatmulBTQuant, not private kernels.
Use independent packed scalar formulas with Q5 high bits, Q4 scale/min groups, signed Q6 subscales and signed Q8 weights.
Cover M1/2/3/6/7/16/83/128, N small/tails and5120/5121, K32/64/576 for32-block formats and256/512/1536/2048 for K-quants.
Cover stride-padded activation rows, zero groups, scale-half rounding, roundf boundaries, original versus reconstructed sums, untouched packed bytes, and FP16 format.
Port applicable tests/test_kernels.py:127-198 parameters: hidden256/1024, dtypesFP16/BF16/F32, seed0, MMVQ M1, MMQ M7/83/128/2048.
Preserve upstream MMVQ atol1 rtol0.1, MMQ atols1/1.5/1.2 and rtols0.1/1e4/20 by dtype, plus tighter formula fixtures.
Upstream fixtures download moving HF snapshots. Use fixed independent packed fixtures as an explicit reproducibility adaptation.
Enumerate formats outside the four primary kernels and test existing generic typed stores rather than claim a full upstream format port.
Test dense and grouped FP16 output, provider selection refusals, MXFP4 refusal, and registered graph capture/replay without CPU readback or expansion.
Compile actual changed CUDA source for sm60/61/70 and inspect cubins. Run focused tests and the bounded existing full quant gate.
The published GLM MoE geometry exceeds P6202GiB and remains an explicit excluded resource gate.

Mutation gates remove the primary production branch, replace FP16 stores with BF16, widen Q8_1 scale, replace original sums, unify Q4 correction arms,
flip Q6 signed scales, change thresholds or bit indices, and let FP16 reach D4. Each focused test must fail and restoration must pass.
A fresh reviewer independently repeats source, runtime, reachability and mutation checks. The operator reruns the gates.

## Risks, evidence, and stop conditions

Float reassociation can change half outputs. Preserve the executing lane/tile chain and record any unavoidable adaptation before claiming parity.
Source formulas are necessary evidence; they do not establish a primary full-model runtime PASS. The primary model lane lacks dependencies and its unchanged build filters sm61.
No package installation or new checkpoint download is authorized by this slice. Existing approved artifact pins stay unchanged.
Stop for missing representation, unimplemented provider admission, public ABI break or an unrepresented primary contract.
Do not relax tolerances or overwrite model goldens. Current model failures remain failures until Wave4's full public model gate.
Evidence belongs under .agents/evidence/legacy-cuda-fp16-wave2 and records exact commands, hashes, RED/GREEN, mutations and limits.
Volta implementation is build-checked; runtime validation and optimization remain TODO until hardware arrives.

## Owed

Wave4 on BACKEND-CUDA-SM061 owns automatic model dtype and loader-to-sampler wiring. The parent design names the row-owned issue.
This stage is reachable through registered operations but does not complete a model capability or performance gate.
MXFP4 FP16 CPU fallback storage and fused grouped FP16 MoE are unimplemented and must be precisely refused where reached.
