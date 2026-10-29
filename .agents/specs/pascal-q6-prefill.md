# Pascal Q6_K MMQ prefill activation format

## Now

ACTIVE. The diagnosed operation gap is proved. Implementation waits for the shared Q5_0/Q8_0 D4 packer head.
Issue: ISSUE-LOCAL-01M640A0QTNZSV3WXBHHN377K0.
Pinned task base: 561ee7b38.

## Scope

Correct standard dense Pascal Q6_K prefill through registered MatmulBTQuant.
Reuse the shared D4 activation packer from the separate Q5_0/Q8_0 prefill implementation.
Keep original compressed Q6_K NK weights and the public ABI.
Preserve decode, grouped operations, unsupported geometry, sm60 standard cuBLAS eligibility, Volta, and newer architecture defaults.
Exclude model dtype, attention, branding, and other weight formats.

## Upstream anchors

The primary vLLM pin cannot execute this Pascal GGUF workload.
Use registered llama-cpp pin 62318cd57dc63900ecb153272c0b10f78550855d.
Read the complete executing chain:

- ggml/src/ggml-cuda/ggml-cuda.cu:1955-2010 selects standard contiguous MUL_MAT, with MMVQ before MMQ.
- ggml/src/ggml-cuda/mmvq.cuh:3 and mmvq.cu:318-355 retain Pascal MMVQ through M=8.
- ggml/src/ggml-cuda/mmq.cu:355-429 requires supported format, at least 48 KiB shared memory, and compiled native DP4A for standard Pascal MMQ.
- ggml/src/ggml-cuda/mmq.cuh:85-92 selects Q6_K D4. Lines 607-611 select its tile load and DP4A dot.
- ggml/src/ggml-cuda/quantize.cu:457-550,574-602 implements per32 F32 D4 scales using inv=127/amax, roundf(x*inv), and d=1/inv.
- ggml/src/ggml-cuda/mmq-load-tiles.cuh:946-1030 reconstructs signed Q6 integers and signed16 subscales.
- ggml/src/ggml-cuda/mmq-vec-dot.cuh:975-1006 and vecdotq.cuh:650-675 contract Q6 with F32 activation scales.

## Eligibility and design

Extend the shared D4 production eligibility gate to Q6_K, rather than create a competing gate.
Require standard ungrouped raw canonical NK storage, contiguous activation rows, K divisible by 256, and M greater than 8.
Require actual Pascal native DP4A eligibility and at least 48 KiB per-block shared memory.
The standard sm60 path fails native DP4A eligibility and retains its previous implementation.
Mirror the engine's explicit carrier adaptation from the shared packer for F32/BF16 input and output.
The upstream arithmetic reference remains its F32 source and destination contraction.

Consume eight F32 D4 activation scales per 256-value Q6_K weight block.
Each32 group contracts two signed16 subscales and signed Q6 integers.
Multiply the integer subtotal by its D4 scale, then fold the widened block half weight scale.
Use safe packed loads for alternating two-byte-aligned 210-byte weight blocks.
Add a separate Pascal hardware helper and reuse the shared activation representation.

## Evidence

The actual same-input shape is M16, K1536, N576 from SmolLM layer0 down projection.
Q8_K per256 activation scales give RMSE 0.029947596 against the executed GPU oracle.
D4 F32 scales give RMSE 1.837913e-7. Q8_1 half scales give RMSE 0.000295233.
Independent raw packed Q6 reconstruction matches the existing decoder exactly.
The raw integer D4 contraction matches the actual oracle with RMSE 2.092407e-7.
Reproducer: /tmp/smol-q6-prefill-formula.py. Executing tensor evidence lives under /tmp/smol-oracle-history-dumps.

## Tests and gates

First capture a failing registered-operation test using independent raw packed Q6 scalar arithmetic.
Cover signed subscales, all high-bit planes, zero input, tiny groups, disparate32-group magnitudes, and half-scale sensitivity.
Cover K256,512,1536, ragged output rows, M9,16,128, and F32/BF16 carriers.
Keep M1 and M8 controls on their existing contracts.
Mutate the production dispatch, narrow the activation scale to F16, and corrupt a packed weight/subscale index.
Require each corresponding focused gate to fail, restore byte-for-byte, and rerun green.
Compile the changed production translation unit for sm60, sm61, and sm70 with at most four jobs.
Run the bounded full quant suite that fits the P620. The large published GLM MoE case requires more GPU memory.
All GPU commands hold /home/user/gpu.lock. Reuse one scratch archive.
The operator reruns exact actual SmolLM token gates before performance acceptance.

## Risks

Float contraction order can differ from the tiled oracle. Operation parity alone does not establish model token parity.
The separate KV storage precision difference remains outside this scope.

## Stop conditions

Do not write product code before the shared Q5_0/Q8_0 packer immutable head is available.
Return NEEDS_CONTEXT if shared eligibility or carrier contracts are missing.
Stop for a required ABI change or edits outside the declared scope.

## Owed

Independent review, actual checkpoint correctness, and performance acceptance remain required.
