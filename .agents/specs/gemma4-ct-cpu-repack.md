# Gemma4 compressed-tensors CPU repack

## Ownership and scope

Row: `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`.
Issue: `ISSUE-LOCAL-01M6AS6AFMM9G0D76SKPJGT9SH`.
Base: `d8593c3a84b6e42c401ba278050dc2ca143f3a73`.

Accept the reported mixed W4A16/W8A16 compressed-tensors checkpoint through
the public engine loader and CPU forward. Preserve compact quantized storage.
Do not expand the complete expert bank to BF16. Existing BF16 and FP8 model
paths retain their behavior. GPU execution, other quantization schemes, and
multimodal inputs are outside this repair.

## Now

Design committed before implementation. The model row retains its existing
lifecycle. This repair does not complete the row's outstanding multimodal gates.

## Reference and authority

The developer states that vLLM cannot be installed on this hardware. On
1 November 2026 the developer explicitly selects the installed llama.cpp and
the original Q4_0 GGUF as this task's numerical comparator. This is a narrow
runtime-oracle substitution. It does not claim vLLM parity or advance its pin.

The supplied llama.cpp checkout is `~/aikar-engine`, clean at
`1a42a089f4db9b716944006e29a1a66b2fedb6d1`. The installed binary reports
`b12164-36a623af6`; record its binary hash separately. Do not attribute its
execution to the checkout revision. The source format anchors are
`ggml/src/ggml-quants.c::dequantize_row_q4_0` and
`ggml/src/ggml-quants.c::dequantize_row_q8_0` in that checkout.

The developer supplies these exact local artifacts:

- Original: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf`.
- Repack: `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/Lumen-3.8-Pulsar-it-qat-w4a16-ct/`.
- Converter: `/mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/convert_gguf_q4_0_to_ct.py`.

The converter preserves quantized codes and F16 block scales. Its
`_convert_q4_0_blocks_to_ct` and `_convert_q8_0_blocks_to_ct` define the exact
repack. Its Marlin padding extends dense intermediate 2112 to 2176 and expert
intermediate 704 to 768. The developer confirms this padding. Zero-filled
additional rows and columns must remain functionally inert.

## Design

Validate the selected compressed-tensors scheme, ordered targets, ignored
parameters, group size 32, symmetric signed integer codes, 4 or 8 bits,
tensor shapes, data lengths and scale dtype. Refuse unsupported configurations
with the tensor or scheme name before interpreting bytes.

Repack INT4 words to the existing Q4_0 block layout. Copy each F16 scale without
rounding and interleave the first and second groups of sixteen codes into
low and high nibbles. Repack INT8 offset codes to existing Q8_0 signed bytes.
Use the native shared quantized matrix multiplication operation, including
merged attention projections and selected experts. Route the dense gated MLP
through `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`; extend that
descriptor if it cannot express GeGLU. Reuse `vt::FusedChain` for fusion.

The existing Gemma4 compute path uses BF16 activations and model parameters.
Both supplied `torch_dtype` fields specify BF16, although `dtype` specifies
F16. Scope F16 converting loads to this explicitly recognized CT arm, preserve
the existing BF16 compute contract, and report this rounding in the comparison.
Do not reinterpret F16 bytes as BF16 or silently widen acceptance globally.
Native F16 Gemma4 compute is outside this repair. If this conversion defeats
the numerical gate, report the measured difference and repair the relevant
precision path rather than weakening the gate.

Production loading remains reachable through `include/vllm.h`, the registered
Gemma4 model and `ModelRegistry::Forward`. CLI changes must not require internal
headers. Retain mmap ownership for borrowed buffers and use bounded scratch
storage for selected experts. Refuse a CPU-only compact arm on other devices
before an unsupported operation executes.

## Risks and stop conditions

Packing order, signed offset, negative scales, padding, expert indexing,
shared K/V, mixed per-projection bit widths and F16 conversion can each change
results. Test these independently. Never infer correctness from a successful
load. Stop for a genuine unsupported artifact or unavailable resource, with
the precise condition and evidence. Do not change the original artifacts or
the comparator checkout. No GPU work or dependency downloads are authorized.

## Gates and evidence

1. Capture a production-loader failure before implementation with a small
   hermetic mixed CT fixture. Test F16 nonquantized parameters and scale values.
2. Prove Q4_0 and Q8_0 code and scale preservation against the converter and
   original GGUF, including padded intermediate regions and selected experts.
3. Check malformed shapes, truncated data and unsupported schemes fail by name.
4. Exercise production load plus registered CPU forward with nonzero MoE
   contribution. Compare hermetic logits with an independent dense reference.
5. Run focused tests, appropriate CPU regressions and the repository preflight.
   The base preflight already fails `check-env-doc` and its script test because
   `VT_PASCAL_KQUANT_PREFILL` is undocumented. Record unchanged baseline failures.
6. Run the real public CLI with `Hello` and greedy short decode, and compare
   identical prompt token IDs and greedy output with the supplied llama.cpp.
   Record hashes, exact commands, binary revisions, thread and attention settings.
   Token differences remain an open correctness finding until explained and fixed.
7. A fresh reviewer performs static review and scratch mutations for each claimed
   guarantee, including deleting the production call site. Restore every mutation.
   The operator reruns focused verification on the immutable reviewed head.

Record performance measurements only after the numerical gate. This repair
makes no throughput, latency or memory superiority claim without matched traces
and repeated idle-host measurements. Public usage changes describe accepted
formats, local artifact provenance and explicitly refused arms. A local file
without an upstream revision must be identified as such, never assigned a fake pin.

## Git integration

The existing preference selects one issue per PR with the spec committed first.
The developer explicitly requests a separate branch and an upstream Draft PR
on 1 November 2026. Push the task branch to the authorized fork and open a
Draft against `localai-org/vllm.cpp`. Do not merge or change shared main.

## Outcome

Pending implementation, independent review and operator verification.

## Comparator evidence: 1 November 2026

The original GGUF sha256 is
`f8428df24ae3961c98ac4137c7cb47c1b6326a45690aa3677601fe2d70b1c69c`.
The CT safetensors sha256 is
`1ec184294c1fda2cfe6281f600478f6fd31b87b23d56aff39dd1f89e455792d8`.
The converter sha256 is
`f3c73977c0fcc14b44c29078c181f1bed1aad3c2093bd6cd118e7b6cc381d572`.
These are local developer-supplied artifacts. Their original publisher revision
is unavailable. No HuggingFace revision is inferred from their file names.

The installed llama.cpp library identifies commit
`36a623af60241c1bf31b20ee94aa086261563349`, build 12164. Its sha256 is
`817cefbaa1aec4d6869d0ef884dad26d81b059e21146e6a8f0257d01d1df97f1`.
The raw completion binary sha256 is
`bb1610e0ed8f98a3d75ea74d4c028eea939269a480a40572f994986cd68f7638`.
A public-API diagnostic uses headers archived from that exact object because
current checkout headers have a different context-parameter ABI.

The numerical repack check covers 39 tensor/expert samples and 156 rows across
layers 0, 11 and 29 and experts 0, 57 and 115. All sampled codes and F16 scale
bits match. Maximum sampled dequantized difference is zero. Dense and expert
Marlin padding contains zero values. Embeddings match GGUF dequantization
rounded to F16, with maximum sampled rounding difference 0.000030517578125.
Converting the F16 embedding table to BF16 adds avoidable rounding, so preserve
the F16 table and use converting lookup operations.

The original GGUF executed on CPU with four threads, context 2048, batch 32,
ubatch 32, F16 KV, flash attention enabled and weight repacking enabled.
Raw `Hello` tokenizes to `[2, 9259]`. Sixteen greedy output IDs are
`[2180, 236786, 31048, 236786, 31048, 236786, 31048, 236786,
31048, 236786, 31048, 236786, 31048, 236786, 31048, 236786]`.
The diagnostic matches `llama-completion` output. Its 16 by 262144 float32
logits capture has sha256
`83a48a2355f6370f2db595aec70edb97cc030bd1dc1691d43794d7bc3aa83b3a`.
The modern `llama-cli` applied conversation formatting despite the requested
raw mode, so its earlier eight-token output is not the matched comparison.

Raw comparator command, exit 0:

```sh
~/aikar-engine/build/bin/llama-completion \
  -m /mnt/openwebui/AIKAR/Lumen-3.1-Pulsar/Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf \
  -ngl 0 -t 4 -c 2048 -b 32 -ub 32 -n 16 -p Hello \
  --temp 0 --seed 1 --no-conversation --single-turn --no-warmup \
  --verbose-prompt --simple-io
```

These timings establish execution only. They are not performance measurements.
The implementing engine's real-model numerical comparison remains pending.

## Owed

- `ISSUE-LOCAL-01M6AS6AFMM9G0D76SKPJGT9SH`: connect the staged shared GeGLU
  descriptor to the production Gemma4 CT loader and registered forward in this
  PR, then complete the declared tests, fresh review and operator verification.
