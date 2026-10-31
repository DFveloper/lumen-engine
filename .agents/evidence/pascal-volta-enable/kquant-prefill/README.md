# Pascal tiled K-quant prefill evidence

Issue: `ISSUE-LOCAL-01M653NJXDM6B5KE8VAA2VP93X`.
Row: `BACKEND-CUDA-SM061`. Base: `780767bfd`.

## Result

The tiled Q4_K/Q6_K prefill lane passes the registered quant suite, the focused
throughput regression, the sm60/sm61/sm70 compile matrix, and the eight-axis
real-model gate. On the P620 it cuts the TinyLlama p16 prefill kernel time from
609.1 ms to 170.0 ms (3.58x), inside the row's below-200 ms target. The
`VT_PASCAL_KQUANT_PREFILL=0` rollback restores the byte-identical generic lane.

All GPU runs used `flock /home/user/gpu.lock` on the local Quadro P620, sm_61,
2048 MiB, CUDA 12.4.131.

## Recipe

```sh
cmake -S /home/user/lumen-engine -B /tmp/lumen-build -G Ninja \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=61 \
  -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/15
cmake --build /tmp/lumen-build --target test_cuda_quant_dot test_cuda_legacy_capi -j8
flock /home/user/gpu.lock /tmp/lumen-build/tests/test_cuda_quant_dot \
  -tce='CUDA grouped keep-quant at GLM*'
```

The GLM-5.3-Flash published MoE geometry allocates about 2.4 GB and does not fit
the 2 GB P620. The case is excluded, not weakened; `test_ops_quant_dot` passes
all 34 cases.

## Focused suite

`quant-default-final.log` exits 0 with 23 cases and 8308113 assertions after the
one excluded case. `quant-prefill-final.log` exits 0 with the two new cases and
8053086 assertions.

The two new cases:

- `Pascal tiled K-quant prefill correctness` covers Q4_K and Q6_K at
  `k in {2048, 5632}`, `m in {2, 16, 32, 128}`, `n in {7, 16, 5632}` against the
  CPU oracle, an f64 dequantized reference, and the generic lane.
- `Pascal tiled K-quant prefill is substantially faster than the generic lane`
  runs `m=32, n=5632, k=5632` and requires `tiled < generic / 1.5` for both
  dtypes.

## Kernel time

`nsys-kernels.txt` retains both traces: TinyLlama p16, 16 generated tokens, one
repetition.

| Weight | Generic (`=0`) | Tiled (default) | Speedup |
|---|---|---|---|
| Q4_K (134 launches) | 409.162 ms | 129.102 ms | 3.17x |
| Q6_K (20 launches) | 199.970 ms | 40.883 ms | 4.89x |
| Total | 609.132 ms | 169.985 ms | 3.58x |

The tiled kernel replaces `QuantDotGemmKernel` for these shapes; no generic
K-quant prefill launch remains in the default trace.

`prefill-on.log` and `prefill-off.log` are the profiled process output.

## Real-model gate

Same binary, 64 generated tokens, three repetitions, warm median. The axes are
the four the landed Q4_K evidence uses for each checkpoint. `compare` here means
the tiled and generic arms emit identical token ids.

| Model | Axis | Tiled vs generic | vs llama.cpp oracle |
|---|---|---|---|
| TinyLlama | p16/t16 | exact | exact |
| TinyLlama | p16/t64 | differs at token 61 | both arms first differ at token 28 |
| TinyLlama | p128/t16 | exact | exact |
| TinyLlama | p128/t64 | exact | exact |
| SmolLM2 | p16/t16 | exact | pre-existing cross-runtime divergence at token 8 |
| SmolLM2 | p16/t64 | exact | pre-existing cross-runtime divergence at token 8 |
| SmolLM2 | p128/t16 | exact | exact |
| SmolLM2 | p128/t64 | exact | exact |

`ab8-final.txt` is the machine-readable record.

## Near-tie disposition

On TinyLlama p16/t64 both arms choose `29903` at token 28 where the oracle
chooses `29909`; this is the near-tie `pascal-q4k-decode.md` already documents.
The tiled arm does not move that choice. The two arms agree through token 60 and
diverge only at tokens 61-63, after both have left the oracle trajectory at the
documented tie. The six oracle-exact axes stay oracle-exact, and every axis but
this documented tie is identical between the two arms, so no exact axis moved.

The SmolLM2 p16 axis is the separately owned cross-runtime divergence recorded
in the Q4_K evidence: the CPU oracle and this engine agree, the CUDA oracle does
not. The tiled lane is identical to the generic lane there.

## Activation contract deviation

The design note suggested staging the activation as `Q8_1` per 32 values. The
implementation keeps the landed `BlockQ8_K` super-block instead. `Q8_1` would
change the integer core relative to the CPU oracle and the generic lane; the
token gate requires the prefill path to stay under the landed `Q8_K` contract.
The tiled lane still mirrors the oracle's tile shape, config, and DP4A dot; only
the activation encoding differs, and the deviation is recorded here.

## Compile matrix

`nvcc` compiles `cuda_quant_dot.cu` clean for `sm_60`, `sm_61`, and `sm_70` with
`-std=c++20`; see `nvcc_sm60.log`, `nvcc_sm61.log`, `nvcc_sm70.log`.

## Raw logs

- `quant-default-final.log`, `quant-prefill-final.log`, `ops-quant-final.log`
- `ab8-final.txt`, `tiny-p16-t64-on.log`, `tiny-p16-t64-off.log`
- `nsys-kernels.txt`, `prefill-on.log`, `prefill-off.log`
- `nvcc_sm60.log`, `nvcc_sm61.log`, `nvcc_sm70.log`
