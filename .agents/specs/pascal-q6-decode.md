# Pascal Q6_K GPU decode activation contract

## Now

ACTIVE. Actual SmolLM Q6_K down projection uses the wrong activation quantization.
Issue: ISSUE-LOCAL-01M63Z1WJK4DYN2VREM2723PJ2.

## Scope and design

Route dense Pascal Q6_K single-row decode through the existing Q8_1 activation preparation.
Add a separate hardware helper. Port the packed DP4A Q6_K decoder from the registered secondary GPU oracle.
Preserve prefill, grouped paths, Volta, and newer GPU defaults. Preserve the public ABI.

## Upstream anchors

The pinned primary vLLM cannot run this Pascal GGUF workload.
The llama-cpp pin is 62318cd57dc63900ecb153272c0b10f78550855d.
Port ggml/src/ggml-cuda/vecdotq.cuh:627-648 and 1019-1042.
Q8_1 preparation follows quantize.cu:64-101.

## Risks

Float reduction order can differ. This operation correction does not establish full model parity.

## Tests and gates

First require an independent scalar Q6_K decode regression through MatmulBTQuant to fail.
Require focused green and a dispatch-absent mutation failure.
Run all 19 quant cases that fit the P620. Record the large GLM MoE allocation blocker.
Compile sm60, sm61, and sm70. The operator runs actual SmolLM correctness before performance acceptance.

## Evidence

Identical actual oracle down inputs give Q8_1 RMSE 1.06e-7 and Q8_K RMSE 0.0256123.
The largest observed activation-contract output difference is 0.100376.
The independent diagnosis script is /tmp/smol-q6-formula.py.

## Stop conditions

Stop for a required ABI change or a change outside the declared hardware and decode scope.

## Owed

Full model correctness, GPU tracing, and performance acceptance remain with the operator.
