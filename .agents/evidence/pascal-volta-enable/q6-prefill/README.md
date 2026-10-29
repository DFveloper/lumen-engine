# Q6_K prefill candidate evidence

Spec: [Pascal Q6_K prefill](../../../specs/pascal-q6-prefill.md).
Issue: ISSUE-LOCAL-01M640A0QTNZSV3WXBHHN377K0.

The committed fixture in `tests/vt/pascal_q6_prefill_fixture.h` contains captured oracle activation row6 and original packed Q6_K weight rows56 to58.
The header records the exact oracle pin, checkpoint revision, checkpoint SHA256, and fixture byte hashes.
The scalar test decodes raw packed Q6_K values without a production decoder.

| Gate | Result |
|---|---|
| Actual-input regression before implementation | FAIL, 48 value mismatches, maximum 0.197464 |
| Synthetic regression before implementation | FAIL, 39177 assertions |
| Candidate focused suite | PASS, 3 cases and 165510 assertions |
| Bounded quant suite | PASS, 20 cases and 255005 assertions |
| Changed TU sm60, sm61, sm70 | PASS |
| Production dispatch deletion | FAIL, 39225 assertions |
| D4 scale narrowed to F16 | FAIL, 39782 assertions |
| Packed subscale index mutation | FAIL, 55083 assertions |
| Restored focused suite | PASS, 3 cases and 165510 assertions |
| Actual SmolLM p16g16, two repetitions | FAIL, token357 at index8 instead of 5649 |
| Actual TinyLlama p16g16, two repetitions | FAIL, token1736 at index12 instead of 338 |

The large published GLM MoE geometry requires more than the available P620 2 GiB memory.
The focused operation results do not establish actual model correctness.
The candidate cannot be accepted while the model regressions remain unresolved.
No golden was changed or model excluded.

Session commands and complete logs live under `/tmp/pascal-q6-prefill-evidence`.
The build driver is `/tmp/q6-prefill-build.py`.
The actual model logs are `/tmp/lumen-smol-q6prefill-candidate.log` and `/tmp/lumen-tiny-q6prefill-candidate.log`.
All GPU runs hold `/home/user/gpu.lock`.
