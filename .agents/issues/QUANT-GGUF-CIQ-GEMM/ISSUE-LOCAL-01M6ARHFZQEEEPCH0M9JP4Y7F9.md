ID: ISSUE-LOCAL-01M6ARHFZQEEEPCH0M9JP4Y7F9
Title: Compose NEON Q4_K dispatch tests with the AVX2 tier
Row: QUANT-GGUF-CIQ-GEMM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: -

## Problem

After upstream merge, the NEON Q4_K dispatch test assumes that no x86 AVX2 tier wraps its selector. Default x86 execution fails pointer and bit-identity assertions despite valid AVX2 dispatch.

## Resolution

-
