ID: ISSUE-LOCAL-01M64S95TXSC1F8382E8WPDTZC
Title: Complete typed FP16 kept-quant CUDA projections
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

Registered kept-quant CUDA projections reject FP16 output or encode two-byte output as BF16. Pascal model dtype cannot propagate through its actual Q4_K, Q5_0, Q6_K and Q8_0 projections.

## Resolution

-
