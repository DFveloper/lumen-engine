ID: ISSUE-LOCAL-01M63NES5TYK3E3WH5B50XVAPP
Title: Legacy prefill exceeds Pascal shared memory for head256
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

CUDA_LEGACY accepts head dimension 256, but portable prefill allocates 65536 bytes on P620 with a 49152 byte per-block limit. Public load succeeds and prompt16 completion fails.

## Resolution

-
