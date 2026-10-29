ID: ISSUE-LOCAL-01M640A0QTNZSV3WXBHHN377K0
Title: Mirror Pascal Q6_K MMQ prefill activation format
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

Actual Q6_K prefill uses per32 F32 D4 activation scales in the GPU oracle, but production contracts Q8_K per256 scales.

## Resolution

-
