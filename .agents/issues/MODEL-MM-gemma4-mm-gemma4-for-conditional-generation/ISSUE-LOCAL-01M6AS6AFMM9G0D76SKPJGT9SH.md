ID: ISSUE-LOCAL-01M6AS6AFMM9G0D76SKPJGT9SH
Title: Gemma4 compressed-tensors F16 checkpoint fails in dense embedding loader
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: -

## Problem

Lumen-3.8-Pulsar-it-qat-w4a16-ct declares dtype float16 and contains F16 embeddings. CPU CLI rejects the embedding in MaterializeBf16Source before loading mixed W4A16/W8A16 packed projections. Verify configured dtype, packed weights and production decode.

## Resolution

-
