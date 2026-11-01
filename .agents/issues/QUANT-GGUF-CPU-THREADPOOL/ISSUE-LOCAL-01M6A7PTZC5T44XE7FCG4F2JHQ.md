ID: ISSUE-LOCAL-01M6A7PTZC5T44XE7FCG4F2JHQ
Title: CPU decode: paged attention runs its per-head loop inline at T=1
Row: QUANT-GGUF-CPU-THREADPOOL
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: -

## Problem

At decode the paged-attention kernel splits work over query tokens only, so T=1 hands ParallelForRows one row and the caller runs every q head of every key single-threaded while the pool workers spin. The W5 SIGPROF and VT_POOL_GAP profile measured 1,596 calls totalling 0.303 s of a 6.7-7.0 s 32-token greedy request (190 us per call) in the inline body, which is 5.5 percent of the current 5.495 s wall; the per-head output elements are independent, so the same output-range split the W5 kernels use applies once the per-token scratch is rescoped per head.

## Resolution

-
