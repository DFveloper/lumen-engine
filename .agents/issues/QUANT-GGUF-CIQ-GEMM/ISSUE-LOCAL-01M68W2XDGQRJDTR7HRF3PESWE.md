ID: ISSUE-LOCAL-01M68W2XDGQRJDTR7HRF3PESWE
Title: x86 CPU quantized GEMM runs the portable vec_dot tier only
Row: QUANT-GGUF-CIQ-GEMM
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

The Flare GSQ IQ3_M serving profile spends most decode time in the portable IQ3_S/Q4_K/Q6_K vec_dot kernels. The CPU has AVX2/FMA and llama.cpp b10451 is ~3.5x faster end to end on the same model, but BlockVecDot never selects an x86 SIMD tier. Add runtime-gated AVX2 kernels for the three dtypes that dominate this model and gate them against the f64 reference.

## Resolution

-
