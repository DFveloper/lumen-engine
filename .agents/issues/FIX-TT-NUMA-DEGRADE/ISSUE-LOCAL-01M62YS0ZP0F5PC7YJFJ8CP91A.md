ID: ISSUE-LOCAL-01M62YS0ZP0F5PC7YJFJ8CP91A
Title: kolibri1 NUMA interleave refusal reds every test that constructs PolicyGuard on restricted runners
Row: FIX-TT-NUMA-DEGRADE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

CI runners (ubuntu image rollout) refuse set_mempolicy(MPOL_INTERLEAVE) -- kolibri1_numa.h:142 VT_CHECK throws, reding test_kolibri1, test_kolibri1_bf16_arm, test_kolibri1_w2, test_kolibri1_numa (CI run 40712283192 job 122900454192, first seen 2026-10-28T05:37Z run 40639035531). Approved semantics (developer, 2026-10-28): warn + default-policy fallback. Knob: unset = best-effort fallback, 0 = off, 1 = strict (throw by name). Spec: .agents/specs/fix-tt-numa-degrade.md

## Resolution

-
