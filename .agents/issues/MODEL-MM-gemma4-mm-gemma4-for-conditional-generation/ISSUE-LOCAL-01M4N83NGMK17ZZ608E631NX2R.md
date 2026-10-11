ID: ISSUE-LOCAL-01M4N83NGMK17ZZ608E631NX2R
Title: Discriminate vision pooling rounding and excess weight storage
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: -

## Problem

Review of 3c799f892 finds two test coverage gaps, not known production bugs: pool-no-round survives width4 scale2 and attention >= size survives absent oversized cases. Add independent non-power-of-two pooling and exact shape refusals; prove both compiled mutations fail.

## Resolution

-
