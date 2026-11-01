ID: ISSUE-LOCAL-01M6AV8N13FD188YMN1TN3N8CN
Title: Gemma4 MoE CPU forward throws in DualRmsNormPlusRes
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

The production Gemma4 MoE forward unconditionally calls vt::DualRmsNormPlusRes. Its CPU arm throws CPU compose not yet wired, so the compact CT production fixture cannot emit logits after loading. Implement the shared composed normalization and residual operation and verify its order and rounding through the production fixture.

## Resolution

-
