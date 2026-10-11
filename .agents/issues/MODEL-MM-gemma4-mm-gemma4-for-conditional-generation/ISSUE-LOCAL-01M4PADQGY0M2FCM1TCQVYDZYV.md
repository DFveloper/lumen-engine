ID: ISSUE-LOCAL-01M4PADQGY0M2FCM1TCQVYDZYV
Title: Honor Gemma reasoning request kwargs
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

Serving constructs Gemma reasoning by name alone and drops request enable_thinking. T03C4 bounds request-aware construction, not HTTP traversal or tokenizer integration.

## Resolution

-
