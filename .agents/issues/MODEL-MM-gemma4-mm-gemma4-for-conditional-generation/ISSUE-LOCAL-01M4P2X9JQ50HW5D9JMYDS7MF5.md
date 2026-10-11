ID: ISSUE-LOCAL-01M4P2X9JQ50HW5D9JMYDS7MF5
Title: Gemma reasoning registry lacks prompt-aware adapter
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

The reasoning factory does not register gemma4 and its ABC cannot forward prompt preparation. Add only the bounded registry and adapter surface; served prompt wiring remains separate.

## Resolution

-
