ID: ISSUE-LOCAL-01M4P1A9D5ED9CY2YT4229BNB9
Title: Prepare Gemma engine state from prompt IDs once per request
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

ParserEngine parse_delta cannot carry prompt IDs, so prompt-open reasoning is emitted as content before the first close.

## Resolution

-
