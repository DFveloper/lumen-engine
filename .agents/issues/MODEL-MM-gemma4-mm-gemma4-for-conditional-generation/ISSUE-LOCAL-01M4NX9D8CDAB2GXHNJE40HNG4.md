ID: ISSUE-LOCAL-01M4NX9D8CDAB2GXHNJE40HNG4
Title: Match Google empty schema mapping truthiness
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

Independent T03A review F1 at f340f218 finds extra items:{} for array items={} versus original Google Jinja2 bytes. Audit analogous mapping truth tests and add independent nested regressions before exact-hash-only repair.

## Resolution

-
