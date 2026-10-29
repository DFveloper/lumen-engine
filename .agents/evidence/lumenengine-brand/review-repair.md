# Independent review repairs

Date: 29 October 2026
Issue: `ISSUE-LOCAL-01M63MYTWBYSRRCEMAY20QR08M`
Reviewed base: `bc9354cc300aa875ec8a8d9faf0962bb004504a2`

The original C11 and C++20 callers included both headers. Removing the forwarding
include from a scratch `lumenengine.h` left both syntax checks green. Each new
caller now includes `lumenengine.h` alone. Separate legacy callers include
`vllm.h` alone. The package gate exercises both languages for both names.

Commands: `cc -std=c11 -Werror -fsyntax-only -I<headers> tests/branding/consumer.c`
and the C++20 equivalent. The reviewed callers exited 0 against the broken
scratch header. The repaired callers both exited 1. The production header stayed
unchanged. Both repaired callers exited 0 against production headers.

The original command checker compared the two version outputs without a pin.
Both scratch server binaries replaced `vllm.cpp` with `wrongver`. The original
checker exited 0. The repaired checker exited 1 with `legacy server version
changed`. The repaired checker exited 0 on unchanged production commands.
The historical output comes from the existing tracked evidence artifact.

Mutation binaries and scratch headers were separate copies. No production
artifact or header was modified. Full repaired packaging verification is PENDING
until the operator-approved fast-forward makes the tests available at the
existing build source path.
