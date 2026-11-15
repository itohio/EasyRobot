---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/eight10b/eightb10b_encoder.go, x/math/dsp/coding/eight10b/eightb10b_decoder.go, x/math/dsp/coding/eight10b/eightb10b_tables.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe (2026-11-09) → data byte 0xFF is dropped (300-byte payload decodes to 299 bytes); max |symbol disparity| 4; max run 10 chips"
---
# 8B/10B (simplified, RD−-only)

**What it does (claimed):** IBM 8b/10b — 5b/6b + 3b/4b with running disparity, DC-balanced, run ≤ 5, comma symbols.

**What it actually does:** fixed table, no running disparity. `eightb10b_tables.go:64-77` maps the **high** 5 bits through the RD− 5b/6b column and the low 3 bits through the RD− 3b/4b column (standard splits the other way: EDCBA = low 5 bits → 6b, HGF = high 3 bits → 4b). The 6b and 4b columns themselves match the standard RD− codes (checked by hand for all 32 + 8 entries); the 3b/4b table is mislabelled "K.x" (`:44-55`). Output is not interoperable with real 8b/10b.

**Where:** encoder `eightb10b_encoder.go:78-91` (10-bit symbol packed as `byte(sym>>2)`, `byte((sym&3)<<6)`), decoder `eightb10b_decoder.go:94-120` (`Process` :32).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1 byte → 2 bytes (10 used bits).

## Completeness review

- Complete: no — no RD tracking (SPEC "simplified"), no K28.5 comma, no alignment, no bit packing (2.0× instead of 1.25×).
- Tested: no tests.
- Correct: no. **Byte 0xFF is lost:** the reverse table uses 0xFF as the "invalid" marker (`eightb10b_tables.go:86`) and the decoder skips any lookup equal to 0xFF (`eightb10b_decoder.go:104`), so the valid data byte 0xFF is dropped (probe: 300 B → 299 B, all-0xFF block → empty). Without RD, symbols have disparity up to ±4 and runs up to 10 (probe), so neither DC balance nor run-length bound holds.

## TODOs

- [ ] Use a separate valid flag (or 16-bit table with sentinel 0xFFFF) so 0xFF decodes — eight10b/eightb10b_tables.go:82-94, eightb10b_decoder.go:104 #p1 #ER-023
- [ ] Implement standard bit order and running disparity (RD±, alternate codes, K28.5) — eight10b/eightb10b_tables.go:64-77 #p2
- [ ] Add roundtrip test over all 256 bytes and disparity/run-length assertions — eight10b/ #p2

## Links

- Module: [[line-coding-and-fec]]
