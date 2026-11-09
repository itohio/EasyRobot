---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/hdb3/hdb3_encoder.go, x/math/dsp/coding/hdb3/hdb3_decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/coding/probe (2026-11-09) → encoder panics 'negative shift amount' on input 0xF0 / 0x00 0xFF / random 300 B; lossy where it does not panic (0xFF→0xAA)"
---
# HDB3 (high-density bipolar 3)

**What it does (claimed):** AMI with every run of four zeros replaced by `000V` or `B00V` so that no more than three zeros occur and violations alternate (E1 lines).

**What it actually does:** same lossy "only positive marks" mapping as [[b8zs-line-code]] (`hdb3/hdb3_encoder.go:85-91`), and on the fourth zero sets bit `bit-1` (:104-107). When the fourth zero lands on bit 0, `1 << (bit-1)` is a **negative shift → runtime panic** (Go panics on negative shift counts). Any input with ≥4 trailing zeros in a byte (e.g. `0xF0`, `0x00`) crashes the encoder. Decoder (`hdb3_decoder.go:76-89`) copies set bits; violations are never removed.

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes.

## Completeness review

- Complete: no — "simplified" per SPEC.
- Tested: no tests.
- Correct: no — probe panics for 0xF0, 0x00 0xFF and random 300-byte input; `0xFF` encodes to `0xAA` (information lost).

## TODOs

- [ ] Fix negative shift panic (`1 << (bit-1)` at bit 0) — hdb3/hdb3_encoder.go:106 #p1 #ER-023
- [ ] Rewrite as ternary HDB3 with 000V/B00V substitution, parity of marks since last V, and decoder inverse — hdb3/hdb3_encoder.go:82-112 #p1 #ER-023
- [ ] Add roundtrip test — hdb3/ #p2

## Links

- Module: [[line-coding-and-fec]]
- Related: [[ami-bipolar-line-code]]
