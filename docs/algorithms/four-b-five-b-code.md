---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "each nibble → 5-bit code word from the FDDI/100BASE-TX table (max 3 consecutive zeros, ≥2 ones per…"
source: [x/math/dsp/coding/four5b/fourb5b_encoder.go, x/math/dsp/coding/four5b/fourb5b_decoder.go]
tests: [x/math/dsp/coding/four5b/fourb5b_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -count=1 ./x/math/dsp/coding/four5b → FAIL TestFourB5B_Roundtrip (expected 01 02, got 10 20) (2026-11-09); probe: every non-symmetric byte comes back nibble-swapped"
---
# 4B/5B block code

**What it does:** each nibble → 5-bit code word from the FDDI/100BASE-TX table (max 3 consecutive zeros, ≥2 ones per word). Table `four5b/fourb5b_encoder.go:22-39` matches the standard data symbols 0x0–0xF.

**Where:** encoder `fourb5b_encoder.go:98-121` (high nibble first, then low; one 5-bit symbol per output **byte**), decoder `fourb5b_decoder.go:112-136` (reverse table built at :20-33, invalid → 0xFF).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1 byte → 2 bytes (each holds a 5-bit symbol in bits 4..0). The comment "Output size: input_size × 1.25" (`fourb5b_encoder.go:11`) is wrong — no bit packing, expansion is 2.0.

## Completeness review

- Complete: no — no bit packing (wastes 3 of 8 bits; the 5-bit run-length property does not hold across the zero padding: probe max run 6 chips), no control symbols (J/K/T/R/idle), no NRZI pairing.
- Tested: `TestFourB5B_Roundtrip` fails for 3 of 4 cases (case `{0x00}` passes only because it is nibble-symmetric).
- Correct: no. **Implementation is wrong, test is right:** encoder emits high nibble first, decoder assembles `nibble1 | nibble2<<4` (`fourb5b_decoder.go:129`), swapping nibbles (probe: `dc`→`cd`). Invalid symbols drop the whole byte silently (`:122-126`), shortening the stream with no error signal.

## TODOs

- [ ] Fix nibble order in decoder (`nibble1<<4 | nibble2`) — four5b/fourb5b_decoder.go:129 #p1 #ER-023
- [ ] Pack 5-bit symbols into a bit stream (true 1.25× rate) or fix the size comment — four5b/fourb5b_encoder.go:11,112-120 #p2
- [ ] Report invalid symbols instead of silently dropping bytes — four5b/fourb5b_decoder.go:122-126 #p2

## Links

- Module: [[line-coding-and-fec]]
