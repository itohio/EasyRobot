---
type: algorithm
status: implemented
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "`1` → transition in mid-cell; `0` → no mid-cell transition, plus a start-of-cell transition only…"
source: [x/math/dsp/coding/miller/miller_encoder.go, x/math/dsp/coding/miller/miller_decoder.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "no repo tests; probe (2026-11-09) → roundtrip OK, streaming OK, survives polarity inversion, max run 4 chips"
---
# Miller (delay) code

**What it does:** `1` → transition in mid-cell; `0` → no mid-cell transition, plus a start-of-cell transition only when the previous bit was also `0`. Rate 1/2 in chips, narrower spectrum than Manchester, DC nearly balanced (49.7 % ones in probe), max run 4 chips.

**Where:** `miller/miller_encoder.go:83-122` (`Encoder.Process`; rule at :92-103, state `lastState`, `lastBit`), `miller/miller_decoder.go:94-140` (mid-transition ⇒ `1`, :124-128).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1 byte → 2 bytes of chips (MSB first); decoder 2 → 1, buffers an odd trailing byte.

## Completeness review

- Complete: yes for the stated scope. Decoder does not check the start-of-cell rule (no violation detection); `lastBit` is tracked but unused in decoding.
- Tested: no tests in package.
- Correct: yes — probe roundtrip of random payloads (1–1000 B), all-zero/all-one blocks and byte-by-byte streaming; decodes after chip inversion because only mid-cell transitions matter.
- Voice-channel fit: usable where bandwidth is tight; worse clock content than FM0 for long `0` runs (run 4).

## TODOs

- [ ] Add roundtrip/streaming/inversion tests — miller/ #p2
- [ ] Detect start-of-cell rule violations for error flags — miller/miller_decoder.go:121-132 #p3

## Links

- Module: [[line-coding-and-fec]]
