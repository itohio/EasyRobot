---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, scrambler]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "XORs data bits (MSB first) with a Fibonacci LFSR keystream; descrambler runs the identical LFSR."
source: [x/math/dsp/coding/lfsr/scrambler_encoder.go, x/math/dsp/coding/lfsr/scrambler_decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/coding/probe (2026-11-09) → keystream period: 7-bit preset 21 (claims 127), 15-bit 32767 (ok), 23-bit 1 (constant 1s: plain bit inversion); roundtrip OK for all three"
---
# LFSR additive (synchronous) scrambler

**What it does:** XORs data bits (MSB first) with a Fibonacci LFSR keystream; descrambler runs the identical LFSR. Additive/frame-synchronous: both ends must start from the same state (all ones after construction/`Reset`). Bit errors do not multiply, but a slip loses sync until `Reset`.

**Where:** `lfsr/scrambler_encoder.go:22-35` (presets), `nextBit` :57-70 (feedback = parity of `state & polynomial`, shift left, insert at bit 0, mask to `width`), `Process` :121-136. Decoder mirrors it (`scrambler_decoder.go:41-56`, :102-117).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes. `NewEncoderWithPolynomial(poly uint32, width int)`.

## Completeness review

- Complete: no — no self-synchronising (multiplicative) variant, no seed parameter, decoder/encoder duplicated code.
- Tested: no tests.
- Correct: no for 2 of 3 presets. The tap mask uses bits `0..width-1`, but the presets encode the `x^width` term as bit `width` (`0x91` = bits 7,4,0 for width 7; `0xC001` = bits 15,14,0 for width 15), which the mask discards. Probe-measured periods:
  - 7-bit (`0x91`, "x^7+x^4+1"): effective taps 4,0 → recurrence of x^5+x^4+1 = (x²+x+1)(x³+x+1), **period 21**, not 127.
  - 15-bit (`0xC001`): effective taps 14,0 → x^15+x^14+1 (reciprocal of primitive x^15+x+1), **period 32767** ✓.
  - 23-bit (`0x40041`): the constant is bits 18,6,0, not "23 and 18" as commented; 3 taps on the all-ones seed give parity 1 forever → **period 1**, keystream all ones, i.e. the scrambler only inverts bits (probe: all-zero input → max run 512).
  Roundtrip itself is correct for all presets (same LFSR both sides).

## TODOs

- [ ] Fix preset constants to tap masks within `width` (bit k = output k+1 steps ago): x^7+x^4+1 → `0x44` (period 127), x^23+x^18+1 → bits 4,22 = `0x400010` (period 8388607), both verified in a Python model of `nextBit`; add period tests — lfsr/scrambler_encoder.go:22-35 #p1 #ER-023
- [ ] Add roundtrip + keystream-period tests — lfsr/ #p2

## Links

- Module: [[line-coding-and-fec]]
- Workstream: [[W7-filters-codecs-voice]]
