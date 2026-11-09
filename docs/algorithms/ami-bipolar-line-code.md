---
type: algorithm
status: stub
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/ami/ami_encoder.go, x/math/dsp/coding/ami/ami_decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/coding/probe (2026-11-09) → encoder output == input for every payload (identity)"
---
# AMI (alternate mark inversion)

**What it does (claimed):** bipolar 3-level code: `0` → 0 V, `1` → alternating +V/−V, giving DC balance and single-violation error detection.

**What it actually does:** identity. Both polarity branches set the same bit (`ami/ami_encoder.go:92-99`: `encodedByte |= 1<<bit` in the positive and the negative case); the polarity toggle at :100 has no effect on output. The decoder copies set bits (`ami_decoder.go:78-83` region). `lastPolarity` in the decoder is unused.

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes. A byte stream cannot carry 3 levels; a real AMI needs a ternary symbol type (e.g. `int8` or 2 bits per symbol) or direct generation of ±1 samples in the modulator.

## Completeness review

- Complete: no — placeholder (SPEC.md "Simplified implementation: full AMI requires 3-level detection").
- Tested: no tests.
- Correct: no as AMI; as a byte pipe it round-trips trivially (probe: identity encoder, roundtrip true).

## TODOs

- [ ] Implement AMI on a ternary symbol type (or emit ±1/0 float samples) with violation detection — ami/ami_encoder.go:87-103 #p2
- [ ] Until then, document as pass-through or remove from the catalog — x/math/dsp/coding/SPEC.md #p3

## Links

- Module: [[line-coding-and-fec]]
- Related: [[hdb3-line-code]], [[b8zs-line-code]]
