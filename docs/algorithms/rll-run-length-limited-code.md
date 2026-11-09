---
type: algorithm
status: stub
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/rll/rll_encoder.go, x/math/dsp/coding/rll/rll_decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/coding/probe (2026-11-09) → encoder output == input (identity), max run 512 bits on all-zero input"
---
# RLL(d,k) run-length-limited code

**What it does (claimed):** RLL(1,7)/RLL(2,7) constrained codes for magnetic storage (≥d and ≤k zeros between ones).

**What it actually does:** copies input to output (`rll/rll_encoder.go:89-90`, `rll_decoder.go:81-82`). `d`, `k` are validated in the constructor and then unused. Source comment calls it a "placeholder" (`rll_encoder.go:74`).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes.

## Completeness review

- Complete: no — stub.
- Tested: no tests.
- Correct: no as RLL (probe: identity; all-zero input keeps a 512-bit zero run).

## TODOs

- [ ] Implement RLL(1,7) / (2,7) with the standard variable-length tables (or drop the package) — rll/rll_encoder.go:72-90 #p3

## Links

- Module: [[line-coding-and-fec]]
