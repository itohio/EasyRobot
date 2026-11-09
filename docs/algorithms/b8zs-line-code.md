---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/b8zs/b8zs_encoder.go, x/math/dsp/coding/b8zs/b8zs_decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/coding/probe (2026-11-09) → roundtrip fails (0x2d→0x24, 0xFF→enc 0xAA→dec 0xAA, 0x00 0xFF→01 55)"
---
# B8ZS (bipolar with 8-zero substitution)

**What it does (claimed):** AMI where eight consecutive zeros are replaced by `000VB0VB` (bipolar violations) so T1 receivers keep clock.

**What it actually does:** a lossy binary mapping. Marks are written only when `nextPolarity` is true (`b8zs/b8zs_encoder.go:84-90`), so every second `1` is emitted as `0`; on the 8th zero a single `1` is set (:93-99) instead of the 8-chip substitution. The decoder (`b8zs_decoder.go:76-89`) copies set bits and never removes substitutions. Encoding is not invertible.

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes.

## Completeness review

- Complete: no — "simplified" per SPEC; no ternary output, no substitution pattern, decoder has no inverse.
- Tested: no tests.
- Correct: no — probe: `FF`→`AA`, `00 FF`→`01 55`; roundtrip false on every random payload; byte-by-byte streaming also wrong.

## TODOs

- [ ] Rewrite on a ternary symbol type with proper 000VB0VB substitution and inverse in the decoder — b8zs/b8zs_encoder.go:82-100 #p1 #ER-023
- [ ] Add roundtrip test that would have caught the lossy encoder — b8zs/ #p2

## Links

- Module: [[line-coding-and-fec]]
- Related: [[ami-bipolar-line-code]], [[hdb3-line-code]]
