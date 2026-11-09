---
type: algorithm
status: stub
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
source: [x/math/dsp/coding/ppm/ppm_encoder.go, x/math/dsp/coding/ppm/ppm_decoder.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe tmp/easyrobot/coding/probe (2026-11-09) → roundtrip loses bit 0 for 2/4/8 positions (0xbb→0xba, 0x8d→0x8c)"
---
# PPM (pulse position modulation) byte code

**What it does (claimed):** M-PPM: each log2(M)-bit symbol becomes one pulse in one of M slots (2/4/8-PPM), power-efficient for IR/optical.

**What it actually does:** a 1:1 bit-pair permutation that ignores `positions` entirely (same code path for 2, 4, 8). Encoder takes `(byte >> (7-2p)) & 3` for p = 0..3 (`ppm/ppm_encoder.go:92-97`), i.e. shifts 7,5,3,1 instead of 6,4,2,0, so bit 0 is never read; decoder places pairs back at the same wrong offsets (`ppm_decoder.go:85-89`). Result: LSB of every byte is lost.

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes; `NewEncoderWithPositions(2|4|8)` panics otherwise.

## Completeness review

- Complete: no — no slot expansion (an M-PPM symbol needs M chips), positions parameter unused; SPEC calls it "simplified".
- Tested: no tests.
- Correct: no — probe roundtrip fails for 2/4/8 (LSB cleared), also in streaming.

## TODOs

- [ ] Implement real M-PPM (symbol → one-hot M-chip frame, decoder = argmax/energy per slot, soft output) — ppm/ppm_encoder.go:85-100 #p2
- [ ] Until then fix the shift (`6-2p`) so the pass-through is at least lossless — ppm/ppm_encoder.go:93, ppm_decoder.go:86 #p1 #ER-023

## Links

- Module: [[line-coding-and-fec]]
