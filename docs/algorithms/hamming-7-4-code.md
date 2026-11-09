---
type: algorithm
status: tested
tags: [easyrobot, dsp, coding, fec]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "each nibble → 7-bit codeword (parity at positions 1, 2, 4; data at 3, 5, 6, 7), decoder computes a…"
source: [x/math/dsp/coding/hamming/hamming_encoder.go, x/math/dsp/coding/hamming/hamming_decoder.go]
tests: [x/math/dsp/coding/hamming/hamming_test.go, x/math/dsp/coding/hamming/hamming_roundtrip_test.go, x/math/dsp/coding/hamming/hamming_lossy_test.go, x/math/dsp/coding/hamming/hamming_streaming_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/math/dsp/coding/hamming → ok, 20/20 PASS (2026-11-09); probe exhaustive: 3584/3584 single-bit errors corrected, 10752/10752 double-bit errors silently mis-decoded (as theory for plain (7,4))"
---
# Hamming(7,4) single-error-correcting code

**What it does:** each nibble → 7-bit codeword (parity at positions 1, 2, 4; data at 3, 5, 6, 7), decoder computes a 3-bit syndrome and flips the indicated bit. Corrects any single bit error per codeword. Rate 4/7 in information, but stored one codeword per byte → 2 output bytes per input byte (rate 1/2 on the wire).

**Where:** `hamming/hamming_encoder.go:29-56` (`encodeNibble`), `Process` :63-148 (low nibble first, then high); `hamming_decoder.go:30-72` (`decodeCodeword`, syndrome at :44-56), `Process` :86-184 (pairs of codewords → byte, buffers an odd trailing byte).

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1 byte → 2 bytes (bit 7 unused), decoder 2 → 1.

## Completeness review

- Complete: yes for (7,4) SEC. The header comments claim it also "detects double-bit errors" (`hamming_encoder.go:13`, `hamming_decoder.go:12`) — false; that needs the extended (8,4) SECDED code with an overall parity bit, and bit 7 is free for exactly that. `decodeCodeword` returns `errorCorrected` but `Process` discards it (`:151-152`), so callers get no error statistics.
- Tested: 20 tests pass: exact codewords, size queries, roundtrips, every single-bit error position corrected, random single-bit errors, multi-chunk streaming encoder/decoder interleaving. `TestHamming_LossyRoundtrip_DoubleBitErrors` only logs (no assertion).
- Correct: yes for its scope. Probe (all 256 bytes × 2 codewords): 3584/3584 single-bit errors corrected; 10 752/10 752 two-bit errors inside one codeword decode to wrong data with no signal; flipping the unused bit 7 is harmless.
- Performance: one `dst.Write([]byte{x})` per output byte (`hamming_encoder.go:123,130`, `hamming_decoder.go:158`) — a heap allocation + interface call per byte [INFERENCE: escape not checked with `-gcflags=-m`].
- Voice-channel fit: corrects isolated bit errors only; audio channels produce bursts (fades, clicks) → needs interleaving across codewords to be useful.

## TODOs

- [ ] Add SECDED (8,4) variant using bit 7 and expose detected-uncorrectable count — hamming/hamming_decoder.go:30-72 #p2
- [ ] Fix "detects double-bit errors" comments — hamming/hamming_encoder.go:13, hamming_decoder.go:12 #p3
- [ ] Batch writes (one `dst.Write` per chunk) — hamming/hamming_encoder.go:123-134, hamming_decoder.go:158 #p3
- [ ] Optional bit packing (7 bits/codeword) and block interleaver for burst channels — hamming/ #p2

## Links

- Module: [[line-coding-and-fec]]
- Workstream: [[W7-filters-codecs-voice]]
