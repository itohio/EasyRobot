---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "maps values to binary-reflected Gray code (`g = b ^ (b>>1)`) and back (prefix XOR), so adjacent…"
source: [x/math/dsp/coding/gray/gray_encoder.go, x/math/dsp/coding/gray/gray_decoder.go]
tests: [x/math/dsp/coding/gray/gray_roundtrip_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/math/dsp/coding/gray → ok, 5/5 PASS (2026-11-09); probe: widths 4 and 8 round-trip, widths 1,2,3,5,6,7 destroy data (0xce→0x02 for width 2)"
---
# Gray code mapping (binary-reflected)

**What it does:** maps values to binary-reflected Gray code (`g = b ^ (b>>1)`) and back (prefix XOR), so adjacent symbol values differ in one bit — useful when a demodulator's likely error is "neighbouring symbol" (MFSK tone ±1, PSK/QAM neighbour).

**Where:** `gray/gray_encoder.go:37-41` (`encodeToGray`), `Process` :92-110; `gray_decoder.go:36-45` (`decodeFromGray`), `Process` :96-114. `NewEncoderWithBits(1..8)`.

**Inputs → outputs:** `chain.Processor[byte,byte]`, 1:1 bytes. Width 8: whole byte; width 4: each nibble separately; any other width: only the low `bits` of each byte are coded, the rest **are discarded** (`gray_encoder.go:106-109`, decoder :110-113).

## Completeness review

- Complete: no — widths other than 4 and 8 do not split the byte into symbols; no packing of non-byte-aligned symbols (3, 5, 6, 7 bits).
- Tested: 5 tests pass: 8-bit and 4-bit roundtrips, edge cases, adjacent-values property, and `TestGray_AllBitWidths` — which calls the private `encodeToGray/decodeFromGray` on values `< 2^bits` only, so it never exercises `Process` for other widths.
- Correct: widths 4 and 8 yes (tests + probe); widths 1,2,3,5,6,7 no (probe: width 2 `ce`→`02`, width 7 `e7`→`67`).
- Integration note: `cmd/audio` uses `gray.NewEncoderWithBits(8)` (`cmd/audio/common.go:479`). Gray only helps if the code width equals the modulator's bits/symbol (e.g. 2 for 4-FSK/QPSK, 4 for 16-QAM); 8-bit Gray on bytes feeding a 2-bit/symbol modulator gives no benefit [INFERENCE from the mapping, not measured]. The needed width-2 path is the broken one.

## TODOs

- [ ] Split each byte into `8/bits` symbols for widths 1 and 2 (and bit-pack for 3,5,6,7) instead of discarding high bits — gray/gray_encoder.go:106-109, gray_decoder.go:110-113 #p1 #ER-023
- [ ] Test `Process` roundtrip for every width (not only the private helpers) — gray/gray_roundtrip_test.go #p2
- [ ] Match Gray width to modulator bits/symbol in `cmd/audio` — cmd/audio/common.go:479 #p2

## Links

- Module: [[line-coding-and-fec]]
- Related: [[modulation]], [[audio-modem-cli]]
