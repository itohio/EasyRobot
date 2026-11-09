---
type: algorithm
status: partial
tags: [easyrobot, dsp, coding, line-code]
workstream: W7
module: "[[line-coding-and-fec]]"
summary: "maps every data bit to a two-chip symbol with a guaranteed mid-bit transition: standard (G.E."
source: [x/math/dsp/coding/manchester/encoder.go, x/math/dsp/coding/manchester/decoder.go, x/math/dsp/coding/manchester/init.go]
tests: [x/math/dsp/coding/manchester/manchester_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/math/dsp/coding/manchester → ok, 15/15 top-level PASS (2026-11-09); probe tmp/easyrobot/coding/probe → aligned roundtrip OK, SetBitOffset odd offsets wrong"
---
# Manchester line code (G.E. Thomas and IEEE 802.3)

**What it does:** maps every data bit to a two-chip symbol with a guaranteed mid-bit transition: standard (G.E. Thomas) 0→`01`, 1→`10`; IEEE 802.3 0→`10`, 1→`01`. Rate 1/2, DC-free, self-clocking.

**Where:** `manchester/init.go:20-88` (4-bit → 8-chip encode tables and 4-chip → 2-bit decode tables), `encoder.go:44-107` (`Encoder.Process`, `NewEncoder`, `NewEncoderIEEE8023`), `decoder.go` (`Decoder.Process` :83, `processBytes` :147, `decodeWithBitOffset` :188, `decodeFastPath` :257, `decodeRemaining` :295, `decodeManchesterSymbol` :378, `SetBitOffset` :70).

**Inputs → outputs:** `chain.Processor[byte,byte]`; bytes MSB first → 2 bytes per input byte (chips packed MSB first). Decoder: 2 bytes → 1 byte; keeps a partial output byte (`currentByte`, `bitPos`) across `Process` calls. `SetBitOffset(0..7)` skips N leading chips.

## Completeness review

- Complete: no. Invalid chip pairs (`00`/`11`) decode silently as `0` in the table path (`init.go:74,82`) — no erasure/error flag, so a downstream FEC cannot use them. No clock/phase recovery: alignment must be known.
- Tested: 15 tests pass. They check exact encoded bytes for both conventions, size queries, roundtrips (several sizes), partial-byte feeding across batches. `TestManchesterDecoder_BitOffset` and `_MisalignedSignal` call `SetBitOffset` **then `Reset()`**, which zeroes the offset (`decoder.go:400`), and only log results — the offset path is effectively untested.
- Correct: aligned path yes (probe: random 1/7/300/1000-byte payloads and byte-by-byte streaming round-trip). `SetBitOffset` no: probe with a genuinely shifted stream → offsets 2 and 4 decode 40/40 bytes, offsets 1, 3, 7 give 0–1 correct bytes. Cause: for odd offsets the symbol straddling a byte boundary resets `bitOffset = 0` (`decoder.go:202-209`) instead of 1, re-reading a consumed chip. Also the offset branch re-runs whenever a chunk starts with `bitPos == 7` (`decoder.go:157`), which would re-skip bits mid-stream (not hit in the probe because 16-byte chunks never re-align).
- Polarity: an inverted channel decodes to the bitwise complement (probe `rt-after-polarity-inversion = no`); use differential biphase ([[biphase-fm0-differential-manchester]]) when the audio path may invert.

## TODOs

- [ ] Fix odd `SetBitOffset` handling (carry chip offset across byte boundary) — decoder.go:202-209 #p1 #ER-023
- [ ] Apply bit offset only once per stream, not whenever `bitPos == 7` — decoder.go:157 #p1 #ER-023
- [ ] Make offset tests real: remove the `Reset()` after `SetBitOffset` and assert decoded bytes — manchester_test.go:401-430 #p2
- [ ] Report invalid chip pairs (erasure count or soft output) so FEC can use them — init.go:74-82 #p2
- [ ] Add preamble/sync-word based alignment instead of a manual offset — decoder.go:70 #p2

## Links

- Module: [[line-coding-and-fec]]
- Workstream: [[W7-filters-codecs-voice]]
