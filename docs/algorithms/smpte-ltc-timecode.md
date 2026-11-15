---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, timecode]
workstream: W7
module: "[[modulation]]"
summary: "build-fail (stale tests); copy: ok 18"
source: [x/math/dsp/modulation/tc/timecode.go, x/math/dsp/modulation/tc/ltc.go, x/math/dsp/modulation/tc/modulator.go, x/math/dsp/modulation/tc/demodulator.go]
tests: [x/math/dsp/modulation/tc/timecode_test.go, x/math/dsp/modulation/tc/ltc_test.go, x/math/dsp/modulation/tc/demodulator_test.go, x/math/dsp/modulation/tc/e2e_test.go, x/math/dsp/modulation/tc/modulator_test.go]
test_result: build-fail
complete: no
correct: unknown
evidence: "go test ./x/math/dsp/modulation/tc → build failed (stale mpsk API, undefined tc.NewModulator) (2026-11-09); copy minus 3 stale test files in  → ok, 18 pass"
---
# SMPTE LTC timecode (BCD frame + bi-phase mark audio)

**What it does:** marshals an `SMPTETimeCode` (h:m:s:f, drop-frame, user bits) into the 80-bit LTC frame (BCD fields plus sync word), and encodes or decodes it as bi-phase-mark audio at 24/25/29.97/30 fps. It also does time arithmetic (parse, `FromTime`/`ToTime`, add/subtract frames, drop-frame counting).

**Where:** `tc/timecode.go:52-112` BCD/bit helpers, `:114-304` `MarshalLTC`/`UnmarshalLTC`, `:306-491` validate/parse/time maths; `tc/ltc.go:12-63` `EncodeBiPhase`, `:67-123` `DecodeBiPhase`, `:126-177` zero-crossing transitions and nearest-transition search; `tc/modulator.go:16-85` `LTCModulator`; `tc/demodulator.go:17-83` `LTCDemodulator`.

**Inputs → outputs:** timecode → 10-byte frame → float32 samples (one frame per `Encode`); samples → bits → timecode.

## Completeness review

- **Complete: no.** `DecodeBiPhase` assumes the frame starts at sample 0 and uses the nominal bit period (`ltc.go:72-91`). There is no sync-word search, no clock recovery (despite the comment at `:66`) and no speed tolerance (real LTC runs at ±10%+ speed). Zero-crossing detection without hysteresis (`:126-147`) means noise adds spurious transitions.
- **Tested:** the in-repo package **does not build its tests**. `demodulator_test.go:19-24,67-68`, `e2e_test.go:62-67,157-158` and `modulator_test.go:17` call the old `mpsk.NewModulator(float32…)` and a removed `tc.NewModulator`/`NewDemodulator` (a PSK-based timecode path that no longer exists). A copy without those three files passes 18 tests: BCD, marshal/unmarshal, validate, parse, add frames, drop-frame counts, `FromTime`/`ToTime`, `EncodeBiPhase_DecodeBiPhase` clean roundtrip, sample count, and invalid inputs (`GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go test -count=1 -v ./tc/` in ``, 2026-11-09). The audio subagent's offline CLI run `gen_tc`→`recv_tc` decoded 01:02:03:04 ([[audio-modem-cli]]).
- **Correct: unknown** under noise, offset or speed change (untested); clean roundtrip yes.

## TODOs

- [ ] Port or delete the stale tests (old mpsk API, removed tc.NewModulator) so the package builds — tc/demodulator_test.go:19-24, e2e_test.go:62-67, modulator_test.go:17 #p1 #ER-022
- [ ] Sync-word search and bit-clock tracking (PLL on transitions) in DecodeBiPhase — tc/ltc.go:67-123 #p2
- [ ] Hysteresis on zero crossings, plus a noise test — tc/ltc.go:126-147 #p2

## Links

- Module: [[modulation]] · [[audio-modem-cli]] · [[W7-filters-codecs-voice]]
