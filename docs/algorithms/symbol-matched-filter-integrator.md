---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation]
workstream: W7
module: "[[modulation]]"
summary: "qam integrate test ok"
source: [x/math/dsp/modulation/internal/matched_filter.go, x/math/dsp/modulation/mpsk/demodulator.go, x/math/dsp/modulation/qam/demodulator.go]
tests: [x/math/dsp/modulation/qam/roundtrip_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "inline variants exercised by passing tests TestIntegrateSymbolMatchesPoint (qam) and TestPhaseDetection_* (mpsk), 2026-11-09; internal version untested and unused"
---
# Symbol matched-filter integrator (integrate-and-dump with envelope)

**What it does:** correlates each symbol window with the transmit envelope and normalises by the envelope energy: (ΣI·e, ΣQ·e)/Σe². That gives one complex point per symbol.

**Where:** inline in `mpsk/demodulator.go:582-602` and `qam/demodulator.go:518-533` (used); a stream version in `internal/matched_filter.go:30-197` (unused, untested).

## Completeness review

- Complete: no. Integrate-and-dump over a **one-symbol** envelope is an optimal matched filter only when symbols don't overlap and the window is aligned. There is no timing search, and no root-raised-cosine (RRC) pair: TX and RX use the same truncated one-symbol window, so the occupied bandwidth is far wider than `RequiredBandwidth()` = Rs·(1+β) claims (`mpsk/config.go:244-246`) [INFERENCE from waveform construction, `mpsk/waveform.go:50-65`]. The internal version reads from `dst` if it is a Reader (`matched_filter.go:76-79`) and allocates per call (`:86,128`).
- Tested: `qam.TestIntegrateSymbolMatchesPoint` checks that the inline integrator returns the constellation point on a clean, aligned symbol (pass).
- Correct: yes for aligned clean symbols; unknown otherwise.

## TODOs

- [ ] Real pulse shaping (RRC over several symbols) with matching TX/RX filters, so bandwidth claims hold and the link fits a 300–3400 Hz voice band — mpsk/waveform.go:50-65, qam/waveform.go:60-75 #p2
- [ ] Delete internal/matched_filter.go or make the demodulators use it — internal/matched_filter.go #p3

## Links

- Module: [[modulation]] · [[mpsk-modem]] · [[qam-modem]] · pulse generators in [[pulse-shape-generators]] · [[W7-filters-codecs-voice]]
