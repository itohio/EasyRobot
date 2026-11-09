---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, synchronisation]
workstream: W7
module: "[[modulation]]"
summary: "probe: 0 outputs"
source: [x/math/dsp/modulation/internal/timing_recovery.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests in repo; probe on a copy (tmp/easyrobot/audio/internalcopy/internal/probe_test.go, 2026-11-09): 1600 input samples (200 symbols, SPS 8) → 0 outputs"
---
# Gardner symbol timing recovery (internal, unused)

**What it does (intended):** resamples complex baseband to symbol strobes using a Gardner timing-error detector, Re[(x(n)−x(n−2))·x*(n−1)], a PI loop and linear interpolation.

**Where:** `internal/timing_recovery.go:43-78` constructor (gains from loop bandwidth/damping), `:83-200` `Process`, `:203-226` `interpolate`. **No demodulator imports it**: the only `modulation/internal` import is `internal.PowerTracker` in mask (`mask/demodulator.go:12,68`).

**Inputs → outputs:** `complex64` stream → (intended) one sample per symbol.

## Completeness review

- **Complete: no / Correct: no.**
  - `Process` reads **one batch** (default 16 samples, `:44,98,104`) per outer iteration, then exits if fewer than 2.5·SPS samples are buffered (`:102-124`). For SPS ≥ 7 it never produces output. Probe: SPS 8, 200 symbols → **0 outputs**.
  - If `dst` is also a `buffer.Reader` (as `*buffer.Values` is), it reads from **dst** and ignores `src` (`:88-92`). This is the in-place convention of `internal/helpers.go`, but surprising for a rate-changing block.
  - The TED takes three consecutive *outputs* as (n−2, n−1, n). Gardner needs strobe/mid-strobe pairs at 2 samples per symbol. The phase advance per output is `kp·e + freq` rather than one symbol period, and the buffer advances by at most one input sample per output (`:131-185`). So it isn't a symbol-rate resampler, and `estimate()` (one output per symbol, `:228-235`) doesn't match.
  - Allocates `readBuf` and `output` on every call (`:98,129`).
- **Tested:** none in repo.

## TODOs

- [ ] Rewrite as a proper interpolating timing loop (Farrow or polyphase, Gardner at 2 sps or Mueller–Müller at 1 sps), with tests for fractional offset and ±ppm drift; wire it into mpsk/qam — internal/timing_recovery.go:83-200 #p1 #ER-022
- [ ] Fix the single-batch read that yields zero output — internal/timing_recovery.go:102-124 #p1 #ER-022

## Links

- Module: [[modulation]] · needed by [[modem-energy-detector-and-preamble-acquisition]], [[mpsk-modem]], [[qam-modem]] · [[W7-filters-codecs-voice]]
