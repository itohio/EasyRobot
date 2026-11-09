---
type: algorithm
status: stub
tags: [easyrobot, dsp, modulation, channel]
workstream: W7
module: "[[modulation]]"
summary: "spec only"
source: [x/math/dsp/modulation/measure/SPEC.md, x/math/dsp/modulation/BLIND_CHANNEL_ESTIMATION.md]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "measure/ holds only SPEC.md (1141 lines); BLIND_CHANNEL_ESTIMATION.md (902 lines) has no code; grep finds no importer (2026-11-09)"
---
# Channel sounding (sync tone + log chirp) and blind channel estimation (specs only)

**What it does:** nothing yet. Two design documents (untrusted claims):
- `measure/SPEC.md`: a channel measurement system with a sync tone, a delay, then a constant-energy logarithmic chirp, from which it derives the channel impulse response, SNR, THD and frequency response, to drive adaptive scheme selection.
- `BLIND_CHANNEL_ESTIMATION.md`: estimating channel state from data symbols during transmission (decision-directed) for equalisation and adaptation.

**Where:** `x/math/dsp/modulation/measure/SPEC.md`, `x/math/dsp/modulation/BLIND_CHANNEL_ESTIMATION.md`. Related code: chirp generators and the chirp `Correlate` in [[dsp-core]] ([[chirp-generators]]: correlation is not a matched filter per that note); `cmd/audio gen_chirp`/`recv_chirp` ([[chirp-sync-receiver]]).

## Completeness review

- Complete: no (no package code). It would be the measurement half of the "noise figures" the owner asks about, and the input to [[adaptive-modem-tier-selector]].

## TODOs

- [ ] Implement channel sounding on the fixed chirp correlator (impulse response, frequency response, SNR per band); use it to characterise real voice/acoustic paths before choosing a modem — measure/SPEC.md #p2

## Links

- Module: [[modulation]] · [[W7-filters-codecs-voice]]
