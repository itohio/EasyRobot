---
type: algorithm
status: stub
tags: [easyrobot, dsp, modulation, ofdm]
workstream: W7
module: "[[modulation]]"
summary: "spec only"
source: [x/math/dsp/modulation/ofdm/SPEC.md, x/math/dsp/modulation/ofdm/IMPLEMENTATION_PLAN.md]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "directory contains only SPEC.md (702 lines) and IMPLEMENTATION_PLAN.md (587 lines); no .go files (2026-11-09)"
---
# OFDM modem (spec only)

**What it does:** nothing yet. `ofdm/SPEC.md` (untrusted claims) describes a complex-baseband OFDM modem: IFFT/FFT subcarriers, cyclic prefix, per-subcarrier QAM/PSK, channel equalisation.

**Where:** `x/math/dsp/modulation/ofdm/SPEC.md`, `IMPLEMENTATION_PLAN.md`. The FFTs it would need exist in [[dsp-core]] ([[fft-complex-1d]]).

## Completeness review

- Complete: no (no code). Tested: no. Correct: n/a.
- Relevance: OFDM with pilots and a cyclic prefix is the standard answer to multipath and band-limited audio channels (it is used in acoustic modems), so it is a candidate for W7. It needs the synchroniser, AGC and FEC pieces that are also missing ([[modem-energy-detector-and-preamble-acquisition]]).

## TODOs

- [ ] Decide whether OFDM is in scope for the voice/acoustic link; if yes, implement it after a shared synchroniser exists — ofdm/SPEC.md #p2

## Links

- Module: [[modulation]] · [[W7-filters-codecs-voice]]
