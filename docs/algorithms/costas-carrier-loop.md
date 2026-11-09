---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, synchronisation]
workstream: W7
module: "[[modulation]]"
summary: "—"
source: [x/math/dsp/modulation/internal/costas.go]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "no tests, no callers (grep for modulation/internal imports finds only mask → PowerTracker), 2026-11-09"
---
# Costas / M-th-power carrier loop (internal, unused)

**What it does:** per complex sample, it derotates by the current phase estimate, computes the phase error as wrap(M·atan2(Q,I))/M (an M-th-power detector for M-PSK), and updates a PI loop (frequency integrator plus phase).

**Where:** `internal/costas.go:34-65` constructor (gains from loop bandwidth relative to **symbol rate**), `:67-97` `Process` (per sample), `:99-113` `phaseDetector`, `:128-130` `FrequencyOffset` (`frequency·symbolRate/2π`).

## Completeness review

- Complete: no. The loop runs per **sample**, but its gains and `FrequencyOffset` scaling assume per-**symbol** updates (`:34-65,128-130`), so the bandwidth is off by a factor of SPS [INFERENCE]. The M-th-power detector is wrong for QAM (the comment claims QAM support, `:99-113`). There is no lock detector. The modems use `x/math/filter/pll` inline instead (see [[filters]]).
- Tested: none. Correct: unknown.

## TODOs

- [ ] Decide between this and filter/pll; remove the duplicate. If kept, run it at symbol rate after the matched filter and test pull-in vs CFO — internal/costas.go #p2
- [ ] Decision-directed detector for QAM — internal/costas.go:99-113 #p2

## Links

- Module: [[modulation]] · [[mpsk-modem]] · [[qam-modem]] · [[W7-filters-codecs-voice]]
