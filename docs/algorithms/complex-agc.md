---
type: algorithm
status: implemented
tags: [easyrobot, dsp, modulation, agc]
workstream: W7
module: "[[modulation]]"
summary: "—"
source: [x/math/dsp/modulation/internal/agc.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "read only; no tests, no callers (2026-11-09)"
---
# Complex AGC (internal, unused)

**What it does:** per complex sample, it tracks power with a [[signal-power-tracker]] EMA, computes gain sqrt(target/power), smooths it with an EMA and clamps to **[0.1, 10]** (±20 dB), then scales I/Q.

**Where:** `internal/agc.go:26-54` constructors, `:56-88` `Process`.

## Completeness review

- Complete: yes for a basic feed-forward AGC. The ±20 dB clamp (`:75-80`) is too narrow for acoustic links (probe: the modems fail at −20 to −40 dB levels; see [[modem-energy-detector-and-preamble-acquisition]]). There is no attack/release asymmetry and no hold during silence, so it pumps noise up between packets. The modems use `x/math/filter/agc` inline instead ([[filters]]), and QAM applies it to per-symbol magnitude, which destroys amplitude information ([[qam-modem]]).
- Tested: none. Correct: unknown.

## TODOs

- [ ] Choose one AGC (this or filter/agc), add a gain hold during the noise floor and a wider range, test with ±40 dB steps — internal/agc.go #p2

## Links

- Module: [[modulation]] · [[W7-filters-codecs-voice]]
