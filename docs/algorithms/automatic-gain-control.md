---
type: algorithm
status: partial
tags: [easyrobot, filter, dsp, audio]
workstream: W7
module: "[[filters]]"
summary: "Instantaneous level, inverted α"
source: [x/math/filter/agc/agc.go]
tests: []
test_result: none
complete: yes
correct: no
evidence: "no tests; code review: per-sample instantaneous |x| level, alpha = timeConst*fs (inverted)"
---
# Automatic gain control

**What it does:** Scales each sample by a gain that is smoothed toward `targetLevel/|x|`, clamped to [0.1, 10].

**Where:** `x/math/filter/agc/agc.go`: `New` :16-23, `SetSampleRate` :26-32, `Process` :35-53, `Reset` :56-58.

**Inputs → outputs:** target level, "time constant", sample → scaled sample.

## Completeness review

- Complete: no — no envelope detector, attack/release, or hold.
- Tested: no test file (`[no test files]`).
- Correct: **no** (code review; not probed).
  - Level = instantaneous `|sample|` (:37). For a sinusoid the desired gain swings between target/peak and the 10× clamp within every cycle, so the AGC modulates the waveform (distortion) instead of normalising its envelope.
  - `alpha = timeConst * sampleRate` clamped to 1 (:28): a *longer* time constant gives a *larger* (faster) α — inverted; standard is `α = 1 − exp(−1/(τ·fs))`. Before `SetSampleRate`, α = timeConst (:21).
  - `math.Abs(float64(...))` (:37) — float64 round trip per sample (soft-float on Xtensa LX6/7 [INFERENCE]).

## TODOs

- [ ] Envelope follower (peak/RMS with attack/release) and α = 1−exp(−1/(τ·fs)) — agc.go:28,37 #p1 #ER-019
- [ ] Tests: sine at −20/0/+10 dB converges to target RMS without THD increase #p2
- [ ] Use math32 — agc.go:37 #p3

## Links

- Module: [[filters]]; audio pipeline: [[dsp-core]], [[audio-io]]
