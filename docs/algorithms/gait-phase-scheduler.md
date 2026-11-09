---
type: algorithm
status: stub
tags: [easyrobot, motion, gait, legged, timing]
workstream: W3
module: "[[motion-and-gait]]"
source: [x/math/control/motion/gait/types/contracts.go]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "grep over all Go files (2026-11-09): GaitScheduler/GaitTemplate are declared only; no implementation, no presets, no `motion/gait/presets/` dir"
---
# Gait phase scheduler (templates, duty factor, phase offsets)

**What it does (intended):** turn a gait template (cycle time, per-leg phase offset, duty factor, contact sequence) into per-leg phase mode + progress at time t, and switch gaits with a transition policy. This is the piece that makes a hexapod walk (tripod / wave / ripple) or a quadruped walk/trot.

**Where:** declarations only: `gait/types/contracts.go:155-164` (`GaitTemplate`), `:166-180` (`LegRoleBinding`, `GaitInstance`), `:182-189` (`TransitionPolicy`), `:191-196` (`GaitScheduler` interface: `Active`, `SetTarget`, `Tick`), `:33-48` (`PhaseMode`, `PhaseState`).

**Inputs → outputs (intended):** template + time → per-leg `PhaseState`.

## Completeness review

- Complete: no — no implementation of `GaitScheduler` anywhere in the repo (grep for `GaitScheduler`, `GaitTemplate`, `tripod`, `duty` in Go files: only these declarations). `DutyCycles`/`PhaseMap` maps keyed by role strings; no validation, no presets, no mapping from duty factor to the four sub-phases of [[gait-support-endpoint-planner]].
- Tested: none.
- Correct: n/a.

## TODOs

- [ ] Implement a phase scheduler with presets: hexapod tripod (β=0.5), ripple (β≈0.67), wave (β=5/6); quadruped walk (β=0.75), trot (β=0.5) (ER-018) — gait/types/contracts.go:191 #p2
- [ ] Map duty factor + swing fractions to Lift/Swing/Touchdown/Support progress — gait (new) #p2
- [ ] Tests: number of stance legs at every t, phase offsets, contact sequence, transition continuity — gait (new) #p2

## Links

- Module: [[motion-and-gait]] · [[gait-support-endpoint-planner]] · [[W3-legged-ik]] · [[W6-decision-trees]]
