---
type: algorithm
status: tested
tags: [easyrobot, motion, gait, legged, timing]
workstream: W3
module: "[[motion-and-gait]]"
source: [x/math/control/motion/gait/scheduler/scheduler.go, x/math/control/motion/gait/types/contracts.go]
tests: [x/math/control/motion/gait/scheduler/scheduler_test.go]
test_result: pass
complete: yes
correct: yes
summary: "Per-leg stance/swing phase from a periodic pattern (offset, duty); tripod, ripple, wave, walk, trot presets; allocation-free."
evidence: "GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -race -count=1 ./x/math/control/motion/gait/... → ok (2026-11-09); 11 scheduler tests failed first on a no-op skeleton [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# Gait phase scheduler (patterns, duty factor, phase offsets)

**What it does:** one global gait phase φ ∈ [0,1) advances by dt/T. Each leg i is in stance when `(φ − Offset_i) mod 1 < Duty_i` (Offset = touchdown phase, Duty = β). Stance progress runs 0 at touchdown → 1 at lift-off; swing progress 0 at lift-off → 1 at touchdown. This is the timing that makes a hexapod walk (tripod / ripple / wave) or a quadruped walk/trot ([[2026-11-09-gaits-w3]] §1).

**Where:** `x/math/control/motion/gait/scheduler/scheduler.go`.
- `Pattern{Name, Legs, Offset, Duty [MaxLegs]float32}` + `Validate()`; presets `Tripod()` (β ½, offsets {0, ½}), `Ripple()` (β ⅔), `Wave()` (β ⅚), `Walk()` (β ¾, order LH LF RH RF), `Trot()` (β ½, diagonal pairs).
- Leg order: left side front → back, then right side front → back (`HexL1..HexR3` = 0..5, `QuadLF, QuadLH, QuadRF, QuadRH`), `MaxLegs = 8`.
- `New(p, period)`, `NewFromTemplate(gaittypes.GaitTemplate, legs)`, `Advance(dt)` (dt ≤ 0, NaN or Inf ignored; dt ≥ 2²⁴·period is ignored too, `TestAdvanceIgnoresHugeDt`), `State()`, `Leg(i)`, `StanceTime(i)`, `SwingTime(i)`, `SetPeriod`, `Reset(phase)`.
- `LegState.PhaseState(liftFrac, touchdownFrac)` maps a leg to Support / Lift / Swing / Touchdown for [[gait-support-endpoint-planner]].

**Numerics:** the phase is stored as an integer fraction of the cycle (2⁻²⁰ steps), so wrap-around is exact and offsets of ⅙ and ⅓ do not overlap through float32 rounding. Consequence: ripple can show 5 stance legs for one step at a hand-off, and wave's β is 0.8333340.

## Completeness review

- Complete: yes for periodic patterns. Gait **transitions** (the `GaitScheduler` interface `Active`/`SetTarget`/`Tick` in `gait/types/contracts.go`) are still declared only.
- Tested: 4096-point phase grid, every hand-off ±1 float32 step, 20 000 `Advance` ticks per preset: tripod exactly 3 stance legs with alternating groups; wave ≥ 5 stance, lift-off order L3 L2 L1 R3 R2 R1; ripple ≥ 4 stance and ≤ 1 swing leg per side; trot diagonal pairs identical; walk exactly one swing leg; 10 wraps and 10 lift-offs per leg in 10 cycles; `AllocsPerRun` = 0 for every preset; `TestOffsetRoundsToGrid` pins offset rounding at the 2⁻²⁰ grid (truncation and ceiling mutants killed). Review dt sweep (1e-6 … 7.25 periods, 200 k steps) kept the stance counts and progress in [0,1).
- Correct: yes on the tests above.

## TODOs

- [x] Implement a phase scheduler with presets: hexapod tripod (β=0.5), ripple (β≈0.67), wave (β=5/6); quadruped walk (β=0.75), trot (β=0.5) (ER-018) — gait/scheduler/scheduler.go #p2
- [x] Map duty factor + swing fractions to Lift/Swing/Touchdown/Support progress — gait/scheduler/scheduler.go `LegState.PhaseState` #p2
- [x] Tests: number of stance legs at every t, phase offsets, contact sequence — gait/scheduler/scheduler_test.go #p2
- [ ] Gait transitions (implement `GaitScheduler.SetTarget`/`Tick` with a transition policy, continuity test) — gait/types/contracts.go:191 #p2 #backlog

## Links

- Module: [[motion-and-gait]] · [[gait-support-endpoint-planner]] · [[gait-foothold-planner]] · [[W3-legged-ik]] · [[W6-decision-trees]]
