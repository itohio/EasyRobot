---
type: algorithm
status: tested
tags: [easyrobot, motion, trajectory, servo]
workstream: W3
module: "[[motion-and-gait]]"
summary: "Signed, near-time-optimal jerk-limited 1-D profile; lands exactly on target within v/a/j limits (ER-018)."
source: [x/math/control/motion/vaj1d.go]
tests: [x/math/control/motion/vaj1d_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race -count=1 ./x/math/control/... → ok (2026-11-09). Before: probe J (0→100 ends 99.319, 0→0.5 never moves, |a| 4403 vs 100); review round 1: first-order lag and float32 stalls → TestVAJ1D_StepProfile; round 2: chatter and NaN dt → TestVAJ1D_ShortMoveSweep, TestVAJ1D_InvalidSamplePeriodIsNoOp; round 3: large-J teleport and long-brake overshoot → TestVAJ1D_SettleRespectsLimits, LongBrakeNoOvershoot, RandomRestToRest; review of ER-017/ER-018: jittered dt reversals 26/27/18/21 of 200 → TestVAJ1D_JitteredSamplePeriod 0/200, NaN target runaway → NonFiniteTargetBrakes, invalid limits → InvalidLimits; all fail first, pass now; 20/20 mutants killed [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# VAJ1D jerk-limited 1-D profile

**What it does:** streams a position `Output` toward `Target` per `Update(dt)` with velocity, acceleration and jerk limits (S-curve). State is signed, so target reversals decelerate through zero instead of snapping direction.

**Method (since ER-018):** each tick bisects the jerk (32 steps) for the largest value that keeps four conditions: (1) this tick does not pass the target; (2) the time-optimal brake that follows (final ramp planned at 0.9·J) does not pass it either, including where velocity crosses zero and the velocity peak inside the tick; (3) ramping acceleration back to zero (at least one tick) keeps |v| ≤ maxV; (4) acceleration is held at ±maxA once reached, even mid-tick. The per-tick controller stays 3·J·dt³ inside the brake curve (capped at the slack and at e/4) so a landing always exists. **Terminal landing:** once the brake would finish within 6 ticks, `planLanding` searches an exact landing over n whole ticks (n from max(3, ⌈tStop/dt⌉), up to 3 more); `solveLanding` starts from the minimum-norm jerk sequence and alternates projections onto |j| ≤ J and the three end conditions (position, velocity, acceleration = 0), ≤ 64 rounds. A plan is accepted only if it stays within limits (incl. the velocity extremum inside each tick), never passes the target and never reverses velocity; the last tick lands exactly. **Settle:** only when the remaining distance is within one tick's reach under all three limits, min(J·dt³/6, A·dt²/2, V·dt) plus a rounding floor, and the settle step ≤ V·dt. A Target or dt change, or an external write to `Input`, `Velocity` or `Acceleration`, discards the plan. Internal state is float64 (see below). No allocation (fixed arrays).

**Where:** `x/math/control/motion/vaj1d.go` (`New1D`, `Reset`, `Update`). Package name `vaj` in dir `motion`.

**Inputs → outputs:** limits (maxV, maxA, maxJ), `Input/Target`, `dt` s → `Output`, `Velocity`, `Acceleration`. **Contract:** dt ≤ 0, NaN or +Inf is a no-op; a zero, negative or NaN limit makes `Update` a no-op (Output = Input); +Inf means unlimited for any of the three (J = Inf reaches A in one tick); a non-finite `Target` is replaced by the brake-to-rest point (the profile brakes and holds); no deadband — callers that want one add it; the landing plan is a continuous-time jerk profile on the tick grid it was solved on, so later ticks with a different (measured) dt advance along it instead of dropping it; an external write to `Input`, `Velocity`, `Acceleration` or `Target` drops the plan. Used by the manipulator firmware (`drivers/manipulator/main.go`, degrees, `New1D(100,100,100)`, measured dt ≈ 100 ms — its moves are now close to time-optimal, so noticeably faster than before) and by [[rigidbody-path-follower]].

## Completeness review

- Complete: yes.
- Tested: `TestVAJ1D_ReachesTarget`, `TestVAJ1D_ReversalStopsWithinStoppingDistance`, `TestVAJ1D_StepProfile` (12 cases), `TestVAJ1D_ShortMoveSweep` (200 sizes, two limit sets: zero reversals), `TestVAJ1D_SettleRespectsLimits`, `TestVAJ1D_LongBrakeNoOvershoot`, `TestVAJ1D_RandomRestToRest` (400 seeded moves), `TestVAJ1D_JitteredSamplePeriod` (200 moves × dt jitter 1e-4, 1e-3, 2 %, 10 %: exact landing, 0 reversals, per-tick deviation ≤ 1.05·J·dt³/6), `TestVAJ1D_NonFiniteTargetBrakes`, `TestVAJ1D_InvalidLimits` (13 hold cases, 7 unlimited combinations), contract tests `TargetChangeMidLanding`, `ResetMidMove`, `ExternalStateWrites`, `InputWriteMidLanding`, `TestVAJ1D_InvalidSamplePeriodIsNoOp`, `TestVAJ1D_UpdateDoesNotAllocate` (gc; `project` is a plain function, `-gcflags=-m` shows no escape — TinyGo heap use not re-measured, TinyGo is not authorised for this peer). Cruise ticks short-circuit (3 feasibility checks instead of 34).
- Correct: yes. Worst settle/time-optimal ratio in the sweep 1.16; 100 units at 10 ms within 1 % at 1.92 s, settled 2.11 s (optimum 2.10 s). At the firmware's 100 ms tick, whole-tick landings are coarse: 30° within 1 % at 2.10 s, settled 2.60 s. A reversal can follow an external `Input`/`Target` change mid-landing (review probe: 232/3000 perturbed runs); none without one.
- **Embedded cost (decision needed):** since review round 3 the internal state (position with compensated summation, velocity, acceleration) and all per-tick maths are **float64**; the exported fields stay float32. Host ≈ 2.7 µs per `Update`. The manipulator node is a SAMD21 (Cortex-M0+, no FPU) and the ESP32 has a single-precision FPU only, so float64 is software-emulated there — not measured. This deviates from the repo's float32 convention; it was the fix for float32 rounding bias on brakes spanning ≥ 1e4 ticks.

## TODOs

- [x] Fix sign test `x1 < 1` → `x1 < 0`; failing test first: 0→0.5 and 0→100 must end within 1e-3 (ER-018) — vaj1d.go rewritten #p1 #ER-018
- [x] Enforce |a| ≤ maxA and bounded jerk (no velocity snap); test with finite differences — vaj1d.go #p1 #ER-018
- [x] Signed velocity/acceleration so target reversals decelerate — vaj1d.go #p2
- [ ] Rename package dir/name mismatch (`motion` dir, `vaj` package) — vaj1d.go:3 #p3

## Links

- Module: [[motion-and-gait]] · [[rigidbody-path-follower]] · [[W3-legged-ik]]
