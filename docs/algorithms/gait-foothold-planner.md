---
type: algorithm
status: tested
tags: [easyrobot, motion, gait, legged, footholds, ik]
workstream: W3
module: "[[motion-and-gait]]"
summary: "Body twist → footholds (Raibert, reach-clipped) → world stance/swing feet → leg IK → servo degrees per tick; allocation-free."
source: [x/math/control/motion/gait/walk/walk.go, x/math/control/motion/gait/walk/reach.go]
tests: [x/math/control/motion/gait/walk/walk_test.go, x/math/control/motion/gait/walk/walk_sim_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/control/motion/gait/walk → ok (2026-11-09): TestHexapodTripodSim max stance drift 9.5e-8 m, min reach margin 9.1 mm, 0 IK errors, 0 allocs/tick; tests failed first against stubs (walk-failing.txt); review round 2: margin −1.1…−79 mm → ≥ 7.9 mm, long dt −213.6 → 10.4 mm, swing jump 31.7 → 11.6 mm (review2-*.txt); review round 3: random commands 5.9–7.0 → 8.0 mm, stalls 70/37/42 → ≤ 5 ticks, dt 0.6 no slew −12.0 → 8.0 mm, plan sampling error 2.87 → 0.000 mm (review3-*.txt, tmp/easyrobot/er018/) [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
---
# Gait foothold planner and walk glue (twist → servo degrees)

**What it does:** one `Walker.Step(dt, cmd, servoDeg)` per control tick runs the whole legged pipeline: [[gait-phase-scheduler]] → footholds → stance/swing feet in the world → body→hip transform → analytic leg IK ([[planar-3dof-leg-kinematics]]) → ServoMap degrees ([[leg-chain-config-and-servo-map]]). This is M-a stage 1 (host simulation) from [[2026-11-09-gaits-w3]].

**Where:** `x/math/control/motion/gait/walk/walk.go` (`Twist`, `Config`, `New`, `Foothold`, `Walker.Step`), `reach.go` (reach annulus margin of a coxa–femur–tibia leg).

**Method:**
- **Footholds (§2):** leg i with neutral foot nᵢ (body frame) touches down at pᵢ* = nᵢ + ½·T_st·(v + ω×nᵢ), so its stance sweeps symmetrically through the neutral point. T_st comes from `Scheduler.StanceTime(i)`.
- **Stance:** feet stay fixed in the world while the body pose integrates the applied twist (exact arc for constant v, ω).
- **Clip:** the commanded twist is scaled uniformly for all legs (never distorted) until every leg's plan **from its real state** keeps its required margin inside its reach annulus sector: a planted foot over the rest of its stance, then the swing from its real lift-off point to the next foothold; a swinging foot over the rest of its swing (then its stance if the foothold is frozen); then the steady cycle. Plans are sampled (13 points per swing, 5 per stance segment) and refined with an 8-step golden-section search around every dip within 5 mm of the smallest; sampling error ≤ 0.2 mm (`planSamplingError`, tested). A NaN plan margin counts as infeasible. **Fallback:** standing still or the previously applied twist, whichever falls less below `Margin`; each leg's required margin is min(`Margin`, its plan under that fallback), so a choice can never lower the margin. The search keeps the previous scale if it still qualifies, otherwise scans k/16 from the top and bisects inside the best cell. If no scale on the commanded direction qualifies and stopping is not safe, the walker **slows along its previous twist**: the smallest feasible fraction k/16 of it, the full previous twist only as the last resort (`Applied()` returns it, `Scale()` returns 0). Reach takes precedence over the slew limit. Checked every sub-step.
- **Slew:** `MaxAccel`/`MaxAngAccel` limit twist changes (0 disables).
- **Swing (§3):** horizontal minimum-jerk (10τ³ − 15τ⁴ + 6τ⁵) from lift-off to the re-predicted touchdown point, delayed by `LiftFraction` at both ends; height = lift→touchdown line + H·(1 − cos 2πτ)/2, apex H at mid-swing. The body-relative foothold freezes at half the horizontal swing, so a late twist change cannot make the foot jump.
- **Robustness:** a non-finite command or dt returns `kintypes.ErrNonFinite`, and |VX|, |VY| > `MaxCommandSpeed` = 100 m/s or |W| > `MaxCommandYawRate` = 100 rad/s returns `ErrCommandTooLarge`, both before any state changes; yaw wraps fully into (−π, π]; dt is split into sub-steps of at most 25 ms and at most half the shortest stance or swing phase (IK once per `Step`); `New` rejects stance or swing phases shorter than `MinPhaseTime` = 20 ms (`ErrInvalidConfig`), so a `Step` has at most 100 sub-steps; one `Step` processes at most `MaxStepDt` = 1 s and returns `ErrDtTooLong` for a longer dt. `New` deep-copies the body (`chain.Body.Clone`); a non-coxa–femur–tibia leg wraps both `ErrInvalidConfig` and `ErrUnsupportedOperation`. Not safe for concurrent use.
- **IK and servos:** `chain.LegSolver.SolveLegs` (preallocated) then `Chain.ToServo` into the caller's `[]float32`.

**Inputs → outputs:** `Twist{VX, VY, W}` (body frame, m/s, rad/s) + dt → servo degrees (18 for a hexapod); accessors `Pose`, `Applied`, `Scale`, `State`, `Feet`, `Joints`. Host cost: 45 µs per unclipped `Step`, 75 µs steady clipped, 0.41 ms mean over random command sequences.

## Completeness review

- Complete: no. No servo-speed-limited swing time (research §3): joint speeds are reported, not limited. No static-stability margin (that is [[ER-037-body-pose-static-stability|ER-037]]), no terrain, no gait transitions.
- Tested:
  - `TestStraightWalk`, `TestTurnInPlace`, `TestCrabWalk` (plus a combined twist): stance drift ≤ 9.5e-8 m, measured from the servo degrees through FK back to the world.
  - `TestClipScalesTwist` (margin ≥ Margin − 0.5 mm, scale within 0.8–1× the steady-cycle bound); `TestClipKeepsMarginFromCurrentState` (from rest at 0.15 and 0.17 m/s, clipped VX/VY/−VX = 1, forward→backward at the clip with and without slew: min margin 7.9–8.1 mm, no errors). Every walk test asserts margin ≥ Margin − 0.5 mm and no IK, limit or servo error on every tick.
  - `TestNonFiniteInputRejected`, `TestLongTickSubsteps` (dt 0.6 and 1.3 s), `TestSwingFootNoJumpOnTwistStep` (MaxAccel 0, step at 50 instants: ≤ 1.5× the steady per-tick move), `TestFoothold`, `TestNewValidates`; `TestStepAllocs`: 0 allocations per tick for steady, changing, clipped and sub-stepped ticks.
  - `walk_robust_test.go`: `TestRandomCommandSequences` (seeded, 4 parallel subtests for hex/quad × slew/no slew, 5000 ticks each, 1500 under `-race`: margin 8.0 mm, no errors), `TestClipNeverStalls` (body stopped ≤ 5 ticks while commanded), `TestLongTickNoSlew` (dt 0.1/0.26/0.6 s), `TestHugeDtClamped`, `TestPlanSamplingError` (≤ 0.2 mm vs a 400-sample dense check).
  - `walk_review_test.go` (roster review): `TestHugeCommandRecovers` (W = 1e10 without slew, then stop: recovers; failed first with permanent `ErrNonFinite`), `TestNaNPlanMarginInfeasible`, `TestNaNPlanThroughStep`, `TestCommandBound`, `TestNewRejectsTinyPhases` (period 1e-3, 1e-20, 16.7 ms swing; failed first by accepting them), `TestStopLatency` (stop within one gait cycle; coasting ≤ 0.6 of the previous speed without slew), `TestBodyCopied`, `TestNewWrapsUnsupportedLeg`, `TestLostMarginFallback`, `TestSubStepRule`, `TestYawWrapsWhileTurning`, `TestWrapPi`. 19/19 mutants killed. `gait/walk` under `-race`: 49 s.
  - Stop latency after the slowed command reaches zero (worst ticks / coasting speed share / distance after the stop command): hex 9 / 0.85 / 7 mm; hex no slew 15 / 0.48 / 54 mm; quad 5 / 0.98 / 5 mm; quad no slew 45 / 0.14 / 24 mm; adverse quad (no slew, Margin 95 % of standing) 36 / 0.45 / 173 mm. Before (holding the full previous twist): 8, 15, 5, 12, 33 ticks at full speed, 8–267 mm.
  - `TestHexapodTripodSim`: probe-R leg (39/45/100 mm, coxa ±60°, femur −60…72°, tibia −149…−20°), tripod, 1 s period, 30 mm swing, 8 mm margin; 10 s at 0.1 m/s with ω = 0.3 rad/s, then 5 s turning in place at 0.5 rad/s, 20 ms ticks. Max stance drift 9.5e-8 m, min reach margin 9.1 mm, 0 IK errors, 0 joints outside limits, 0 servo values outside travel, swing apex 29.9 mm, scale 1.0.
  - `TestQuadrupedWalkSim` (walk preset, no stability check): drift 1.05e-7 m, no errors, min margin 8.1 mm.
- Correct: yes for kinematic consistency. **Joint speed exceeds a typical hobby servo:** the hexapod sim peaks at 8.03 / 6.62 / 7.05 rad/s (coxa/femur/tibia) against 5.24 rad/s (300°/s); the quadruped at 11.03 / 7.11 / 9.34 rad/s. Swing speed relative to the body ≈ 1.875·v_foot/((1−β)(1−2·LiftFraction)) and does not drop with a longer period; femur/tibia speed is dominated by the height bump.

## TODOs

- [ ] Limit joint speed: stretch swing time or clip the twist so every joint stays under `Joint.MaxVel` (research §3); assert it in the hexapod sim — walk.go `clip` #p1 #backlog
- [ ] MCU cost: measure `Step` on the ESP32 (host mean 0.41 ms under random commands, up to the clipped worst case) and cut the plan check if it does not fit a 20 ms tick — walk.go `clip` #p2
- [ ] Owner decision on stop semantics: slowing along the previous twist travels least (above) but the quad without slew now takes up to 45 ticks to stop instead of 12; alternatively an explicit stop path with a recovery gait — walk.go `clip` #p2
- [ ] Static stability margin in the clip (support polygon, CoM) — walk.go #p2 #ER-037
- [ ] Gait transitions without foot jumps (pattern change mid-walk) — walk.go #p3

## Links

- Module: [[motion-and-gait]] · [[gait-phase-scheduler]] · [[gait-support-endpoint-planner]] · [[planar-3dof-leg-kinematics]] · [[leg-chain-config-and-servo-map]] · [[W3-legged-ik]]
