---
type: workstream
status: proposed
tags: [easyrobot, workstream, kinematics, ik, gait, legged]
workstream: W3
---
# W3 — Legged IK: joint configuration helpers, Jacobian check, strides, gaits, timing

**Goal:** an inverse kinematics model with helpers to configure joints and a checked Jacobian, ending in hexapod or quadruped inverse kinematics with strides, gaits and timing.

The full review, with the derivation, probe numbers, error list, helper API and tickets, was in the IK review (2026-11-09, HEAD abb5c9c).

## Answers to the owner's questions

| Question | Answer | Evidence |
|---|---|---|
| Does FK work? | **Yes**: positions correct | 5 hand-computed hexapod-leg poses to 1.5e-8 m; DH and planar FK agree to 3e-8 m |
| Is the Jacobian solved properly? | **Yes**: standard-DH geometric position Jacobian `z_i × (p − o_i)` / `z_i` | matches float64 analytic to 4.4e-7 and central FD to 5.5e-5 over 7 chains × 1000 poses |
| Does IK work? | **Partly**, see below | probes C, R, Y |

What "partly" means for IK:
- DH IK seeded with the true solution ± 0.2 rad converges 100 % in ≤ 7 iterations (hexapod leg). Seeded from the previous tick's answer, as a gait loop would do, it converges 191/500.
- From a cold start it converges 37–81 %.
- Chains with more than 3 joints always fail.
- The orientation output is inverted.
- The analytic leg IK is exact when the foot is in front of the hip yaw axis. It is silently wrong (no error) when the foot is behind that axis, out of reach, or at a joint limit.

Helpers to configure joints exist only as raw `dh.Config` / `planar.Config` structs:
- There is no validation; zero-value limits pin every joint at 0.
- There is no zero offset, sign or servo map.
- The host CLI already mixes degrees/mm with radians.

An API sketch was part of the IK review.

## What exists

| Piece | Note | Status |
|---|---|---|
| DH FK | [[dh-forward-kinematics]] | partial (quaternion inverted) |
| DH Jacobian | [[dh-geometric-jacobian]] | implemented, correct |
| DH IK (pinv Newton) | [[dh-jacobian-pseudoinverse-ik]] | partial |
| Coxa–femur–tibia analytic FK/IK | [[planar-3dof-leg-kinematics]] | partial |
| 2-DOF yaw+pitch | [[planar-2dof-arm-kinematics]] | tested |
| Jerk-limited servo profile | [[vaj1d-jerk-limited-profile]] | partial (sign bug) |
| Foot trajectory per phase | [[gait-support-endpoint-planner]] | partial (clearance bug) |
| Gait scheduler (templates, duty factor) | [[gait-phase-scheduler]] | stub |
| Body planner | [[gait-rigidbody-planner]] | stub |
| Swing curve primitives | [[interpolation]] (Bezier, Hermite) | see that module |
| PID loops | [[pid-control]] | tested |
| Firmware manipulator (3 servos, dndm) | `drivers/manipulator` (TinyGo) | IK adapters are placeholders; raw mode only |
| Modules | [[kinematics]], [[motion-and-gait]] | — |

## Gaps

Each line is a capability needed for a walking hexapod or quadruped, verified against code:
1. Joint/leg configuration, validation and servo mapping (offsets, sign, ratio, channel, units).
2. Body → hip transforms and body pose (height, roll, pitch, yaw); neutral stance; reach check.
3. Stride/foothold generation from the body twist (stance feet fixed in the world).
4. Gait patterns: hexapod tripod / ripple / wave; quadruped walk / trot. Phase offsets, duty factor, cycle time from speed.
5. Swing trajectory with correct apex; stance trajectory; mid-swing re-targeting.
6. Support polygon and static stability margin; body sway for slow gaits.
7. Servo output timing: fixed-rate synchronous update of 12–18 channels, per-joint limits from config, interpolation between ticks.
8. Gait transitions; terrain and contact feedback (a hook exists).
9. Tests and simulation of a full step cycle.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Joint/leg helper package (ER-018) — x/math/control/kinematics (new) #p2
- [ ] Foothold/stride generator and body→leg transforms (ER-018) — motion/gait (new) #p2
- [ ] Support polygon / stability margin (ER-037) — motion/gait (new) #p2
- [ ] Firmware: honour `ManipulatorConfig.Motion` limits — drivers/manipulator/main.go:351-356 #p2
- [ ] Leg, gait and full-cycle tests (none exist for gait) — motion/gait/* #p2

## Inferred objectives (proposals)

- **Proposal — milestone 1 (host only, no hardware):**
  - Do [[ER-017-dh-planar-ik-robustness|ER-017]] (analytic leg IK errors, config validation) and [[ER-018-leg-helpers-gait-scheduler|ER-018]] (VAJ1D, joint/leg helpers, scheduler, strides).
  - Build a host simulation of a hexapod tripod gait with IK-review probe-style checks: stance feet fixed in the world, every foot reachable, every joint inside its limits, static margin above zero.
- **Proposal — milestone 2 (ESP32):**
  - Allocation-free 50 Hz loop for 18 joints: scheduler → footholds → curves → analytic IK → servo map → PCA9685.
  - Measure the cost per tick on the device. Flashing needs the owner's approval of the exact scope.
- **Proposal — milestone 3:** quadruped walk, then trot (adds dynamic-stability questions); gait selection exposed to [[W6-decision-trees]].
- **Proposal:** run IK on the host or on an FPU-equipped ESP32. The XIAO SAMD21 used by `drivers/manipulator` has no FPU [INFERENCE], so it should only receive joint targets.

## Research questions for ideation

1. Phase-based scheduler vs central pattern generator (coupled oscillators) for hexapod gait timing and smooth speed changes — which is simpler on an ESP32 with float32?
2. Foothold selection: fixed neutral stance + velocity offset vs reachability-maximising placement; how to handle turning in place and crab walking with a 3-DOF leg.
3. Swing curve choice (cubic Bezier vs cycloid vs minimum-jerk) for hobby servos with ~0.1–0.2 s/60° speed: what apex and swing fraction avoid stalls?
4. Static stability margin vs energy stability margin for slow hexapod gaits; is a support-polygon check enough for a quadruped walk (β = 0.75) without IMU feedback?
5. Servo calibration workflow: how to estimate the zero offset and sign per joint from a few measured foot positions (least squares on FK), so the helper's `CheckZeroPose` becomes an auto-calibration.
6. Can gait choice and step height be learned as decision-tree policies from camera features (link to W6) without a simulator?
