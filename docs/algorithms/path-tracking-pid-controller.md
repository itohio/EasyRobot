---
type: algorithm
status: partial
tags: [easyrobot, motion, control, pid]
workstream: [W3, W6]
module: "[[motion-and-gait]]"
summary: "partial (lateral sign, heading wrap)"
source: [x/math/control/motion/rigidbody/motion.go]
tests: [x/math/control/motion/rigidbody/motion_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test motion/rigidbody → ok (2026-11-09) (TestBackwardControlClamping checks clamps only); probe M: lateral term steers away from the path, heading error not wrapped at ±π"
---
# Path-tracking PID controller (Motion.Backward)

**What it does:** Three [[pid-controller]] loops (speed, heading, lateral offset) add increments to the previous `[linear, angular]` command, then clamp acceleration, turn rate, turn acceleration and lateral acceleration (`|v·ω| ≤ a_lat`).

**Where:** `motion/rigidbody/motion.go:303-345` (`Backward`), `:347-393` (`planBackward`), `:504-510` (`lateralError`), `:512-523` (`limitForTorque`).

**Inputs → outputs:** current and desired state 6×1, controls ≥2×1 → updated controls (effort rows passed through).

## Completeness review

- Complete: yes for its scope.
- Tested: `TestBackwardControlClamping` (limits only, no sign/direction assertions). Pass (2026-11-09).
- Correct: **no**.
  - Lateral sign: `lateralError` = (desired − actual)·left-normal; PID target 0 → output `−P·err` (`:368-371`). Probe M: robot right of the path (path on the left) → angular −0.0628 (turns right, away); robot left → +0.0628. (Magnitude saturated by turn-acceleration clamp.)
  - Heading: `headingPID.Target − Input` is not wrapped (`:364-366`); yaw 3.10 → −3.10 (true error +0.083 rad) commands negative ω.
  - Commands integrate PID outputs (`controls.Linear + speedPID.Output`, `:373`, `:377`), so the P term acts as an I term on the command.

## TODOs

- [ ] Fix lateral-error sign; failing test: path on the left ⇒ ω > 0 (ER-038) — motion.go:368-371 #p1 #ER-038
- [ ] Wrap heading error into (−π, π]; test across the ±π seam — motion.go:364-366 #p1 #ER-038
- [ ] Decide absolute vs incremental command semantics and document — motion.go:373,377 #p3

## Links

- Module: [[motion-and-gait]] · [[pid-controller]] · [[rigidbody-path-follower]]
