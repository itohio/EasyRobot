---
type: algorithm
status: partial
tags: [easyrobot, kinematics, planar, ik, legged]
workstream: W3
module: "[[kinematics]]"
summary: "partial (silent wrong answers behind the yaw axis, out of reach, at limits)"
source: [x/math/control/kinematics/joints/planar/planar3dof.go, x/math/control/kinematics/joints/planar/config.go]
tests: [x/math/control/kinematics/joints/planar/planar3dof_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test planar → ok, 9 tests/18 subtests (2026-11-09); probe A/I/X/Y/R in [[ER-R2-ik-review]]: FK exact; IK exact when the foot is in front of the coxa axis, silently wrong (err=nil) behind it, for unreachable targets and on limit clamps"
---
# Planar 3-DOF leg kinematics (coxa–femur–tibia)

**What it does:** Closed-form FK/IK for a yaw joint followed by two pitch joints in a vertical plane — exactly a hexapod/quadruped leg.

**Where:** `planar/planar3dof.go:53-94` (`Forward`), `:96-152` (`Backward`); `planar/config.go` (`Config{Min, Max, Length}`, `Limit`).

**Inputs → outputs:** joints `[a0 a1 a2]` radians (a0 coxa yaw about +z, a1 femur pitch, a2 tibia **relative** to femur; positive = up) → `[x y z roll pitch yaw]` with `x' = l0 + l1·cos a1 + l2·cos(a1+a2)`, `z = l1·sin a1 + l2·sin(a1+a2)`, `x = x'·cos a0`, `y = x'·sin a0`; `pitch = a1+a2` (sign opposite to the right-hand rule about y), `yaw = a0`, `roll = 0`. IK: `a0 = atan2(y,x)`, `x' = √(x²+y²) − l0`, law of cosines, knee branch `a2 = β − π ∈ [−π,0]`, `a1 = atan2(z,x') + α`. Equivalent to a DH chain with α₁=+π/2, a=(l0,l1,l2) (probe A: FK agree to 3e-8 m).

## Completeness review

- Complete: no — no reachability or limit reporting, one knee branch only, no zero offsets/sign/servo mapping, no angle wrapping.
- Tested: `TestPlanar3DOF_Forward` (zero and 90° base), `TestPlanar3DOF_Backward` (targets (3,0,0),(0,3,0), unit links), `TestPlanar3DOF_RoundTrip` (3 poses). Pass (2026-11-09). Tests use only stretched or easy poses.
- Correct: FK yes (probe A: 5 hand-checked leg poses, ≤7.5e-9 m). IK **no** in general:
  - Foot in front of the yaw axis: exact (probe R: 422/422 realistic leg targets; probe Y with leg limits: 15 365/15 382 below 0.1 mm).
  - Foot behind the yaw axis (tucked under the body): `x'` uses the unsigned radius (`:122`) and `atan2` flips the coxa by π (`:142`), then the clamp silently gives a wrong pose; all 78 failures in the realistic set, worst 8.3 cm, `err=nil`.
  - `a1 = atan2 + acos` is not wrapped before `Limit` (`:143`): with ±π limits, 1 620/20 000 random poses clamp to π (probe X).
  - Unreachable targets: cosines clamped (`:131`, `:139`) → returns the stretched pose with `err=nil` (probe I: (0.5,0,0) → FK (0.184,0,0)).
  - Limit violations are clamped silently (`:142-144`).
- Units trap: `cmd/manipulator/main.go:346-348` sends limits in degrees and lengths in mm; this package reads radians, so ±90 means ±90 rad (probe I: coxa 100° not clamped).

## TODOs

- [ ] Return `ErrUnreachable` / `ErrJointLimit` instead of clamping silently — planar3dof.go:131,139,142-144 #p1 #ER-017
- [ ] Handle foot behind the yaw axis (signed reach, or pick the coxa solution inside limits) — planar3dof.go:122,142 #p1 #ER-017
- [ ] Wrap angles into [Min,Max] (±2π) before clamping — planar3dof.go:143 #p2
- [ ] Knee-branch option (knee up/down) and per-joint zero offset/sign — planar/config.go:3 #p2
- [ ] Tests: random realistic workspace round trip, unreachable target error, foot-behind-axis case (ER-017) — planar3dof_test.go #p2
- [ ] Fix units in cmd/manipulator config (radians, metres) — cmd/manipulator/main.go:346 #p1 #ER-039

## Links

- Module: [[kinematics]] · [[planar-2dof-arm-kinematics]] · [[dh-jacobian-pseudoinverse-ik]] · [[ER-R2-ik-review]] · [[W3-legged-ik]]
