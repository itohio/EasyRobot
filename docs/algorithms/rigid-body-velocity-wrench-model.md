---
type: algorithm
status: tested
tags: [easyrobot, kinematics, rigidbody]
workstream: W3
module: "[[kinematics]]"
summary: "Quasi-static proportional map from body velocity (or velocity error) to a clamped force/torque: `F…"
source: [x/math/control/kinematics/rigidbody/model.go, x/math/control/kinematics/rigidbody/tensor_helpers.go]
tests: [x/math/control/kinematics/rigidbody/model_test.go, x/math/control/kinematics/rigidbody/tensor_helpers_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test kinematics/rigidbody → ok, 3 tests (2026-11-09); also driven by motion/rigidbody TestTriangularChassisFollowsSinusoid (pass)"
---
# Rigid-body velocity → wrench model

**What it does:** Quasi-static proportional map from body velocity (or velocity error) to a clamped force/torque: `F = clamp(m·k_lin·v)`, `τ = clamp(k_ang ⊙ (I·ω))`. Plus inertia helpers (sphere, cylinder, box, `R·I·Rᵀ`).

**Where:** `rigidbody/model.go:113-123` (`Forward`), `:125-136` (`Backward`, error-driven), `:93` (`AngularAcceleration = I⁻¹τ`); `tensor_helpers.go:7-30`.

**Inputs → outputs:** 6×1 `[vx vy vz wx wy wz]` → 6×1 wrench.

## Completeness review

- Complete: yes for its stated toy scope (`rigidbody/DESIGN.md`: gains, no drag/gravity).
- Tested: `TestForwardProducesWrench`, `TestBackwardUsesError` (proportional values), `TestInertiaHelpersProducePositiveDiagonal`. Pass (2026-11-09).
- Correct: yes for the formulas; inertia helpers match textbook diagonals. "Forward" maps velocity to force (a P-controller, not kinematics) — naming is misleading. Uses `Flat()` index access, so non-contiguous matrices would misread.

## TODOs

- [ ] Rename/document as a velocity P-controller; keep "kinematics" interface only if planners need it — rigidbody/model.go:112 #p3

## Links

- Module: [[kinematics]] · [[thruster-wrench-allocation]] · [[rigidbody-path-follower]] · uses `Matrix3x3.Inverse` for I⁻¹ (closed form; correct per [[matrix-inverse-and-lu]], but absolute det tolerance — see TODO)
