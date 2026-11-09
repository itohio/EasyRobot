---
type: algorithm
status: stub
tags: [easyrobot, motion, gait, legged]
workstream: W3
module: "[[motion-and-gait]]"
source: [x/math/control/motion/gait/rigidbody/planner.go]
tests: []
test_result: none
complete: no
correct: unknown
evidence: "read only (2026-11-09): Advance returns ErrNotImplemented without integrating pose; EstimateFromEndpoints returns ErrNotImplemented"
---
# Gait rigid-body planner (body pose from twist / from feet)

**What it does (intended, `gait/DESIGN.md`):** integrate body twist into pose and distribute body motion to leg endpoints; estimate body pose/twist from stance-foot positions; keep CoM inside the support polygon.

**Where:** `gait/rigidbody/planner.go:60-72` (`Advance`), `:75-77` (`EstimateFromEndpoints`), `:81-91` (`integratePose`, unused; rotation `TODO` at `:89`).

**Inputs → outputs:** `RigidBodyCommand{LinearVelocity, AngularVelocity}`, `dt` → `RigidBodyState`; feet map → state.

## Completeness review

- Complete: no — both methods return `ErrNotImplemented`; `Advance` only stores the twist and advances time; translation integration helper exists but is not called; no rotation integration, no support polygon.
- Tested: none.
- Correct: n/a.

## TODOs

- [ ] Implement `Advance` (position + quaternion integration of body twist) with tests — gait/rigidbody/planner.go:60-71,89 #p2
- [ ] Implement `EstimateFromEndpoints` (rigid fit of stance feet, e.g. Kabsch/SVD) — planner.go:75 #p2
- [ ] Support-polygon / stability margin helper used by body planner — gait/rigidbody (new) #p2

## Links

- Module: [[motion-and-gait]] · [[gait-support-endpoint-planner]] · [[gait-phase-scheduler]] · [[W3-legged-ik]]
