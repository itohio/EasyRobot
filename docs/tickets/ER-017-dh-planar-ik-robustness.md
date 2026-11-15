---
id: ER-017
title: "DH and planar IK robustness"
type: ticket
status: done
kind: implementation
owner: easyrobot
workstream: W3
milestone: M-a
tags: [easyrobot, ticket, w3, kinematics]
created: 2026-11-09
updated: 2026-11-09
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# ER-017 — DH and planar IK robustness

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

Make leg and chain IK either correct or explicitly failing. Accepted recommendation: analytic 3-DOF leg IK plus a damped, step-limited, limit-aware solver for longer chains. Defects from the IK review:

- **n>3 chains fail** (E1): J is 3×n and `Gesvd` requires rows ≥ cols — `x/math/control/kinematics/joints/dh/denavithartenberg.go:216-217`, `x/math/primitive/fp32/la.go:760`. `Matrix.DampedLeastSquares` exists unused at `x/math/mat/pseudo_inverse.go:48`.
- **Undamped, no step limit, absolute σ cutoff 1e-10 in float32** (E3): cold-seed convergence 37–81 % on legs — `denavithartenberg.go:182-229`, `la.go:712`.
- **Silent clamp, no best-effort result or reason** (E4) — `denavithartenberg.go:126`, `:110-112`.
- **Zero-value `Config` pins joints at 0** (E5) — `dh/config.go:37-46`, `denavithartenberg.go:35`.
- **Analytic leg IK wrong with `err = nil` behind the yaw axis** (E8) and `a1` not wrapped before `Limit`, silent clamps (E9) — `joints/planar/planar3dof.go:122`, `:131`, `:139`, `:142-144`.

Algorithm notes: [[dh-jacobian-pseudoinverse-ik]], [[planar-3dof-leg-kinematics]]; module [[kinematics]]; workstream [[W3-legged-ik]].

## Dependencies and scope

Dependencies: None blocking (ER-001 done). Runs before or in parallel with [[ER-018-leg-helpers-gait-scheduler|ER-018]]; ER-018's `Chain.Model` uses this ticket's solvers. The quaternion fix is [[ER-021-mat-vec-quaternion-qr-inverse|ER-021]], not here.

Not in scope: No orientation targets (E6), no allocation work beyond preallocated solver buffers (E7 stays p3), no manipulator firmware ([[ER-039-manipulator-integration|ER-039]]).

Source/test edits are approved by the standing autonomy instruction; scheduling is decided separately. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [x] **DH IK for n>3** — `TestBackward_RedundantChains` (4-link planar, 6-DOF): `ErrUnsupportedOperation` before, 300/300 each after; `TestBackward_AlphaAndLengthJoints` 0/200 → 200/200. Fix: damped least squares with a 3×3 Cholesky solve (`mat` untouched).
- [x] **Robust DH IK** — `TestBackward_HexlegRealisticWorkspace` nominal seed 401/500 → 500/500 (p90 8 iterations); `TestBackward_TypedFailures` F1/E2 → `ErrJointLimit`, E1 → `ErrUnreachable`, best-effort q in limits; `TestBackward_NoAllocations` 156 → 0; review round: `TestBackward_FullLimitsNoFalseJointLimit` 1543/2000 → 2000/2000, `TestBackward_PrismaticUnitInvariance` (gantry mm 0/300 → 300/300), `TestBackward_NonFiniteInput` (new `ErrNonFinite`).
- [x] **Analytic leg IK errors** — `TestPlanar3DOF_ProbeRWorkspace` 422/500 → 500/500 exact; `TestPlanar3DOF_FullRangeExact` 1486/2000 → 2000/2000; `TestPlanar3DOF_TypedErrors`, `TestPlanar3DOF_NonFiniteTarget`. Fix: four candidates (coxa forward/reversed × knee down/up), angle wrap, typed errors.
- [x] **Config validation** — `TestConfig_Validate`, `TestNew_RejectsInvalidConfig` (dh and planar): compile failure before; `dh.New`, `planar.New2DOF/New3DOF` now return `(…, error)`.
- [x] Each test recorded failing before its fix; `go test -race -count=1 ./x/math/control/kinematics/...` passes; algorithm notes updated ([[dh-jacobian-pseudoinverse-ik]], [[planar-3dof-leg-kinematics]], [[planar-2dof-arm-kinematics]], [[dh-forward-kinematics]]).

## Effort uncertainty and revision trigger

Size guess: part of M-a stage 1 (**M–L** together with ER-018); this ticket alone ≈ **M**. Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

Report: ER-017-018. The lead's probe (30/60/90 mm, ±π limits, seed 0.1) went from 29/500 to 500/500.
