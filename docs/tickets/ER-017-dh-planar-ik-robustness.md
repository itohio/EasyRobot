---
id: ER-017
title: "DH and planar IK robustness"
type: ticket
status: ready
kind: implementation
owner: easyrobot
workstream: W3
milestone: M-a
maps: [T-IK-02, T-IK-03, T-IK-05, T-IK-06]
tags: [easyrobot, ticket, w3, kinematics]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-017 — DH and planar IK robustness

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Make leg and chain IK either correct or explicitly failing. Accepted ER-R2 recommendation 1: analytic 3-DOF leg IK plus a damped, step-limited, limit-aware solver for longer chains (roster `decisions.md`, "EasyRobot ER-R2"). Defects from [[ER-R2-ik-review#4. Error list (all reproducible with the probe)|IK review §4]] (E1, E3, E4, E5, E8, E9):

- **n>3 chains fail** (E1): J is 3×n and `Gesvd` requires rows ≥ cols — `x/math/control/kinematics/joints/dh/denavithartenberg.go:216-217`, `x/math/primitive/fp32/la.go:760`. `Matrix.DampedLeastSquares` exists unused at `x/math/mat/pseudo_inverse.go:48`.
- **Undamped, no step limit, absolute σ cutoff 1e-10 in float32** (E3): cold-seed convergence 37–81 % on legs — `denavithartenberg.go:182-229`, `la.go:712`.
- **Silent clamp, no best-effort result or reason** (E4) — `denavithartenberg.go:126`, `:110-112`.
- **Zero-value `Config` pins joints at 0** (E5) — `dh/config.go:37-46`, `denavithartenberg.go:35`.
- **Analytic leg IK wrong with `err = nil` behind the yaw axis** (E8) and `a1` not wrapped before `Limit`, silent clamps (E9) — `joints/planar/planar3dof.go:122`, `:131`, `:139`, `:142-144`.

Algorithm notes: [[dh-jacobian-pseudoinverse-ik]], [[planar-3dof-leg-kinematics]]; module [[kinematics]]; workstream [[W3-legged-ik]].

## Dependencies and scope

Dependencies: None blocking (ER-001 done). Runs before or in parallel with [[ER-018-leg-helpers-gait-scheduler|ER-018]]; ER-018's `Chain.Model` uses this ticket's solvers. The quaternion fix (T-IK-04) is [[ER-021-mat-vec-quaternion-qr-inverse|ER-021]], not here.

Not in scope: No orientation targets (E6), no allocation work beyond preallocated solver buffers (E7 stays p3), no manipulator firmware ([[ER-039-manipulator-integration|ER-039]]).

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **T-IK-02** — failing test first: 4-link planar and 6-DOF reachable targets converge (probe G1/G2). Fix: `DampedLeastSquares` (`mat/pseudo_inverse.go:48`) or SVD of Jᵀ.
- [ ] **T-IK-03** — failing test first: hexleg realistic workspace, nominal seed ≥ 95 % within 100 iterations; F1 returns `ErrJointLimit` plus best-effort q. Fix: DLS with adaptive λ, step cap, limit-aware (clamp-and-zero-column), relative σ cutoff; preallocated buffers.
- [ ] **T-IK-05** — failing test first: unreachable → `ErrUnreachable`; foot behind the yaw axis solved or `ErrJointLimit`; probe R set 500/500 exact. Fix: signed reach, angle wrap, typed errors (`planar3dof.go:122-144`).
- [ ] **T-IK-06** — failing test first: `dh.New` with a zero-value `Config` returns an error (or `Validate()` does). Fix: `New` returns `(…, error)` or a separate `Validate`.
- [ ] Each test above recorded failing before its fix; `GOTOOLCHAIN=local GOPROXY=off go test -count=1 ./x/math/control/kinematics/...` passes afterwards; algorithm notes updated per [[conventions#After a fix|After a fix]].

## Effort uncertainty and revision trigger

Product's guess: part of M-a stage 1 (**M–L** together with ER-018); this ticket alone ≈ **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
