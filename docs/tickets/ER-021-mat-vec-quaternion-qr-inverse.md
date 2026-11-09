---
id: ER-021
title: "mat/vec: quaternion from matrix, QR and inverse tolerance"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: [W3, W9]
milestone: M-a
maps: [T-IK-04]
tags: [easyrobot, ticket, w3, w9, mat-vec]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-021 — mat/vec: quaternion from matrix, QR and inverse tolerance

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Remaining `x/math/mat` defects after O2 moved the homogeneous helpers and the 4×4 inverse pivoting into [[ER-018-leg-helpers-gait-scheduler|ER-018]] ([[mat-vec]]):

- **Quaternion from matrix is the inverse rotation** (E2), layout `[x y z w]` vs `[w x y z]` elsewhere — `x/math/mat/mat.go:626-648`, used at `denavithartenberg.go:139`; dense `Matrix.Orientation` not Hamilton — `mat.go:261-291`; `Matrix3x3/2x2.Quaternion` missing — `mat3x3.go:334`. Accepted decision 3: `[x y z w]` in Go, converted at the proto boundary.
- **QR**: tau/C handling and R extraction — `x/math/mat/qr.go:57`, `:109` ([[mat-qr]]).
- **Inverse tolerance**: absolute det tolerance 1e-6 rejects small SI inertia (E23; 1 kg/0.2 m box, det 3.0e-7) — `x/math/mat/inverse.go:14`, `:78`; `kinematics/rigidbody/model.go:47`; `thrusters/helpers.go:12-15`.
- `Matrix.LU` permutation and pivoted `Det` — `mat.go:557-622`.

## Dependencies and scope

Dependencies: None blocking. Not on the M-a stage 1 path after O2; the DH orientation output depends on it.

Not in scope: Homogeneous helpers and `Matrix4x4.Inverse` (now ER-018); wide pseudo-inverse is ER-017's DLS.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **T-IK-04** — failing test first: Rz(90°) → `[0 0 √½ √½]`; `Orientation(Quaternion(R)) == R`. Fix: `mat.go:626-648`; pick one layout repo-wide.
- [ ] QR: failing test Q·R ≈ A and QᵀQ ≈ I in `verifyQRDecomposition` (`qr_test.go`), then the fix.
- [ ] Inverse: failing test that `rigidbody.NewModel` accepts a 1 kg, 0.2 m box (probe `tmp/easyrobot/ik/inertia`); relative tolerance.
- [ ] LU/Det: `TestMatrix_Det`/`TestMatrix_LU` (`mat_full_test.go:1309`, `:1336`) filled with matrices needing pivoting.
- [ ] `go test -count=1 ./x/math/mat/... ./x/math/vec/...` passes.

## Effort uncertainty and revision trigger

Product's guess: none given; **S–M** (inferred). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
