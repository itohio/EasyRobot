---
id: ER-030
title: "Vision compile fixes and phase-correlation maths"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W5
milestone: M-c
tags: [easyrobot, ticket, w5, vision]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-030 — Vision compile fixes and phase-correlation maths

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Accepted ER-R2 decision 7: hybrid vision — OpenCV for calibration/datasets on the host, pure Go on device ([[vision]], [[W5-feature-tensor-experiments]]).

- `x/vision/types` does not compile (14 type errors) — `x/vision/types/stereo_camera.go:35,84,108,135,181,190,193`.
- Signature drift — `transform/background/processor.go:35-37`, `transform/histogram/processor.go:274`, `extract/blobs/processor.go:78-79`.
- Unbound gocv calls (SGBM, reprojection) — `stereo/matching/processor.go:19-50`, `stereo/depth/processor.go:72`.
- **Phase correlation** lacks conjugate product and normalisation; wraps/crops in image size, not DFT size — `x/vision/odometry/phase/processor.go:196-236` ([[phase-correlation-odometry]]). Pure-Go path per the hybrid decision.

## Dependencies and scope

Dependencies: None blocking for the pure-Go phase-correlation maths. Building the gocv-dependent packages needs an approved OpenCV install on the host.

Not in scope: No relative pose, NMS, warp or dataset reader (M-c, unticketed).

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Failing test first: pure-Go phase correlation recovers a known integer and sub-pixel shift (±0.1 px) of a synthetic image.
- [ ] With OpenCV approved: `go vet ./x/vision/...` compiles; until then the pure-Go packages build and the gocv packages are listed as unverified.

## Effort uncertainty and revision trigger

Product's guess: none given; **M** (inferred). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
