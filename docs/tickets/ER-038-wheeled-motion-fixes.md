---
id: ER-038
title: "Wheeled motion fixes: path-tracking signs and steer4 geometry"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
maps: [T-IK-12, T-IK-13]
tags: [easyrobot, ticket, w3, w6, kinematics]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-038 — Wheeled motion fixes: path-tracking signs and steer4 geometry

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Wheel kinematics and path tracking for M-d's robot ([[motion-and-gait]], [[kinematics]]):

- **Path-tracking lateral PID wrong sign; heading error not wrapped** (E17) — `x/math/control/motion/rigidbody/motion.go:364-371`, `:504-510` ([[path-tracking-pid-controller]]).
- **steer4**: Ackermann angles use a rear-axle centre, wheel rates the chassis centre; drive flags unused (E18) — `kinematics/wheels/steer4/drive.go:69-72`, `:173-174`, `:28-29` ([[ackermann-steer4-kinematics]]).
- differential "wheel rates" are linear speeds; radius unused (E19, p3) — `wheels/differential/drive.go:11,27,80,116`; included because M-d names it.

## Dependencies and scope

Dependencies: None blocking. Needed before M-d stage 2 driving.

Not in scope: No new controllers.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **T-IK-12** — failing test first: probe M cases: path on the left ⇒ ω > 0; across the ±π seam ⇒ ω > 0. Fix: `motion.go:364-371`.
- [ ] **T-IK-13** — failing test first: wheel heading equals the rigid-body velocity direction at each wheel (< 1e-4 rad). Fix: one reference point.
- [ ] Differential: wheel rates in rad/s from a given radius, tested against v/r.

## Effort uncertainty and revision trigger

Product's guess: none given; **S** (inferred). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
