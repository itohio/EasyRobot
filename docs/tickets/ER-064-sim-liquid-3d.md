---
id: ER-064
title: "3D liquid automaton driven by the IMU's continuous gravity vector"
type: ticket
status: proposed
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-12
---
# ER-064 — 3D liquid automaton driven by the IMU's continuous gravity vector

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · Brief: `~/src/docs/tasks/ER-SIMALGO.md` (owner 2026-11-12: "liquid engine belongs in easyrobot …"; paths and RNG per Leader's decisions there)

**Brick:** the 2D liquid's rules (ER-061) on a w×h×d grid, driven by the IMU's gravity vector as a continuous direction, not snapped to 26 directions. Small tilts move the liquid proportionally.

## Goal and evidence

Owner (ER-SIMALGO): the liquid engine is 2D now, 3D next. The badge's MPU6050 gives a gravity vector ([[mpu6050-imu-driver]], tilt via [[mahony-ahrs]]). Owner, 2026-11-12: a slight tilt left the snapped 2D water frozen, so the engine must follow continuous gravity in 3D too.

## Dependencies and scope

Same package as ER-061 or `liquid3`, decided when ER-061 lands. Memory budget on the RP2040 (264 KB RAM): e.g. 32×32×16 = 16 KB at one byte per cell. Rendering a 3D grid on the 2D screen is the demo's.

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, verify, then commit. No hardware or flashing is authorised by this ticket; tinygui runs it on the badge.

## Acceptance evidence required later

- [ ] Count conserved for gravity directions sampled over the sphere; a basin levels perpendicular to an oblique gravity within a bounded number of steps.
- [ ] Proportional tilt: net sideways transport grows monotonically with tilts of 2°, 5°, 10° and 20° about either horizontal axis. Tilt 0 gives no drift.
- [ ] Step time on the RP2040 within the badge's frame budget (measured by tinygui); 0 allocations.

## Effort uncertainty and revision trigger

easyrobot's guess: **M**. Revisit when the badge's frame budget or the owner's scope changes.

## Implementation evidence

None.
