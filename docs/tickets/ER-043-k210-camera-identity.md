---
id: ER-043
title: "UnitV camera identity via DVP SCCB"
type: ticket
status: ready
kind: implementation
owner: k210
workstream: K210-camera
milestone: K210-M3
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-043 — UnitV camera identity via DVP SCCB (K210 M3)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

The sensor varies by UnitV variant (OV7740 on U078-C; OV2640, M12). The K210 DVP has its own SCCB master. This unit has an **OV2640** (M3).

## Scope

- `drivers/camera/sensor` (host-independent SCCB `Bus`, `Identify`), `drivers/camera/dvp_k210.go` (clocks, XCLK, power/reset, SCCB master), `drivers/camera/camera_k210.go` (`Identify`), board pins `unitv.CameraPins`.
- Example `drivers/unitv/examples/camid`.

## Acceptance

Prints the sensor PID/VER over UART (e.g. `OV7740 77 42`).

## Dependencies

ER-041 ✓. Camera pins: the Sipeed/MaixPy IO40–47 mapping, confirmed on the board by the id read (no M5 config fetch was needed). Roster task ER-K210; evidence `docs/evidence/ER-K210-M3.md`.

## Evidence

Verify accepted (roster verification ER-CAM-BATCH-2); code review pending: [[ER-K210-M3]] (`camid` prints `sensor=OV2640 addr=0x30 pid=0x26 ver=0x42`).
