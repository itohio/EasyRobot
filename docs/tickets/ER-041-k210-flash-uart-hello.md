---
id: ER-041
title: "UnitV flashing path and UART hello"
type: ticket
status: done
kind: hardware
owner: k210
workstream: K210-camera
milestone: K210-M1
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-041 — UnitV flashing path and UART hello (K210 M1)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

First contact with the board (`/dev/cu.usbserial-6D52BBCC47`). Flashing this board and ktool in a uv venv were approved. From 2026-11-09 (mid-M1) only the camera and KPU matter, not the factory AI-camera image, so the factory backup and restore steps are dropped. Standing rule for every later session: the original board firmware does not matter and no Wi-Fi is required. No flash backups and no restores; test firmware stays on the board between sessions.

## Scope

- Read the factory boot banner at 115200 (identifies what was on the board).
- Flash `drivers/unitv/examples/hello` (then at `x/devices/unitv/examples/hello`) with `ktool`; observe a line per second.

## Acceptance

hello lines observed at 115200, one per second (host-timed); the flash command and its reset scheme recorded for later milestones.

## Dependencies

ER-040. One milestone per round.

## Evidence

Verified; review approved.
