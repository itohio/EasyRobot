---
id: ER-047
title: "UnitV camera → KPU → feature tensor stream"
type: ticket
status: blocked
kind: implementation
owner: k210
workstream: [K210-camera, K210-nn, W4]
milestone: K210-M7
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-047 — UnitV camera → KPU → feature tensor stream (K210 M7)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

Joins the camera and NN workstreams: the K210-scale version of W4's feature-tensor stream.

## Scope

- Firmware node (path proposed at M7, likely `drivers/unitv`): camera → KPU backbone → last feature map in W4 chunk framing.
- USB UART first, then Grove UART (G34/G35) to an ESP32.

## Acceptance

Sustained fps recorded; CRC-clean for 10 minutes on each link.

## Dependencies

ER-044, ER-046. One milestone per round, each with its evidence record.

## Evidence

—
