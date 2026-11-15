---
id: ER-042
title: "UnitV WS2812 LED and buttons"
type: ticket
status: backlog
kind: implementation
owner: k210
workstream: K210-camera
milestone: K210-M2
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-042 — UnitV WS2812 LED and buttons (K210 M2)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

UnitV's only LED is a WS2812 on G8: it needs 800 kHz timed pulses (bit-banged at 390 MHz, or SPI-encoded bits).

## Scope

- WS2812 driver (path decided if this ticket is ever scheduled).
- Example `drivers/unitv/examples/buttons`: A/B presses printed over UART.

## Acceptance

LED cycles red→green→blue (observed); button A/B presses print over UART; host test of the bit encoder where one exists.

## Dependencies

ER-041 ✓. Backlog (2026-11-09): K210 priority is camera and KPU work only.

## Evidence

—
