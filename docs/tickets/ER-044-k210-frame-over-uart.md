---
id: ER-044
title: "UnitV one camera frame over UART"
type: ticket
status: ready
kind: implementation
owner: k210
workstream: K210-camera
milestone: K210-M4
tags: [easyrobot, ticket, k210, tinygo]
created: 2026-11-09
updated: 2026-11-09
source_revision: db9eb8f2a50e0c84dc7f32a9109148dd13516d9e
---
# ER-044 — UnitV one camera frame over UART (K210 M4)

[[tickets/index|Ticket index]] · [[2026-11-09-k210-camera-nn|K210 plan]] · [[build-targets]]

## Goal

Smallest useful frame: 160×120 Y (19.2 KB). DVP writes memory itself; no DMA.

## Scope

- OV2640 tables in `drivers/camera/sensor/ov2640` (from esp32-camera v2.1.9, Apache-2.0).
- DVP frame capture (polled frame start/finish) in `drivers/camera/camera_k210.go`.
- Camera frame format v1 ("ERIM" chunks with CRC-32, shared with the ESP32 camera) via `drivers/camera/frame`; the shared host tool `cmd/camframe` writes PNGs.

## Acceptance

Acceptance (2026-11-09): capture a valid image and fetch it over UART to the Mac, with the camera pointed at a ceiling/wall with some objects around; the PNG is saved and recognisably shows the scene. Engineering targets: 160×120 Y, then QVGA RGB565, sent framed and CRC-clean (≥ 921600 baud if the app UART holds it); fps logged.

## Dependencies

ER-043 ✓.

## Evidence

Verified; code review pending: QVGA RGB565 and Y8 room images fetched over UART at 1.5 Mbaud; link loss ~1% of chunks with pacing (FT232BM). Lossless delivery with chunk resend: see `cmd/camframe` resend.
