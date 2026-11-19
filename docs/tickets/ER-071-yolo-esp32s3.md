---
id: ER-071
title: "YOLOX-nano on the ESP32-S3 in TinyGo: first run, then optimised kernels, detections from camera frames"
type: ticket
status: blocked
kind: hardware
owner: EasyRobot
workstream: [W2, W5, W9]
milestone: YOLO-S3
tags: [easyrobot, ticket, esp32s3, yolo, tinygo, nn]
created: 2026-11-19
updated: 2026-11-19
---
# ER-071 — YOLOX-nano on the ESP32-S3

[[tickets/index|Ticket index]] · [[roadmap#8. Next|roadmap §8]]

**Outcome:** the XIAO ESP32-S3 Sense runs the same YOLOX-nano as the UnitV ([[ER-069-yolo-k210-tinygo-cpu|ER-069]]) and logs classes, boxes, confidences and fps from its camera. The kernel speed-ups made for it also speed up the K210's CPU layers and the host.

## What exists

- On the host the int8 path is done (see ER-069).
- A TinyGo build for the S3 (stock `xiao-esp32s3` target, model and input embedded as strings so they stay in flash, arena in PSRAM) exists but has not run on the board.
- Estimate from measured S3 kernel costs (conv 1.5–2 cycles/MAC, depthwise 18–24, element-wise ~51 per element): ≈ 2.5 Gcycles, about 10 s per frame at 240 MHz before PSRAM penalties. Element-wise layers (SiLU's LOGISTIC + MUL, ADD) are about half of it.
- The PSRAM driver (`drivers/esp32s3/psram`) is on the `esp32cam` branch, not on main.

## Blocked on

- Merging the `esp32cam` branch into main (or a build tree with both).
- The board: it currently serves as a camera for another project; running this firmware replaces that firmware until it is flashed back.

## Steps

| Step | Scope | Check |
|---|---|---|
| A | First run of the existing build on a still image | output FNV-1a equal to the host qnet run; detections equal; ms per frame by layer kind; PSRAM and DRAM use |
| B | Optimise the slowest layer kinds first (element-wise SiLU, depthwise), each change its own commit with a host exactness test and a board timing | output unchanged (bit-exact); ms per frame before/after |
| C | Camera frames and the demo log | ≥ 2 minutes of per-frame results with the camera pointed at the room; one annotated frame matches the scene |

## Acceptance evidence required later

- [ ] A: hashes and detections equal the host's; timing.
- [ ] B: per-change timing, exactness kept.
- [ ] C: demo log and annotated frame.
