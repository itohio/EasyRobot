---
id: ER-069
title: "YOLOX-nano on the UnitV in TinyGo, CPU path: still image, then camera frames, detections over UART"
type: ticket
status: ready
kind: hardware
owner: EasyRobot
workstream: [K210-nn, W5]
milestone: YOLO-K210
tags: [easyrobot, ticket, k210, yolo, tinygo, nn]
created: 2026-11-19
updated: 2026-11-19
---
# ER-069 — YOLOX-nano on the UnitV in TinyGo, CPU path

[[tickets/index|Ticket index]] · [[roadmap#8. Next|roadmap §8]] · [[k210-kpu]] · [[kpu-compiler]]

**Outcome:** the UnitV runs YOLOX-nano from a TinyGo build and prints, per frame, the detected classes, boxes, confidences and the frame time. This ticket is the end-to-end slice on the K210's CPU (qnet's int8 kernels). [[ER-070-yolo-k210-kpu|ER-070]] then moves the convolutions to the KPU, and this ticket's output is ER-070's on-board reference.

## What exists (main)

- `x/math/nn/qnet/tfl` loads the YOLOX-nano int8 TFLite file (RTS export of Megvii's `yolox_nano`, Apache-2.0, `huggingface.co/rts-ieper/yolox-nano-coco`, `perchannel.tflite`): 416×416×3 uint8 in, scores [3549×80] and box regression [3549×4] out, 1,204 qnet layers, 522.6 MMAC, 0.90 MB of weights. On the host, qnet's output equals LiteRT's on every layer.
- `x/vision/detect/yolox` decodes the heads into boxes and `x/vision/detect/nms` suppresses overlaps, with no allocation after setup.
- `drivers/unitv` (board target, examples) and `drivers/camera` (OV2640 over DVP) run on the board.
- Activation peak by tensor liveness at 416: 1.78 MB.

Model files are never committed: the example takes the model from a file outside the repository at build time, and its README says where to get it (URL, licence, SHA-256).

## Steps

| Step | Scope | Check |
|---|---|---|
| A | Example `drivers/unitv/examples/yolo`: model and one test image embedded at build time as strings (one copy in memory, no writable duplicate), qnet runner with its arena allocated once, YOLOX decode + NMS, results over UART | TinyGo build for the UnitV target: image size and the memory map (code, model, arena, stacks) fit the K210's 6 MiB general SRAM. On the board: FNV-1a of both output tensors equals the host qnet run on the same image; the detection list equals the host's; ms per frame, split by layer kind; heap unchanged across 10 runs |
| B | Camera frames: OV2640 → letterbox to 416×416 → the same pipeline, one result line per frame | camera pointed at the room with objects around: per-frame classes, boxes, confidences and fps logged for ≥ 2 minutes; one saved frame with its boxes drawn on the host matches the scene; no growth of the heap |

## Not in scope

- The KPU ([[ER-070-yolo-k210-kpu|ER-070]]).
- A smaller input than 416: it needs a re-export of the model (an owner decision, taken once ER-070's timing is known).
- Logging to the SD card: the UnitV has no working card yet.
- Wi-Fi (none on any board).

## Acceptance evidence required later

- [ ] A: build size and memory map; board output hashes equal the host's; detections equal; ms per frame by layer kind; steady heap.
- [ ] B: two-minute camera run log; one annotated frame recognisably showing the scene.
