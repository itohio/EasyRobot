---
id: ER-069
title: "YOLOX-nano on the UnitV in TinyGo: camera frames with detections streamed over USB to a static web page (CPU path)"
type: ticket
status: active
kind: hardware
owner: EasyRobot
workstream: [K210-nn, W5]
milestone: YOLO-K210
tags: [easyrobot, ticket, k210, yolo, tinygo, nn, web-serial]
created: 2026-11-19
updated: 2026-11-19
---
# ER-069 — YOLOX-nano on the UnitV: detections streamed to a web page (CPU path)

[[tickets/index|Ticket index]] · [[roadmap#8. Next|roadmap §8]] · [[k210-kpu]] · [[kpu-compiler]]

**Outcome (owner, 2026-11-19):** the UnitV on its USB-C cable alone streams its camera image with the detections to a static web page, which draws the image, the boxes, the class names and the confidences, live. It proves that the model runs adequately on the board, so that it can go into larger systems. This ticket builds the whole path on the K210's CPU (qnet's int8 kernels), one vertical slice per step. Speed up to video is [[ER-070-yolo-k210-kpu|ER-070]], and this ticket's output is ER-070's on-board reference.

## What exists (main)

- `x/math/nn/qnet/tfl` loads the YOLOX-nano int8 TFLite file (RTS export of Megvii's `yolox_nano`, Apache-2.0, `huggingface.co/rts-ieper/yolox-nano-coco`, `perchannel.tflite`): 416×416×3 uint8 in, scores [3549×80] and box regression [3549×4] out, 1,204 qnet layers, 522.6 MMAC, 0.90 MB of weights. On the host, qnet's output equals LiteRT's on every layer.
- `x/vision/detect/yolox` decodes the heads into boxes and `x/vision/detect/nms` suppresses overlaps, with no allocation after setup.
- `drivers/unitv/examples/yolo` (step A): the detector on the board, its arena in the KPU memory lent to the CPU (`kpu.LendMemory`).
- `drivers/camera` (OV2640 over DVP; RGB565, Y8, JPEG) and `drivers/camera/frame` (camera frame format v1: chunked, CRC-32, resend control) with `cmd/camframe` on the host.
- The UnitV's USB serial link: loss-free at 115200 baud (~11 KB/s); at 1.5 Mbaud it delivers ~64 KB/s paced but drops bytes, which frame v1's chunk resend repairs. Other rates give garbage ([[ER-044-k210-frame-over-uart|ER-044]]).

Model files are never committed: the example takes the model from a file outside the repository at build time, and its README says where to get it (URL, licence, SHA-256).

## Steps

| Step | Scope | Check |
|---|---|---|
| A ✓ | Example `drivers/unitv/examples/yolo`: model and one test image embedded at build time, qnet runner with its arena allocated once, YOLOX decode + NMS, results over UART | done: board output hashes and detections equal the host's; 32.8 s per frame at 416 on the CPU; steady allocations |
| B | **One camera frame end to end.** The board captures one OV2640 frame, letterboxes it to 416×416, runs the detector, and sends the image and its detections over the USB serial link in a framed protocol: frame v1 chunks for the image, plus one CRC-protected detection record (frame sequence, then per detection class, score, box in image pixels). A **static page**, one HTML file in the repo with no server, no build step and no network, reads the stream through the browser's serial API (Web Serial) and draws the image, the boxes, class names and confidences. A **replay file** of a recorded stream lets the page run without the board, through a file input. | Host: encoder/decoder round trip and reject cases for the detection record (red first); the replay file decodes in a host test to the same detections as the board's log line. Board: the page, opened locally, shows a frame of the room with boxes that match the objects; the same page shows the replay file. Measured: bytes per frame, link throughput, ms for capture, letterbox, detect, send |
| C | **Continuous stream.** The same, frame after frame, at whatever the CPU gives (~33 s per frame now); the page shows the newest frame and the time per stage | 10-minute run: every frame received or dropped visibly (no stale boxes on a new image), allocations steady on the board, frames per second end to end |

## Not in scope

- Speed: the KPU and a smaller input ([[ER-070-yolo-k210-kpu|ER-070]]); a compressed or smaller image on the link once detection is faster than the link (see ER-070's frame-rate table).
- Logging to the SD card; Wi-Fi (none on any board); hosting the page anywhere.

## Acceptance evidence required later

- [x] A: build size and memory map; board output hashes equal the host's; detections equal; ms per frame by layer kind; steady heap.
- [x] B: camera QVGA RGB565 → letterbox → YOLOX-nano → frame v1 image + detection record at 115200 baud (image 13.4 s on the line, 0 CRC errors in 8 images); on 4 camera frames the board's detector input and outputs equal a host run bit for bit; a frame takes 46.4 s; protocol, letterbox, streamer and replay tests; the one-file page draws a replay and a capture. Boots clean with the precise collector and a cleared stack (20/20, heap ~2.37 MB after setup). Still open: a screenshot of the page connected to the live board (Web Serial needs a person to pick the port).
- [ ] C: 10-minute run log and the end-to-end frame rate.
