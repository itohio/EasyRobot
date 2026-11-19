---
id: ER-070
title: "YOLO on the K210's KPU: graph passes so the compiler runs long KPU segments, then fps on the UnitV"
type: ticket
status: proposed
kind: design
owner: EasyRobot
workstream: [K210-nn, W5]
milestone: YOLO-K210
tags: [easyrobot, ticket, k210, kpu, yolo, nn, compiler]
created: 2026-11-19
updated: 2026-11-19
---
# ER-070 — YOLO on the K210's KPU

[[tickets/index|Ticket index]] · [[roadmap#8. Next|roadmap §8]] · [[kpu-compiler]] · [[k210-kpu]]

**Outcome:** the stream of [[ER-069-yolo-k210-tinygo-cpu|ER-069]] (the UnitV's camera image with detections, drawn by a static web page) becomes video: the KPU runs the convolutions and the CPU only the rest. ER-069's CPU run is the on-board reference.

## Measured on the host (main `c19643ce`, 2026-11-19)

`compile.Compile(&g, kpu.NewEmulated)` on the YOLOX-nano int8 graph that ER-069 runs (`tfl.Load`, 1,204 layers, 522.6 MMAC):

- **Every conv but the stem lowers:** 454 layers claimed (all 94 depthwise and 348 of 352 conv layers, with their pads); 448 MMAC (86 %) can run on the KPU as the compiler stands.
- **Not claimed:** the four 6×6 stride-2 stem convs (the Focus stem folded into one conv, 74.8 MMAC).
- **But it gives 442 segments, nearly all of one layer:** every KPU layer would pay an upload and a read-back. Two properties of this export cause it:
  1. **SiLU is LOGISTIC + MUL** after nearly every conv (262 + 259 layers), and the compiler fuses only ReLU/ReLU6, so every conv ends its segment.
  2. **The export is "range-split":** tensors are cut into channel slices of 1–14 channels, each conv is computed per input slice and summed with ADD layers (188), and slices are joined by CONCAT. A 16-channel conv becomes 3–4 small convs plus adds.
- SPP max pools (5×5/9×9/13×13 stride 1), RESIZE_NEAREST, the head's TRANSPOSE/RESHAPE and the CONCATs stay on the CPU in any design (cheap at those sizes).

## Frame rate per speed option

Measured on the UnitV (ER-069 A, 416×416, CPU at 390.0 MHz, data cache on): **32.8 s per frame** at TinyGo's default `-opt=z`. That splits into 1×1 convs 18.8 s, 3×3/stem convs 3.2 s, depthwise 2.0 s, ADD 5.1 s, MUL 3.1 s, LOGISTIC 0.3 s and the rest 0.3 s. The same build at `-opt=2` takes 25.2 s, bit-exact.

A tight int8 dot loop bottoms out at ≈ 7 cycles/MAC (4.3 with two products per 64-bit multiply), so the CPU's convolution floor is ≈ 5.7 s per frame.

**KPU on YOLOX-shaped layers** (measured):

| Layer | KPU compute | Upload | Read-back |
|---|---|---|---|
| 1×1 conv 104×104×32→64 | 1.14 ms (22.2 M MAC, ≈ 19 G MAC/s) | 4.30 ms | 8.93 ms |
| depthwise 3×3 52×52×64 | 0.023 ms | 7.99 ms | 9.16 ms |

So every segment boundary costs **13–17 ms of CPU copies**, against ≤ 1.1 ms of compute. MobileNetV1-0.25 at 256×256 (~53 M MAC) runs its convs in 4.2 ms on this board, ≈ 12.6 G MAC/s.

The link (ER-044): a QVGA RGB565 image (153.6 KB) takes 13.4 s at the loss-free 115200 baud (measured again in ER-069 B), and ~2.4 s at 1.5 Mbaud with chunk resend. A QVGA JPEG of a room is ~4.7 KB (0.4 s at 115200).

| Option | Detect per frame | Frames/s (detect only) | Basis |
|---|---|---|---|
| 1. CPU, 416 (today) | 32.8 s (`-opt=z`); 25.2 s (`-opt=2`) | 0.03–0.04 | measured |
| 2. CPU, 416, range split merged (graph pass) | ~23 s at `-opt=2` | 0.04 | most of the ADD time (2.25 s at `-opt=2`) removed [INFERENCE] |
| 3. CPU, smaller input by a graph resize (no new download; accuracy to measure): 256 / 224 / 160 | ~9.5 / 7.3 / 3.7 s at `-opt=2` | 0.1 / 0.14 / 0.27 | time ∝ pixels [INFERENCE] |
| 4. KPU, compiler as it stands (442 one-layer segments) | ~10–12 s | 0.09 | convs ~40 ms on the KPU; 442 boundaries × 13–17 ms of copies ≈ 6–7 s; ADD + MUL + LOGISTIC stay on the CPU (4.3 s at `-opt=2`, measured) [INFERENCE from the measured layer costs] |
| 5. KPU + graph passes (SiLU as a fused table, range split merged, stem as space-to-depth), 416 | ~0.5–1 s | 1–2 | the CPU keeps residual adds, concats, resizes, SPP pools and the head, and each CPU boundary costs 13–17 ms [INFERENCE; open: SiLU table accuracy, boundary count] |
| 6. Option 5 at 224–256 | ~0.15–0.4 s | 3–6 | ×0.3–0.4 of option 5 [INFERENCE] |

Video needs option 5 or 6, or a detector built for the KPU. SSD-MobileNetV1 has its backbone in one KPU segment and few boundaries: ~10–15 frames/s estimated from the measured KPU rates. It is the recommendation on the table for the owner.

At those rates the link becomes the limit, so the image must travel as JPEG (~5 KB) or as a small Y8/RGB frame. That choice is made when detection gets below ~2 s per frame.

## Design questions (answered in a design note before code)

1. **SiLU on the KPU:** a 16-segment piecewise-linear table in the activation stage, fused into the conv. It is an approximation: measure its error per layer and its effect on detections against the exact CPU run.
2. **Undoing the range split:** merge each split conv back into one conv over the concatenated input and output, re-quantising the merged tensors from calibration images (int8 in, int8 out, a requantisation pass in Go), or keep the split and chain small KPU layers through AI SRAM (the adds stay a problem).
3. **The stem:** a 6×6 stride-2 TF SAME conv on an even input equals a space-to-depth by 2 (CPU, a copy) followed by a 3×3 stride-1 conv on 208×208×12, which the KPU runs exactly.
4. **Model source:** whether a clean export (one tensor per layer, at 416 or smaller) beats graph passes. A new export needs tools or weights not yet approved: an owner decision, with this ticket's numbers.
5. **Memory:** every KPU layer's input + output within 2 MiB of AI SRAM at 208×208 after the stem (else row-band tiling, which is the compiler's open K9g step). ER-069 lends that same 2 MiB to the CPU as its activation arena (`kpu.LendMemory`), so the KPU plan needs the CPU-side tensors back in main memory. That means a lighter `tfl.Load` (TinyGo's heap fragments after it today) or a smaller input.

## Steps (after the design note is accepted)

| Step | Scope | Check |
|---|---|---|
| D | Design note: the options above with host numbers (segments, uploads per frame, estimated KPU and CPU time, SiLU error) and a recommendation | reviewed and accepted before code |
| K1… | The chosen graph passes and compiler additions, each with a host check: per layer ≤ 1 LSB against qnet's CPU run where exact, error bound stated where approximate (SiLU); detections against the CPU run on ≥ 20 images | host, emulated KPU |
| K… | The same on the UnitV | per layer against the host emulation; detections equal ER-069's on the same image; ms per frame vs ER-069 |

## Acceptance evidence required later

- [ ] Design note accepted, with the owner's decision on the model source if needed.
- [ ] Host: segment count, per-layer error, detection agreement.
- [ ] Board: per-layer agreement, detection agreement, ms per frame.
