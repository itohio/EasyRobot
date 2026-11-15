---
type: workstream
status: proposed
tags: [easyrobot, workstream, k230, esp32, spi, camera]
workstream: W4
source: [x/devices/spi.go, x/devices/spi_linux.go, x/devices/spi_stub.go, x/devices/spi_tinygo.go, cmd/elrs-rx/platform_esp32.go, proto/types/math/tensor.proto, proto/types/core/frame.proto, x/marshaller/proto/convert.go, x/marshaller/storage/flash.go, x/marshaller/gocv/, x/marshaller/v4l/]
evidence: "source reading + probes (2026-11-09); K230 search grep -rli 'k230|kendryte|canmv|nncase|k210|k510' → no code"
---
# W4 — K230 feature stream to an ESP32 recorder

**Goal:** a K230 chip with a camera (M5Stack Kamera module) runs modern embedded-model feature extraction (no fully connected head, just the last feature tensor) and streams the tensor to an ESP32 board over SPI; the hardware needed for TinyGo camera and TinyGo K230 neural-net compute work is to be provided. The W4 breakdown reads "feature-tensor stream and recordings", i.e. the ESP32 records the stream. Hardware work needs approval of its exact scope.

Target pipeline, as stated: camera → K230 backbone (no classifier head) → last feature tensor → SPI → ESP32 → recording.

## What exists

| Piece | In the repo | Status / evidence |
|---|---|---|
| K230 code (CanMV, nncase, KPU, RISC-V) | **none** | `grep -rli -E "k230\|kendryte\|canmv\|nncase\|k210\|k510"` → only a byte sequence in `x/marshaller/testdata/mnist_model.h5`; tickets `docs/tickets/ER-007`/`ER-008` are discovery stubs ([[build-targets]]) |
| M5Stack camera / sensor drivers (OV2640, GC2145, …) | **none** | `grep -rli "m5stack\|ov2640\|ov5640\|gc2145"` → none |
| Camera capture (host) | `x/marshaller/gocv` (OpenCV `VideoCapture`, device controls, FOURCC), `x/marshaller/v4l` (Linux V4L2 via go4vl, `MultiStream` sync) → `types.FrameStream` of `types.Frame{Index, Timestamp, Metadata, Tensors}` | [[camera-io]]; not built here (OpenCV missing; v4l linux-only). [INFERENCE] if the K230 runs Linux and exposes `/dev/video*`, `x/marshaller/v4l` is the closest existing Go path; it compiles for `linux/riscv64` only if go4vl's cgo builds there (cgo deps excluded from the riscv64 cross-compile in [[build-targets]]) |
| Pure-Go portability to the K230 Linux core | 138/140 portable packages type-check under TinyGo tags; all pure-Go packages compile for `GOOS=linux GOARCH=riscv64` | [[build-targets]] (cross-compile 2026-11-09; not executed) |
| SPI abstraction | `devices.SPI{Tx(w, r []byte) error; Transfer(b byte) (byte, error)}` (`x/devices/spi.go:5–14`) — **master-only** semantics; Linux spidev implementation `LinuxSPI` (`spi_linux.go:46–185`); TinyGo file is empty ("Use device-specific SPI directly", `spi_tinygo.go:5`); stub `NewSPI()` takes no argument while Linux `NewSPI(device string)` takes one (`spi_stub.go:12` vs `spi_linux.go:58`) | [[device-io]]; no tests; only `x/devices/adns3080` uses `SPI` |
| SPI slave / DMA on ESP32 | **none** | no slave-mode interface anywhere in `x/devices` |
| ESP32 code | `cmd/elrs-rx/platform_esp32.go` (UART1 CRSF at 420 kBd, 6 PWM pins; PWM output and battery ADC are TODO stubs, `:52–59`) | the only ESP32 file; cannot build under TinyGo as tagged (desktop file also selected, [[build-targets]]) |
| Tensor serialisation | `types.math.Tensor` (`proto/types/math/tensor.proto`, `data_bytes` field 10), `types.core.Frame` (`frame.proto`); `x/marshaller/proto` (uses the orphan `easyrobot.core.types.Tensor`) | [[dndm-and-protobuf]]. Probe: `x/marshaller/proto` **drops INT8/INT16 data and panics on UINT8**; `data_i8` costs 344,972 B vs 62,732 B with `data_bytes` for a 7×7×1280 int8 map |
| Quantisation | `x/math/learn` post-training quantisation, `x/math/primitive/{qi8,qu8,…}` int kernels | [[learn]], [[primitive-kernels]]; `types.math.Tensor` has no scale/zero-point fields |
| Recording / storage | `x/marshaller/storage`: file, mmap, tar (desktop, `!tinygo`), memory (all), flash (`tinygo`): `NewFlashMap()` is a placeholder that always errors (`flash.go:267–281`); `NewFlashMapFromData(data, ro)` wraps a byte slice | [[marshallers]]; storage tests pass on darwin (18, baseline). **No SD card / FAT / SDMMC support** anywhere |
| Transport | dndm over serial (`network/serial` → `tarm/serial`), `stream.NewWithContext` over any `io.ReadWriter` | [[dndm-and-protobuf]]; firmware wiring broken (invalid route, tarm on MCU); no SPI link type |
| NN on device | native Go [[tensor]]/[[nn]] (Conv2D, DepthwiseConv2D in eager `x/math/tensor/eager/conv.go:1247`, pooling, BatchNorm2D, activations incl. ReLU6/Swish); TFLite wrapper (linux, cgo) | [[nn]], [[tflite]], [[W9-go-nn-framework]]; no KPU/nncase backend; eager tensors are NCHW (`conv.go:1256`), K230/TFLite models are commonly NHWC [INFERENCE] |

## Gaps

1. **K230 side does not exist**: no capture, no model runner, no SPI master code, no build for its cores. Whether it runs Go at all (Linux big core vs RTOS/MicroPython CanMV image) is unknown here.
2. **ESP32 SPI slave receive path**: the `SPI` interface is master-only; TinyGo's SPI slave support on ESP32 is unknown (research question).
3. **Framing over SPI**: no length-prefixed/CRC'd frame format for SPI; dndm has no SPI link; flow control (who clocks, ready/IRQ line, back-pressure when the SD card stalls) is undefined.
4. **Tensor contract**: no scale/zero-point/layout fields; int8 path broken in `x/marshaller/proto`; two tensor schemas.
5. **Recorder storage**: no SD/FAT writer, flash storage is a placeholder.
6. **Timing**: no timestamp sync between K230 and ESP32 (Frame has `timestamp` but no clock source contract).

## Budget arithmetic (sizes and rates; no hardware facts verified)

Feature-map sizes for candidate backbones (standard architecture shapes, [INFERENCE] to confirm on the chosen model), int8 = 1 B/value:

| Backbone @ input | Last feature map | Bytes/frame (int8) | @10 fps | @30 fps |
|---|---|---|---|---|
| MobileNetV2 1.0 @224 (after final 1×1 conv) | 7×7×1280 | 62,720 | 627 kB/s = 5.02 Mbit/s | 1.88 MB/s = 15.1 Mbit/s |
| MobileNetV2 1.0 @224 (before final 1×1 conv) | 7×7×320 | 15,680 | 157 kB/s = 1.25 Mbit/s | 470 kB/s = 3.76 Mbit/s |
| ResNet-18 @224 | 7×7×512 | 25,088 | 251 kB/s = 2.01 Mbit/s | 753 kB/s = 6.02 Mbit/s |
| YOLOv8n backbone P5 @320 | 10×10×256 | 25,600 | 256 kB/s = 2.05 Mbit/s | 768 kB/s = 6.14 Mbit/s |
| YOLOv8n P3+P4+P5 @320 (for a detection head) | 40×40×64 + 20×20×128 + 10×10×256 | 179,200 | 1.79 MB/s = 14.3 Mbit/s | 5.38 MB/s = 43.0 Mbit/s |

- **Wire overhead:** `types.math.Tensor` with `data_bytes` adds 12 B to 62,720 B (probe); with `data_i8` it is 5.5× larger — never use `data_i8` for this.
- **SPI capacity** (single-lane, 8 bits/byte, no gaps): 10 MHz → 1.25 MB/s; 20 MHz → 2.5 MB/s; 40 MHz → 5 MB/s. So 7×7×1280 @10 fps needs ≥5.02 MHz of clock before framing/gaps; @30 fps ≥15.1 MHz; the three-scale YOLO set @30 fps (43 Mbit/s) exceeds a 40 MHz single-lane link.
- **ESP32 RAM:** double-buffering a 62,720 B frame needs 125,440 B; the classic ESP32 has 520 KiB SRAM in total [INFERENCE: datasheet figure, not verified], so this fits only with care; PSRAM boards relax it.
- **Recording volume:** 627 kB/s × 3600 s = 2.26 GB/hour (7×7×1280 @10 fps); 157 kB/s → 0.56 GB/hour (7×7×320 @10 fps). A 32 GB card holds ~14 h or ~57 h respectively.
- **SD write budget:** must sustain the table rate plus FAT/metadata overhead; the sustained SD write rate achievable from TinyGo on ESP32 is unknown (no SD code exists) — measure before choosing the backbone/frame rate.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Add an SPI slave (peripheral) receive interface with DMA-sized buffers to `x/devices`, separate from the master `SPI` interface — `x/devices/spi.go:5–14` #p2
- [ ] Unify `NewSPI` signatures across linux/stub/tinygo — `x/devices/spi_stub.go:12`, `x/devices/spi_linux.go:58` #p3
- [ ] Define the feature-frame wire contract: `types.core.Frame` + `types.math.Tensor` with `data_bytes`, plus scale, zero-point and layout fields — `proto/types/math/tensor.proto` #p2
- [ ] Implement a TinyGo SD/FAT (or raw-block) `MappedStorage`/append-only recorder and replace the placeholder `NewFlashMap` — `x/marshaller/storage/flash.go:267–281` #p2
- [ ] Specify SPI framing: sync word, length, sequence number, CRC (reuse `x/math/dsp/coding/crc` once its failing tests are fixed, [[line-coding-and-fec]]), ready/IRQ handshake — no code exists #p2
- [ ] Record the K230 platform facts (board, OS image, SDK/nncase versions, camera module) in ER-007 before any code — `docs/tickets/ER-007-k230-feasibility.md` #p2

## Inferred objectives (proposals)

- **Proposal:** start host-side: a Go "feature recorder" that reads `core.Frame` feature tensors from a file/serial/SPI-USB bridge and writes them with `x/marshaller/storage`, tested with synthetic tensors; the ESP32 then only has to implement the same frame reader + appender.
- **Proposal:** the ESP32 records raw length-prefixed frames (no protobuf decode on the MCU, just CRC check + append), and the host decodes later; this keeps ESP32 CPU and RAM free for SPI DMA and SD writes.
- **Proposal:** use dndm only on the host side at first; carry the SPI link as a dndm link type later, after the owner's link-context design lands ([[dndm-and-protobuf]], owner overhead requirement).
- **Proposal:** pick the backbone by the budget table: a 7×7×320 or 10×10×256 int8 map keeps SPI under ~6 Mbit/s at 30 fps and recording under ~2.8 GB/hour.

## Research questions for ideation

1. **Feature extractor on K230:** which backbone (MobileNetV2/V3, EfficientNet-Lite, YOLOv8n backbone, ResNet-18) compiles with nncase to the KPU at int8 and at what fps/latency on the owner's board? Which layer is the "last feature tensor" that also serves W5 (YOLO head, BoW, VO teacher)?
2. **K230 runtime choice:** CanMV MicroPython vs K230 Linux SDK (C/C++ nncase runtime) vs Go on the Linux core calling nncase via cgo — which gives camera access + KPU + SPI master with least glue? Does Go on `linux/riscv64` run on the K230's big core image?
3. **SPI on ESP32:** what is the maximum reliable SPI **slave** clock on ESP32/ESP32-S3 (GPIO matrix vs IOMUX pins), and does TinyGo 0.43 expose slave mode/DMA at all, or is a C (ESP-IDF) recorder needed?
4. **Tensor framing over SPI:** fixed-size DMA transactions with a header (sync, seq, len, CRC32) vs length-prefixed protobuf `Frame`; how to signal frame-ready (IRQ line) and handle ESP32 back-pressure when the SD card stalls?
5. **Compression:** is int8 → zero-run/entropy coding worthwhile for post-ReLU feature maps (many zeros), or does it cost more ESP32 CPU than it saves in SD bandwidth? Measure sparsity on real maps.
6. **SD recording rate:** sustained write rate of SDMMC 4-bit vs SPI-mode SD on ESP32 under TinyGo vs ESP-IDF, and whether pre-allocated contiguous files avoid FAT stalls; does it cover the table rates with margin?
7. **Clock sync:** how to timestamp frames consistently across K230 and ESP32 (frame counter + periodic sync message vs shared GPIO pulse)?
