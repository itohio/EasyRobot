# nnbench — int8 CNN benchmark for qi8 backends

Runs small int8 CNNs built as `x/math/nn/qnet` graphs, planned and run by
the qnet `Runner` (which prepares `x/math/primitive/qi8` layers), and
reports speed, cycles, RAM and an output checksum. One source, four builds:

| Build | Backend (`qi8.Backend()`) |
|---|---|
| TinyGo, ESP32-S3, default | `esp32s3-pie` (PIE SIMD conv2D/FC/depthwise; pools, Add generic) |
| TinyGo, ESP32-S3, `-tags qi8generic` | `generic` (pure Go) |
| TinyGo, K210 (UnitV) | `generic` (arena capped at 4 MiB) |
| host, `go run ./cmd/nnbench` | `generic` |

Weights (`qnet.SyntheticParams`) and inputs (SplitMix64) are deterministic and
their seeds are printed, so **`out_fnv` (FNV-1a of the final output) and
`all_fnv` (every layer and Add output of the first run) must be identical in all
builds**. That is the whole-network bit-exactness check across backends.

## Models

| Model | Layers | MACs |
|---|---|---|
| TinyCNN 32×32×3 | 3 × (conv 3×3/1 SAME + ReLU6, maxpool 2) with 16/32/64 ch, global avg pool, FC→10 | 2.80 M |
| MobileNetV1-0.25 96×96×3 | `qnet.MobileNetV1(0.25, 96, 96, 0)`: feature extractor, output 1×1×256 | 7.49 M |
| DS-CNN-style 64×64×3 | conv 3×3/2→32, 4 × (depthwise 3×3/1 + pointwise 1×1→64), global avg pool, FC→12 | 17.63 M |
| ResNet-8 32×32×3 | MLPerf Tiny image-classification topology: conv 3×3→16, 3 residual stacks 16/32/64 (2 × conv 3×3, stride 2 and a 1×1 projection shortcut in stacks 2–3, requantised `qi8.Add` + ReLU6), global avg pool, FC→10 | 12.50 M |

Convs fuse ReLU6 (`qnet.MobileNetActQuant`); FCs output logits
(`qnet.MobileNetLogitQuant`). In ResNet-8 the second conv of each branch and
the projection have no activation and their own quantisation, so every Add
rescales both inputs; the skip connections are qnet `Add` layers with two
producers (`Layer.From`).

## How it measures

- All weights, bias/requant, backend `Scratch` and activations come from one
  arena allocated once at start-up (on the chip: free heap − 32 KiB) and
  reused by every model. `qnet.PlanMemory` lays out the runner's part:
  backend scratch, then activations reused once every reader of a tensor
  has run (a sequential net reaches the ping-pong bound). When a layer's
  backend needs `Scratch` (PIE conv2D/FC/depthwise), its weights are placed
  at the start of that scratch (`MemPlan.ScratchSlice`), so `Prepare` packs
  them in place and big weights exist once.
- Per model: 1 warm-up run, then at least 10 runs and at least 2 s. Reports
  ms/inference (mean, min), CCOUNT cycles/inference, cycles/MAC (nominal MACs
  incl. padded taps), a per-kind breakdown (conv, depthwise,
  pointwise = 1×1 conv, fc, pool, add; Add also per element), arena use,
  heap numbers and heap allocations during the timed loop (must be 0).
- CPU MHz = CCOUNT delta / µs over a 1 s busy wait. Off the chip there is no
  CCOUNT: cycles read 0 and the breakdown is in ns.
- On the chip the report repeats every 5 s, so a monitor can attach any time.

## Build, flash, watch (XIAO ESP32-S3 on `/dev/cu.usbmodem101`)

From the repository root, TinyGo 0.43:

```sh
# PIE backend (default)
tinygo build -target ./cmd/nnbench/target/xiao-esp32s3-noradio.json \
  -o nnbench-pie.bin ./cmd/nnbench
esptool --port /dev/cu.usbmodem101 --chip esp32s3 write-flash 0x0 nnbench-pie.bin

# generic backend
tinygo build -target ./cmd/nnbench/target/xiao-esp32s3-noradio.json -tags qi8generic \
  -o nnbench-generic.bin ./cmd/nnbench
esptool --port /dev/cu.usbmodem101 --chip esp32s3 write-flash 0x0 nnbench-generic.bin

# watch (opening the port does not reset the board; the report repeats)
tinygo monitor -port /dev/cu.usbmodem101
# or: python3 -m serial.tools.miniterm /dev/cu.usbmodem101 115200

# K210 (UnitV) build: generic qi8 on the RISC-V cores, cycle counter via CYCLE
tinygo build -target=drivers/unitv/target/unitv.json -o nnbench-k210.bin ./cmd/nnbench

# host comparison (same checksums)
go run ./cmd/nnbench
```

`target/xiao-esp32s3-noradio.json` is TinyGo's `xiao-esp32s3` without the
`espradio` build tag and the Wi-Fi blob's `--wrap=malloc…` linker flags; its
`linkerscript` and `extra-files` are relative to `TINYGOROOT`, so it works on
any TinyGo 0.43 install.

## Results

Measured numbers are not kept in the repository (they change with the
code). The first measurement — both builds on a XIAO ESP32-S3 at 240 MHz,
with identical checksums — was recorded outside the
repository. Re-run the firmware to get current numbers.
