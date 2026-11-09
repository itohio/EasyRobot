---
type: architecture
status: active
tags: [easyrobot, architecture, build, tinygo]
workstream: [W1, W2, W4, W9]
source: [go.mod, Makefile, x/devices/, cmd/elrs-rx/, drivers/lidar/, drivers/manipulator/, x/types/, x/iter/loop_experiments/, x/marshaller/storage/]
evidence: "cross-compiles 2026-11-09 with GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly CGO_ENABLED=0 go build ./... for windows/amd64, linux/arm, linux/riscv64, js/wasm; TinyGo file-selection via go list -tags; TinyGo tag-set compile with gc (GOOS=linux GOARCH=arm); desktop results from baseline-2026-11-08"
---
# Build targets

What builds where, what is untested, and the traps in the build-tag split. Layering: [[overview]]. Package status table: [[packages]]. Desktop results are from the [[baseline-2026-11-08|baseline]] (same HEAD); cross-target rows below were run 2026-11-09 (compile only, no execution). **No TinyGo build, no board, no OpenCV** was used (roster rules); TinyGo rows are tag-selection and type-check evidence only.

## Native (cgo) dependencies

The repo has no `import "C"` (baseline). cgo enters only through modules:

| Module | Version | Needed by | Native requirement | On this host |
|---|---|---|---|---|
| `gocv.io/x/gocv` | v0.42.0 | 33 packages: all `x/vision/*` but `match/corners`, `x/marshaller/gocv`, `x/math/tensor/gocv`, `cmd/{calibrate,cr30,display,spectrometer}`, `drivers/lidar` (desktop) | OpenCV 4 (`pkg-config opencv4`) | missing → not built |
| `github.com/gen2brain/malgo` | v0.11.24 | `x/marshaller/audio`, `cmd/audio` | vendored miniaudio C, cgo | builds on darwin with clang |
| `github.com/mattn/go-tflite` | v1.0.5 | `x/marshaller/tflite` (`//go:build linux`) | libtensorflowlite_c | not present |
| `github.com/vladimirvivien/go4vl` | v0.3.0 | `x/marshaller/v4l` (`//go:build linux`) | Linux V4L2 headers (cgo) | linux only |
| `github.com/tarm/serial` | 2018 pseudo-version | `cmd/manipulator` via `dndm/network/serial` (`go list -deps ./cmd/manipulator`) | cgo on darwin | builds on darwin |

`Makefile` sets `CGO_ENABLED?=0` but `test` (`go test -race ./...`) needs cgo for `-race`; there is no target matrix in the Makefile.

## Desktop and Linux targets

| Target | Command | Result | Source |
|---|---|---|---|
| darwin/arm64 (cgo on) | per-package `go build -o /dev/null` | 155 ok; 33 OpenCV missing; 6 darwin gaps (F1 `x/devices/{encoder,motor}` stub signature, F2 no darwin serial file → `cmd/monitor`, `cmd/spectrometer/*`, F3 `cmd/elrs-rx` desktop file is `linux \|\| windows`); 1 arm64 asm gap (F4) | baseline §3–4 |
| darwin/amd64 `CGO_ENABLED=0` | `go build ./...` | as darwin/arm64 minus the asm gap; plus malgo/tarm need cgo | baseline §3 |
| linux/amd64 `CGO_ENABLED=0` | `go build ./...` | fails only in gocv, go-tflite, go4vl, malgo (cgo) — **no project-code failure** | baseline §3 |
| linux/arm64 `CGO_ENABLED=0` | `go build ./...` | as linux/amd64 + `x/iter/loop_experiments` (asm) | baseline §3 |
| windows/amd64 `CGO_ENABLED=0` | `go build ./...` | fails in `x/devices/encoder` (`encoder.go:77`: `configurePins(...) (no value) used as value` — F1 also hits Windows, because `stub.go` is `!tinygo && !linux`), `x/marshaller/audio` (malgo, cgo), gocv | run 2026-11-09 |
| linux/arm (32-bit) `CGO_ENABLED=0` | `go build ./...` | fails in `x/iter/loop_experiments` (`bce_experiments.go:317–318: missing function body`), malgo, go-tflite, go4vl, gocv | run 2026-11-09 |
| **linux/riscv64** `CGO_ENABLED=0` (K230 Linux core class) | `go build ./...` | fails only in `x/iter/loop_experiments`, malgo, go-tflite, go4vl, gocv: **all pure-Go packages compile for riscv64** | run 2026-11-09 |
| js/wasm `CGO_ENABLED=0` | `go build ./...` | fails in `cmd/elrs-rx` (no platform file), `cmd/spectrometer/ports`, `x/devices/encoder`, `x/iter/loop_experiments`, **`x/types/st`** (`cast.go:82,293: undefined: elemConvertToInt` — defined only in `cast_i32.go`/`cast_i64.go`, whose GOARCH lists omit `wasm`), `tarm/serial`, gocv | run 2026-11-09 |

Nothing above was executed except darwin/arm64 tests (baseline). Linux runtime (`x/devices/*_linux.go` spidev/i2c-dev/gpio, V4L2, TFLite) is **untested**.

## TinyGo targets present in the tree

Build tags in use (first `//go:build` line per file, `grep -rh '^//go:build'`): `tinygo` (8 files), `!tinygo …` (19), `rp2040` (2), `esp32` (1), `sam && xiao` (4, + `planar`/`dh` variants 3), `xwpftb && rp2040`, `ld06 && rp2040`, `!rp2040 && cgo`, and `!sam || !xiao`. Old-style `// +build` remains in 6 files (`ants`, `logless`, `tflite`, `!wasm`).

| Target | Files | Status |
|---|---|---|
| `esp32` (Xtensa) | `cmd/elrs-rx/platform_esp32.go` only | **only ESP32 code in the repo**; PWM output is a stub (`setPWMMicroseconds` TODO, `platform_esp32.go:56–59`) and battery voltage is hard-coded 11.1 V (`:52–54`) |
| `rp2040` | `cmd/elrs-rx/platform_rp2040.go`, `drivers/lidar/main_rp2040.go` (+ `main_ld06.go`, `main_xwpftb.go`) | not built |
| `sam && xiao` (SAMD21 XIAO) | `cmd/elrs-rx/platform_xiao.go`, `drivers/manipulator/*`, `x/devices/xiao/xiao_pwm.go` | not built; flashing only via the confirm-first `make flash-manipulator` (the `go:generate tinygo flash` line was removed in [[ER-cleanup]]) |
| generic `tinygo` | `x/devices/{i2c,pin,serial,spi}_tinygo.go`, `x/devices/encoder/tinygo.go`, `x/devices/servo/array_tinygo.go`, `x/marshaller/storage/flash.go`, `x/devices/tca9548a/router_example_test.go` | not built |
| `unitv` (K210; `drivers/unitv/target/unitv.json`, inherits `maixbit`) | `drivers/unitv/…` | **built and run**: `hello` on the board ([[ER-K210-M1]]); build from the repo root (`make unitv`) |

### TinyGo facts from the installed target files (read, not run)

TinyGo 0.43.0 is installed at `/opt/homebrew/Cellar/tinygo/0.43.0` (not executed). Its target JSONs set **`"goos": "linux", "goarch": "arm"` and build tag `linux`** for `xtensa` (ESP32/ESP32-S3), `cortex-m` (RP2040, SAMD21) and `riscv` (ESP32-C3, generic RISC-V) (`targets/{xtensa,cortex-m,riscv}.json`). `esp32.json` adds tags `esp32 esp espradio` and `"scheduler": "tasks"` (the `xtensa` base has `"scheduler": "none"`); `esp32c3.json` inherits `riscv32`, tags `esp32c3 esp espradio`.

Consequences, confirmed by emulating TinyGo's tag sets with `GOOS=linux GOARCH=arm go list -e -tags tinygo,<board tags>,baremetal,linux,arm`:

1. **`linux || windows` files are selected under TinyGo.** `cmd/elrs-rx` resolves to `main.go platform_desktop.go platform_esp32.go` (esp32), `… platform_desktop.go platform_rp2040.go` (rp2040), `… platform_desktop.go platform_xiao.go` (xiao): `initHardware`, `getStream`, `onChannels`, `onLinkStats`, `heartbeat`, `readBatteryVoltage` are then declared twice → no TinyGo build of `cmd/elrs-rx` can succeed as tagged (`platform_desktop.go:1`). Fix: `//go:build !tinygo && (linux || windows)` (and add darwin, F3).
2. `drivers/lidar` under rp2040 tags selects `config.go device_factory.go display.go main_rp2040.go` (+ `main_ld06.go` with `ld06`): the three desktop files are `linux || windows` and pull in `gocv.io/x/gocv`, `x/marshaller/gocv`, `x/math/tensor/gocv` → the firmware cannot build (`go list -f '{{.Imports}}'` with those tags).
3. `x/marshaller/tflite` and `x/marshaller/v4l` (`//go:build linux`) are selected under every TinyGo target; harmless unless imported, but `linux` alone is the wrong guard.
4. Files using `!tinygo && linux` (all of `x/devices/*_linux.go`) are correct.

### TinyGo tag-set compile check (gc compiler, not TinyGo)

`GOOS=linux GOARCH=arm CGO_ENABLED=0 go build -tags tinygo,esp32,esp,xtensa,baremetal,linux,arm <pkgs>` over the 140 of 181 packages under `x/`, `types/`, `internal/` that neither import `machine` (17 `x/devices*` packages do) nor depend on cgo modules (24: GoCV users, malgo, go-tflite, go4vl); same with rp2040 tags. Package list: `~/src/tmp/easyrobot/arch/tinygo-esp32-list.txt`.

- 138/140 compile. Failures: `x/iter/loop_experiments` (amd64 asm) and **`x/marshaller/graph`** (`storage.go:11,22,23`: `storage.NewFileMap`, `storage.TarFactory`, `storage.NewTarMap` are `!tinygo`-only in `x/marshaller/storage`).
- What this does **not** check: TinyGo's own compiler (reflection limits, missing stdlib pieces such as parts of `encoding/*`/`os`, goroutine/scheduler behaviour, `unsafe` use), code size, heap/stack limits, float performance on Xtensa (single-precision FPU only on ESP32 [INFERENCE: Xtensa LX6 FPU is single-precision]), or anything under `x/devices`.
- Word-size split: TinyGo's `goarch=arm` picks the `*_i32.go` files in `x/types`, `x/types/st`, `x/types/internal` (lists `386 || arm || mips || mipsle`); `riscv64` is in the i64 list; `wasm`, `loong64` and `riscv32`-as-GOARCH are in neither list (wasm failure above).

## K230 (Kendryte) and other accelerators

No code: `grep -rli -E "k230|kendryte|canmv|nncase|k210|k510"` over the repo (excluding `docs/`, `.git`) hits only a byte sequence inside `x/marshaller/testdata/mnist_model.h5`; `m5stack|ov2640|ov5640|gc2145` → none (a `unitv` hit is `UnitVector` in `x/math/vec/vec3d_test.go`). The only evidence relevant to a K230 Linux core is the linux/riscv64 cross-compile above. Prior planning tickets: `docs/tickets/ER-007`/`ER-008` (K230 feasibility/adapter), `ER-005`/`ER-006` (Xtensa). See [[W4-k230-feature-stream]], [[W9-go-nn-framework]].

## Assembly (arm64 gap and beyond)

Only amd64 assembly exists: `x/iter/loop_experiments/{asm,asm_op,asm_matmul}_amd64.s`, `x/types/st/convert_asm_amd64.s` (`find . -name '*.s'`). The Go declarations have no generic fallback: `x/iter/loop_experiments/matmul.go:5` (`asmMatMul`), `bce_experiments.go:317–318`, and `x/types/st/convert_bench_test.go:583` (`convertInt16ToFloat32Assembly`, test-only) → build failures on arm64, arm, riscv64 and wasm. No arm64 NEON, Xtensa or RISC-V vector code exists anywhere. [INFERENCE] TinyGo does not assemble Go `.s` files, so any embedded speed-up needs pure-Go kernels or C/LLVM intrinsics. Optimisation planning: [[W2-code-optimisation]], [[primitive-kernels]].

## Feature tags and heavy tests

`use_mt`/`!use_mt` (9+9 files: threaded vs single-thread loops in `x/iter`, `x/workers`) — `use_mt` builds were never run (baseline §6). `gocv_display` (manual display test), `tflite`, `mnist_test`, `mnist_large_test`, `xor_test` (heavy tests, not run). The `logless` (`x/logger`) and `ants` (`internal/concurrency`) tags went with those packages in [[ER-cleanup]]; the manipulator flash commands no longer pass `-tags logless`.

## What is untested, per target

| Target | Builds (evidence) | Runs/tests |
|---|---|---|
| darwin/arm64 | 155/197 packages | 56 test packages pass, 16 fail, 11 fail to compile tests (baseline) |
| linux amd64/arm64/riscv64, arm | pure-Go packages compile (`CGO_ENABLED=0`) | nothing executed; spidev/i2c/gpio/V4L2/TFLite untested |
| windows/amd64 | all but `x/devices/encoder` + cgo packages | nothing executed |
| js/wasm | fails in `x/types/st` (blocks `x/types`, `x/iter`, tensor stack) | — |
| TinyGo esp32/rp2040/xiao | gc type-check of 138/140 portable packages under TinyGo tags; firmware mains fail by tag selection (elrs-rx, lidar) | no TinyGo build, no board |
| K230 | — (no code) | — |

## TODOs

- [ ] Guard `drivers/lidar/{config,device_factory,display}.go` with `!tinygo` so the rp2040 firmware does not pull GoCV — `drivers/lidar/config.go:1` #p1 #ER-026
- [ ] Make `x/devices/encoder/stub.go` (and motor) return `error` like the linux/tinygo files — `x/devices/encoder/encoder.go:77` (darwin + windows) #p1 #ER-026
- [ ] Add `wasm` (and `loong64`, `riscv32`) to the word-size GOARCH lists or use `math/bits.UintSize` — `x/types/st/cast_i32.go:1`, `cast_i64.go:1` #p2
- [ ] Split `x/marshaller/graph/storage.go` so the graph marshaller builds without file/tar storage on TinyGo — `x/marshaller/graph/storage.go:11–23` #p2
- [ ] Use `!tinygo && linux` for `x/marshaller/{tflite,v4l}` — `x/marshaller/v4l/device.go:1` #p3
- [ ] Add a CI/Makefile matrix: darwin+linux amd64/arm64, linux/arm, linux/riscv64, windows, wasm, plus `tinygo build` for esp32/rp2040/xiao once TinyGo runs are approved — `Makefile` #p2
- [x] Convert the 6 remaining `// +build` lines to `//go:build` — four were in files deleted by [[ER-cleanup]]; the other two (`x/marshaller/audio/device_malgo.go:2`, `x/marshaller/tflite_unmarshaller_test.go:2`) already carry `//go:build` #p3

## Inferred objectives (proposals)

- **Proposal:** treat "portable core compiles under TinyGo tags for esp32 and as linux/riscv64" as the gate for any code meant for the ESP32 recorder or the K230 Linux side; it already holds for 138/140 portable packages under gc.
- **Proposal:** the first real TinyGo build (with owner approval of TinyGo runs, no flashing) should be `cmd/elrs-rx` for `esp32-coreboard-v2` after the tag fix — smallest existing ESP32 program.
