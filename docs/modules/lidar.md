---
type: module
group: "Devices and apps"
status: partial
tags: [easyrobot, devices, lidar]
workstream: W8
packages: [x/devices/lidar, x/devices/lidar/ld06, x/devices/lidar/xwpftb, x/devices/lidar/vl53l0x, drivers/lidar]
test_result: fail
---
# LiDAR drivers and node

**Purpose:** Turn 2-D range sensors into a common `lidar.Device` that delivers whole scans as a 2×N `float32` matrix (row 0 distance mm, row 1 angle deg), and publish them over [[dndm-and-protobuf]] as `types/devices.LIDARReading` from a host or RP2040 node.

**Packages:**

| Package | Files (LOC) | Tags | Role |
|---|---|---|---|
| `x/devices/lidar` | `lidar.go` (36) | — | `Device` interface |
| `x/devices/lidar/ld06` | `device.go` 506, `decoder.go` 96, test 109, DESIGN.md | — | LDROBOT LD06 UART decoder |
| `x/devices/lidar/xwpftb` | `device.go` 611, `decoder.go` 54, test 75, DESIGN.md | — | unidentified 55AA-protocol spinning lidar |
| `x/devices/lidar/vl53l0x` | `array.go` 243, DESIGN.md | — | N × VL53L0X ToF via TCA9548A as a fixed-angle "lidar" |
| `drivers/lidar` | `main.go` 387 (`!rp2040 && cgo`), `config.go`/`device_factory.go`/`display.go` (`linux \|\| windows`), `main_rp2040.go` 239 (`rp2040`), `main_ld06.go` (`ld06 && rp2040`), `main_xwpftb.go` (`xwpftb && rp2040`) | see column | host viewer/publisher and RP2040 firmware node |

## Public API

```go
type Device interface {               // x/devices/lidar/lidar.go:9
    Configure(init bool) error        // starts goroutines
    OnRead(fn func(matTypes.Matrix))  // callback per scan; view into internal buffer
    Read(dst matTypes.Matrix) int     // copy latest scan
    GetMinAngle() float32; GetMaxAngle() float32; GetPointCount() int
    Close()
}
ld06.New(ctx, ser devices.Serial, motor devices.PWM|nil, targetPoints, maxPoints) *Device
xwpftb.New(ctx, ser, motor, targetPoints, maxPoints) *Device
vl53l0x.New(ctx, vl53l0x.Config{Router *tca9548a.Router, Angles []float32, SensorAddresses []uint8})
```

Composition: `devices.Serial` (host: [[device-io]] `NewSerialWithConfig`; RP2040: `machine.UART` wrapper `drivers/lidar/main_rp2040.go:127-142`) → driver goroutine → `OnRead` → `matrixToLIDARReading` (copies into new slices, `main.go:314-340`) → `bus.Producer[*LIDARReading]` on intent `LIDARReading@lidar.scan` (RP2040, `main_rp2040.go:79-80`) or `--intent` routes via `cmd/display/destination.IntentDestination.SetLIDARReading` (host, `main.go:248-256`); host also draws a GoCV polar plot (`main.go:346-387`, 0° up, clockwise).

## Algorithms

<!-- vault:algorithms h=0b759d968b44 -->

Counts: 0 tested · 0 implemented · 3 partial · 0 stub · 0 unverified (3 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[ld06-packet-decoder-and-scan-assembly]] | partial | fail | 5 defects (VerLen, speed units, int16 angles, wrap test, test stream); never emits a scan |
| [[lidar-motor-point-count-pid]] | partial | none | Control sign inverted, never wired |
| [[xwpftb-frame-decoder-and-scan-assembly]] | partial | fail | Test never starts reader, format mismatch, angles > 245.75° wrong |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Decoders are pure Go, no CGO, no reflection; they use `float64` math per point, `sync.Mutex`, goroutines and `time` (fine on RP2040/ESP32 TinyGo, but float64 is soft-float on Cortex-M0+ [INFERENCE]). Buffers preallocated (`2·maxPoints` float32 + 4 KiB) but `View()` allocates per scan/`Read`, `matrixToLIDARReading` allocates two slices per scan, and xwpftb logs hex dumps per read (`device.go:178-189`).
- `drivers/lidar` host: needs CGO + OpenCV (GoCV) — not buildable here (baseline: OpenCV missing). Even with OpenCV it **cannot build on darwin**: `main.go` (`!rp2040 && cgo`) uses `Config`/`SetupDisplay` from `linux || windows` files and `devio.NewSerialWithConfig` (linux/windows only, F2).
- `drivers/lidar` RP2040: **does not compile** for either tag — `targetPoints` declared in `main_rp2040.go:41` and in `main_ld06.go:14`/`main_xwpftb.go:14`, `maxPoints` in `main_rp2040.go:42` and `main_xwpftb.go:15` (go/types check of the tag sets, probe). Further [INFERENCE]: TinyGo builds bare-metal targets with `GOOS=linux`, so the `linux || windows` files (`display.go` imports GoCV) would also be pulled into the RP2040 build. Motor PWM placeholder at `main_rp2040.go:62-68`.
- Serial on Linux is opened with `VMIN=0, VTIME=0` and returns `(0, nil)` when idle (`x/devices/serial_linux.go:86-87,178-180`); both lidar `readLoop`s spin without sleeping on that (`ld06/device.go:166-193`) → 100 % CPU core on the host.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 ./x/devices/...` (2026-11-09): `lidar/ld06` FAIL (TestCRC8 pass, TestParseAndAssembleFullRotation timeout 3.0 s), `lidar/xwpftb` FAIL (TestCRC16Cumulative pass, full-rotation timeout 1.0 s); `lidar`, `lidar/vl53l0x` no test files. `go vet ./x/devices/...` clean for lidar packages. Linux/windows arm64 cross-builds of `x/devices/...` OK for lidar packages.
- Probe (synthetic streams, no device): ld06 — 0 scans for the test stream, realistic packets (speed 3600/65535) and datasheet-format uint16 packets; uint16 359.60° decoded as 64.24°. xwpftb — test stream + Configure → 0 scans; decoder-format frames → 240-point scan, angles > 245.75° wrong.
- Not run: anything on a serial port; `drivers/lidar` (GoCV/TinyGo).

## API notes (ease of use, composability)

- Good: one small interface for very different sensors; matrix output plugs straight into `x/math` (e.g. [[filters]] SLAM takes a `matTypes.Matrix`); motor optional; context-driven lifetime.
- Friction: (1) row order and units differ from the SLAM consumer — SLAM expects row 0 angles in **radians**, row 1 distances in **metres**, fixed `numRays` (`x/math/filter/slam/slam.go:376-395`) while lidars give row 0 mm, row 1 deg, variable N; an adapter (bin into fixed rays, swap rows, convert units) is missing. (2) `OnRead` hands out a view into a buffer that the reader goroutine overwrites — consumers must copy inside the callback; document or double-buffer. (3) No per-point intensity, timestamp or validity; `0` distance means both "invalid" (vl53l0x) and "no return". (4) `Configure(init bool)` ignores `init` and silently starts goroutines; `New` takes a positional `(targetPoints, maxPoints)` pair — an options struct would read better. (5) Duplicated motor/assembly code in ld06/xwpftb — factor a shared `scanAssembler` (points in, scans out) so new lidars only write a frame parser. (6) `drivers/lidar/device_factory.go` is unused (main duplicates the switch, `main.go:148-172`).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Make host node build on darwin (darwin serial + move display/config under `!tinygo`) — drivers/lidar/main.go:1; x/devices/serial_stub.go #p2
- [ ] Add a lidar→SLAM adapter (angular binning to fixed rays, rad/m, row swap) with a test — x/devices/lidar (new) ; x/math/filter/slam/slam.go:376-395 #p2
- [ ] Delete unused DeviceFactory or use it in main — drivers/lidar/device_factory.go #p3

## Inferred objectives (proposals)

- **Proposal:** one tested LD06 path end-to-end: golden capture → decoder test → `LIDARReading` over dndm → SLAM adapter → occupancy grid in [[grid-maps]]. This is the shortest route to "lidar integrated" (W8).
- **Proposal:** a `lidar/scan` helper package: assembler, angular binning, polar→Cartesian, range/intensity filtering (median from [[filters]]), so lidars and the ToF array share it.
- **Proposal:** keep the RP2040 (or ESP32) as a pure UART→dndm bridge that publishes raw packets or binned scans; do decoding on the host when CPU is tight.

Workstream: [[W8-lidar]] · Related: [[device-io]], [[sensors]], [[filters]], [[grid-maps]], [[dndm-and-protobuf]], [[build-targets]], [[apps]]
