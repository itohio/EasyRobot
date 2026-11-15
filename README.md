# EasyRobot

EasyRobot is a Go library and toolset for robots and embedded sensing: device drivers, signal processing, filters and navigation, vision, small neural networks, and the plumbing between them. The same packages build for a desktop or a Linux board with Go and, where they avoid the heap and cgo, for microcontrollers with [TinyGo](https://tinygo.org): develop and test an algorithm on the desktop, then run the same code on the robot.

Module: `github.com/itohio/EasyRobot` (Go 1.25).

## Module map

| Path | What is there |
|---|---|
| `x/math` | numerics and algorithms: `vec`, `mat`, `tensor`; `filter` (Mahony AHRS, Kalman and extended Kalman, FIR, IIR, moving average, median, Gaussian, Savitzky–Golay, PLL, AGC, SLAM); `nav`; `control` (PID); `dsp`; `interpolation`; `graph` (search and planning); `grid` (occupancy maps); `fluid` (a 2D FLIP liquid); `swarm` (boids); `physics`; `gps` (geodesy); `colorscience`; `learn`; `nn`; `protocol`; `shape`; `primitive` (kernels) |
| `x/math/nav` | motion from an IMU: `strapdown` (dead reckoning with rest detection), `eskf` (error-state Kalman filter: attitude, velocity, position and both biases estimated jointly), `calib` (calibration at rest and from several poses), `pdr` (step detection) |
| `x/devices` | sensor and actuator drivers over a small bus abstraction (I2C, SPI, serial, pins): IMU (`mpu6050`), optical flow (`adns3080`), time of flight (`vl53l0x`), spectral (`as734x`, `cr30`), lidar, GPS (`nmea`), RC link (`crsf`), servos and PWM (`servo`, `pca9685`), I/O expanders and muxes (`pcf8574`, `tca9548a`), motors, encoders |
| `x/vision` | detection (including int8 YOLOX decoding), feature extraction and matching, stereo, visual odometry, transforms; mostly on OpenCV through GoCV |
| `x/knowledge` | a local knowledge engine: chunking, indexing and retrieval over Markdown and Go sources |
| `x/chain`, `x/iter`, `x/workers` | composition of processing stages over typed buffers, iteration and worker helpers |
| `x/marshaller` | encoding and I/O for frames, audio, tensors and storage (protobuf, GoCV, TFLite, V4L2, flash) |
| `x/buffer`, `x/types` | bit buffers; numeric types and conversions |
| `drivers` | larger drivers: the Kendryte K210 KPU (`kpu`, with a model compiler and an emulator), M5Stack UnitV (`unitv`), camera, lidar, a manipulator arm |
| `cmd` | tools and demos: `kernelbench` (times the embedded kernels on a board or the host), `kmodelc` (K210 model compiler), `nnbench`, `qi8bench`, `calibrate`, `camframe`, `display`, `monitor`, `spectrometer`, `cr30`, `elrs-rx` (CRSF receiver firmware), `manipulator`, `audio` |
| `proto`, `types` | protobuf messages for data that crosses a process or device boundary, carried by [dndm](https://github.com/itohio/dndm) |
| `docs` | the documentation: architecture, one note per module and algorithm with status and evidence, tickets |

## Status

From the module notes in `docs/modules/` (`make docs-check` checks them against the code). *Tested*: covered by tests that pin its behaviour; *partial*: some parts tested, some gaps; *unverified*: present, not checked.

| Area | Status | Notes |
|---|---|---|
| Navigation (`x/math/nav`) | tested | strapdown, error-state Kalman filter, calibration and step detection, each with its own tests; zero allocation per sample, TinyGo-checked |
| PID control, simulation algorithms (FLIP liquid, boids, rigid 2D bodies) | tested | |
| Knowledge engine | implemented, tests pass | |
| Vectors and matrices, tensors, primitive kernels, interpolation, DSP core, learning, kinematics, motion and gait, grid maps, bit buffer, marshallers, iterators and workers, CRSF/ELRS, audio I/O | partial, tests pass | |
| Filters, colour science, GPS and geodesy, NMEA, lidar, modulation, line coding and FEC, neural networks, protocol parser | partial, some tests fail | see each module note |
| Graph search and planning | partial, a flaky test | |
| Vision, actuators, type conversion | partial, fail to build in some configurations | vision needs OpenCV; see the module notes |
| Device I/O | partial, some tests | |
| Apps, audio modem CLI, pipeline and chain, store and plugin | partial, no tests | |
| Camera I/O, TFLite, sensors | unverified | |

## Using it

```sh
go get github.com/itohio/EasyRobot@latest
```

```go
import (
	"github.com/itohio/EasyRobot/x/math/filter/ahrs"
	"github.com/itohio/EasyRobot/x/math/nav/strapdown"
)
```

The module currently resolves `github.com/itohio/dndm` from a sibling checkout (`replace github.com/itohio/dndm => ../dndm` in `go.mod`). Until that is a released version, a module that depends on EasyRobot needs the same: clone [dndm](https://github.com/itohio/dndm) next to it and add the same `replace` line to its own `go.mod`. Packages that don't import dndm (most of `x/math` and `x/devices`) build without it.

## Build and test

```sh
go build ./...
go test ./...            # or: make test (adds -race, which needs cgo)
make docs-check          # the docs' module and algorithm notes against the code
make help                # the other targets: protobuf generation, mocks, coverage, the docs, flashing
```

Most packages are pure Go. With `CGO_ENABLED=0`, `go build ./...` builds everything on linux/amd64 except the packages that need native libraries (below); on linux/arm, arm64 and riscv64 `x/iter/loop_experiments` also fails (it has amd64 assembly), and windows/amd64 and js/wasm have further gaps (see `docs/architecture/build-targets.md`). The native dependencies, through their modules:

| Needs | Packages |
|---|---|
| OpenCV 4 (`gocv.io/x/gocv`) | most of `x/vision`, `x/marshaller/gocv`, `x/math/tensor/gocv`, `cmd/calibrate`, `cmd/display`, `cmd/spectrometer`, `cmd/cr30`, `drivers/lidar` on the desktop |
| TensorFlow Lite C (`github.com/mattn/go-tflite`, Linux) | `x/marshaller/tflite` |
| V4L2 headers (`github.com/vladimirvivien/go4vl`, Linux) | `x/marshaller/v4l` |
| miniaudio through cgo (`github.com/gen2brain/malgo`) | `x/marshaller/audio`, `cmd/audio` |
| `github.com/tarm/serial` (through `dndm/network/serial`; cgo on darwin) | `cmd/manipulator` |

Without them, those packages are skipped or fail; the rest builds.

## TinyGo

The embedded paths allocate nothing after setup, checked with `tinygo build -print-allocs`. Build tags pick the per-target files (`tinygo`, `rp2040`, `esp32`, `sam && xiao`, `unitv`).

| Target | Board | In the tree | Status |
|---|---|---|---|
| RP2040: `tufty2040` and other boards | Pimoroni Tufty 2040 and other RP2040 boards (Cortex-M0+, no FPU) | `cmd/kernelbench` (IMU, filter, navigation, liquid and flocking kernels), `drivers/lidar`, `cmd/elrs-rx` | `kernelbench` builds for the Tufty 2040 (`tinygo flash`, see its README); the lidar and receiver firmware are not built routinely |
| K210: `unitv` (`drivers/unitv/target/unitv.json`) | M5Stack UnitV (Kendryte K210, RISC-V) | `drivers/unitv`, the KPU driver and model compiler | builds and runs on the board (`make unitv`) |
| ESP32 | ESP32 | `cmd/elrs-rx` | only the receiver's platform file; its PWM output is a stub |
| SAMD21: `xiao` | Seeed XIAO | `drivers/manipulator`, `x/devices/xiao`, `cmd/elrs-rx` | not built routinely |

```sh
tinygo build -target=tufty2040 -print-allocs=strapdown ./cmd/kernelbench   # what allocates
tinygo flash -target=tufty2040 ./cmd/kernelbench                      # timings over USB serial
make unitv                                                            # the K210 build
```

## Documentation

`docs/index.md` is the entry point: architecture notes, a note per module and algorithm with its status, tests and evidence, and the tickets. It is plain Markdown with wiki links; `make docs` regenerates its tables from the notes' front matter.

## License

See [LICENSE](LICENSE).
