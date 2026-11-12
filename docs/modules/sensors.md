---
type: module
group: "Devices and apps"
status: unverified
tags: [easyrobot, devices, sensors]
workstream: [W8, W3, W4, W1]
packages: [x/devices/adns3080, x/devices/as734x, x/devices/cr30, x/devices/mpu6050, x/devices/vl53l0x, x/devices/lidar/vl53l0x, x/devices (spectrometer.go)]
test_result: none
---
# Sensors (optical flow, spectral, colorimeter, IMU, ToF)

**Purpose:** device drivers for sensors on the generic bus interfaces of [[device-io]] (`devices.I2C`, `devices.SPI`, `devices.Pin`, `devices.Serial`), usable from Linux (Raspberry Pi) and, in principle, TinyGo. Covers an optical-flow mouse sensor, AS7341/AS7343 spectral sensors, the CR30 colorimeter (serial), the MPU6050 IMU, the VL53L0X ToF sensor and a multi-VL53L0X "array lidar". Only `mpu6050` has tests (fake I²C bus, 2026-11-12, [[mpu6050-imu-driver]]); no hardware was used for this review.

**Packages** (LOC from `wc -l`, HEAD abb5c9cc; no build tags in any file):

| Package | Files (LOC) | Bus |
|---|---|---|
| `x/devices/adns3080` | adns3080.go 277 | SPI + CS/reset pins |
| `x/devices/as734x` | device.go 167, as7341.go 228, as7343.go 126, flicker_measure.go 157, types.go 119, bus.go 64, registers.go 56, bank.go 46, smux.go 24; calib.md 684, SPD_GUIDE.md 344 (docs only) | I²C 0x39 |
| `x/devices/cr30` | device.go 499, packet.go 378, DESIGN.md 165 | Serial 19200 8N1 |
| `x/devices/mpu6050` | mpu6050.go 251, mpu6050_test.go 280 | I²C 0x68 |
| `x/devices/vl53l0x` | vl53l0x.go 399, tuning.go 24 | I²C 0x29 |
| `x/devices/lidar/vl53l0x` (package `array`) | array.go 243, DESIGN.md 116 | via `tca9548a.Router` |
| `x/devices` | spectrometer.go 39 (`Spectrometer` interface) | — |

## Public API

- `devices.Spectrometer` (spectrometer.go:17-38): `NumWavelengths`, `Wavelengths(dst vec)`, `Measure(ctx, dst mat)`, `WaitMeasurement(ctx, dst mat)`; 1-row dst → SPD, 2-row → wavelengths + SPD. Implemented only by `cr30.Device` (compile-time assertion cr30/device.go:20); consumed by `cmd/spectrometer` and `cmd/cr30`. `as734x` does **not** implement it.
- `adns3080.New(spi, cs, reset).Configure(init)`, `ReadMotion`, `FrameCapture`, `Read8/Write8`.
- `as734x.New(i2c, addr).Configure(Config)`, `Read() RawMeasurement`, `Variant()`, `MeasureFlickerFrequency(dur, rate)`; `DefaultConfig()`; `Gain.Multiplier()`.
- `cr30.New(serial)`, `Connect`, `Handshake`, `DeviceInfo`, `Measure`, `WaitMeasurement`, plus exported `Packet`, `PacketBuilder`, `PacketParser`.
- `mpu6050.New(i2c, addr)`, `Probe() (id, ok, err)`, `Configure(Config{SampleRateDiv, DLPF, Accel, Gyro, DataReadyInterrupt})`, `ReadRaw(*Raw)`, `Read(*Sample)` (m/s², rad/s, °C), `DataReady()`, `Config.Scale(Raw)` — [[mpu6050-imu-driver]].
- `vl53l0x.New(i2c, addr).Configure(init)`, `ReadRangeSingle`, `StartContinuous/StopContinuous/ReadRangeContinuous`, `SetSignalRateLimit`, `Calibrate`, `Connected`.
- `array.New(ctx, Config{Router, Angles, SensorAddresses})` → `lidar.Device` (Configure/OnRead/Read/GetMin/MaxAngle/GetPointCount/Close); verified by a compile-time assertion in the probe. Composition: `devices.I2C` → `tca9548a.Router.Channel(i)` → `vl53l0x.Device` → 2×N matrix. Family context, other lidars (LD06, XWPFTB) and integration plans: [[lidar]], [[W8-lidar]].

## Algorithms

<!-- vault:algorithms h=bd827e092245 -->

Counts: 1 tested · 0 implemented · 4 partial · 0 stub · 2 unverified (7 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[mpu6050-imu-driver]] | tested | pass | one-burst reads, rate/DLPF/ranges, SI units, data-ready IRQ, no alloc; fake-bus tests only |
| [[adns3080-optical-flow-driver]] | partial | none | partial (discards all received bytes) |
| [[as734x-flicker-detection]] | partial | none | partial (decoder and frequency estimator wrong per probe) |
| [[vl53l0x-ranging-driver]] | partial | none | partial (no SPAD init/calibration/timing budget) |
| [[vl53l0x-tof-array-scan]] | partial | none | Unsynchronised buffer, sensors ≥ 8 alias 0–7 |
| [[as734x-smux-and-spectral-read]] | unverified | none | unverified (integration-time overflow found) |
| [[cr30-serial-packet-protocol]] | unverified | none | unverified (no device; self-consistent per probe; receive checksum unchecked) |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, no CGO; only std + `x/devices`, `x/math/mat`, `x/math/vec`, `x/devices/tca9548a`. No reflection, no float64 in these packages.
- TinyGo suitability [INFERENCE, TinyGo not run except mpu6050]: vl53l0x/adns3080 are small but allocate per read; as734x allocates per read and uses `time.Now()` timestamps; cr30 allocates per packet and uses `context`, `time.After`, `strings`; the array lidar starts a goroutine with `time.Ticker` and `sync.Mutex` (fine on TinyGo with scheduler) and allocates a view per scan. Board-tag variants (rp2040/esp32/sam&&xiao) exist only in [[device-io]] bus shims, none here. mpu6050 builds for `tufty2040` with no allocation per read ([[mpu6050-imu-driver]]).
- Apps: `cmd/cr30` and `cmd/spectrometer` use cr30 via `devices.NewSerialWithConfig` (linux/windows only, baseline F2) and import GoCV → `go build -o /dev/null ./cmd/cr30` fails on darwin (opencv4 pkg-config missing, 2026-11-09).

## Tests and evidence

All with `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly`, 2026-11-09, packages `./x/devices/{adns3080,as734x,cr30,mpu6050,vl53l0x,lidar/vl53l0x}` + `./x/devices`:
- `go build` → ok; `go vet` → ok; `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build` → ok.
- `go test -count=1` → `[no test files]` for all seven (2026-11-09; `mpu6050` has tests since 2026-11-12, see [[mpu6050-imu-driver]]).
- `go build -o /dev/null ./cmd/cr30` → fail (gocv: opencv4 not found).
- Probe `~/src/tmp/easyrobot/devices/sensors/` (own go.mod, `go run .`; `ascopy/` holds verbatim copies of unexported as734x functions): results quoted in each algorithm note (its MPU6050 results describe the driver before the 2026-11-12 rewrite). `go run -race .` → no race reported (array scan; single run).
- Hardware/serial not touched; all runtime behaviour on real sensors is unverified.

## API notes (ease of use, composability)

- Constructors are uniform (`New(bus, addr)` with 0 → default address) but configuration is not: `Configure(init bool)` (adns3080, vl53l0x, array), `Configure(Config)` (as734x, mpu6050), `Connect()` (cr30).
- Return styles differ: pointer-allocating results (`*MotionData`), value structs (`RawMeasurement`), caller-owned destinations (`mpu6050.Read(*Sample)`), destination matrices (`Spectrometer`, `lidar.Device`). Destination-style APIs type-assert to the concrete `mat.Matrix`/`vec.Vector` and panic on other implementations (cr30/device.go:218, 318; lidar/vl53l0x/array.go:161-162).
- Physical-unit outputs: VL53L0X (mm), MPU6050 (m/s², rad/s, °C); no shared sensor interface (IMU, range, spectral) beyond `Spectrometer` and `lidar.Device`.
- Errors mostly use `devices.Err*` sentinels; cr30 and as734x define their own.
- Drivers are not goroutine-safe; only the array lidar has a mutex (and still races, see its note).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Add fake-bus unit tests for the other sensor packages (mpu6050 has them) — x/devices/vl53l0x/vl53l0x.go:112 #p2
- [ ] Remove per-read allocations (vl53l0x read8, as734x buffers) for TinyGo — x/devices/vl53l0x/vl53l0x.go:391 #p3
- [ ] Unify `Configure` signatures / result styles across sensor drivers — x/devices/adns3080/adns3080.go:87 #p3

## Inferred objectives (proposals)

- **Proposal:** the VL53L0X array is a cheap short-range "lidar" for obstacle maps in [[W8-lidar]]; making it trustworthy needs the VL53L0X calibration work above and an XSHUT/readdress path to exceed 8 sensors.
- **Proposal:** AS734x + CR30 + [[colorscience]] form a spectral-measurement chain (CR30 as reference instrument to calibrate AS734x per `calib.md`); the calibration API described in `calib.md` is the intended next step and AS734x should implement `devices.Spectrometer`.
- **Proposal:** MPU6050 + ADNS3080 are odometry/attitude inputs for legged or wheeled robots ([[W3-legged-ik]], [[W4-k230-feature-stream]] recorder). The MPU6050 now outputs SI units; fusion is in [[filters]]; the ADNS3080 still needs unit scaling.
- **Proposal:** a common `IMU` / `RangeSensor` interface in [[device-io]] would let [[pipeline-and-chain]] / [[dndm-and-protobuf]] publish sensor data uniformly.

## Links

[[device-io]] · [[lidar]] · [[W8-lidar]] · [[colorscience]] · [[filters]] · [[build-targets]] · [[packages]]
