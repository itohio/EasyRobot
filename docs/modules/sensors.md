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

**Purpose:** device drivers for sensors on the generic bus interfaces of [[device-io]] (`devices.I2C`, `devices.SPI`, `devices.Pin`, `devices.Serial`), usable from Linux (Raspberry Pi) and, in principle, TinyGo. Covers an optical-flow mouse sensor, AS7341/AS7343 spectral sensors, the CR30 colorimeter (serial), the MPU6050 IMU, the VL53L0X ToF sensor and a multi-VL53L0X "array lidar". None has tests; no hardware was used for this review.

**Packages** (LOC from `wc -l`, HEAD abb5c9cc; no build tags in any file):

| Package | Files (LOC) | Bus |
|---|---|---|
| `x/devices/adns3080` | adns3080.go 277 | SPI + CS/reset pins |
| `x/devices/as734x` | device.go 167, as7341.go 228, as7343.go 126, flicker_measure.go 157, types.go 119, bus.go 64, registers.go 56, bank.go 46, smux.go 24; calib.md 684, SPD_GUIDE.md 344 (docs only) | I²C 0x39 |
| `x/devices/cr30` | device.go 499, packet.go 378, DESIGN.md 165 | Serial 19200 8N1 |
| `x/devices/mpu6050` | mpu6050.go 166 | I²C 0x68 |
| `x/devices/vl53l0x` | vl53l0x.go 399, tuning.go 24 | I²C 0x29 |
| `x/devices/lidar/vl53l0x` (package `array`) | array.go 243, DESIGN.md 116 | via `tca9548a.Router` |
| `x/devices` | spectrometer.go 39 (`Spectrometer` interface) | — |

## Public API

- `devices.Spectrometer` (spectrometer.go:17-38): `NumWavelengths`, `Wavelengths(dst vec)`, `Measure(ctx, dst mat)`, `WaitMeasurement(ctx, dst mat)`; 1-row dst → SPD, 2-row → wavelengths + SPD. Implemented only by `cr30.Device` (compile-time assertion cr30/device.go:20); consumed by `cmd/spectrometer` and `cmd/cr30`. `as734x` does **not** implement it.
- `adns3080.New(spi, cs, reset).Configure(init)`, `ReadMotion`, `FrameCapture`, `Read8/Write8`.
- `as734x.New(i2c, addr).Configure(Config)`, `Read() RawMeasurement`, `Variant()`, `MeasureFlickerFrequency(dur, rate)`; `DefaultConfig()`; `Gain.Multiplier()`.
- `cr30.New(serial)`, `Connect`, `Handshake`, `DeviceInfo`, `Measure`, `WaitMeasurement`, plus exported `Packet`, `PacketBuilder`, `PacketParser`.
- `mpu6050.New(i2c, addr).Configure()`, `ReadAccelerometer/ReadGyroscope() (*{X,Y,Z int16})`, `ReadTemperature() float32 °C`, `Connected()`.
- `vl53l0x.New(i2c, addr).Configure(init)`, `ReadRangeSingle`, `StartContinuous/StopContinuous/ReadRangeContinuous`, `SetSignalRateLimit`, `Calibrate`, `Connected`.
- `array.New(ctx, Config{Router, Angles, SensorAddresses})` → `lidar.Device` (Configure/OnRead/Read/GetMin/MaxAngle/GetPointCount/Close); verified by a compile-time assertion in the probe. Composition: `devices.I2C` → `tca9548a.Router.Channel(i)` → `vl53l0x.Device` → 2×N matrix. Family context, other lidars (LD06, XWPFTB) and integration plans: [[lidar]], [[W8-lidar]].

## Algorithms

<!-- vault:algorithms h=55a1a400b8a3 -->

Counts: 0 tested · 0 implemented · 4 partial · 0 stub · 2 unverified (6 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[adns3080-optical-flow-driver]] | partial | none | partial (discards all received bytes) |
| [[as734x-flicker-detection]] | partial | none | partial (decoder and frequency estimator wrong per probe) |
| [[vl53l0x-ranging-driver]] | partial | none | partial (no SPAD init/calibration/timing budget) |
| [[vl53l0x-tof-array-scan]] | partial | none | Unsynchronised buffer, sensors ≥ 8 alias 0–7 |
| [[as734x-smux-and-spectral-read]] | unverified | none | unverified (integration-time overflow found) |
| [[cr30-serial-packet-protocol]] | unverified | none | unverified (no device; self-consistent per probe; receive checksum unchecked) |

<!-- /vault:algorithms -->

### MPU6050 IMU driver (trivial; kept here)

`x/devices/mpu6050/mpu6050.go`: `Configure` (:83-110) writes PWR_MGMT_1=0x00 (wake, internal 8 MHz oscillator), SMPLRT_DIV=7, ACCEL_CONFIG=0 (±2 g), GYRO_CONFIG=0 (±250 °/s), CONFIG=0x06 (DLPF ≈5 Hz [INFERENCE]). Reads are 6-byte big-endian bursts from 0x3B/0x43 (:113-138), temperature `raw/340+36.53` (:141-150). Register map, WHO_AM_I 0x68 and temperature formula match the MPU-6000/6050 register map as recalled [INFERENCE]. With DLPF on, gyro output rate is 1 kHz so 1 kHz/(1+7)=125 Hz (comment :89 correct).
- Probe (fake I²C, 2026-11-09): config writes `6B=00 19=07 1C=00 1B=00 1A=06`; accel bytes `40 00 C0 00 FF FF` → {16384, −16384, −1}; gyro `7F FF 80 00 00 01` → {32767, −32768, 1}; temp `F2 30` → 26.130 °C; `Connected()` true with WHO_AM_I 0x68.
- Gaps: raw counts only — no LSB/g (16384) or LSB/(°/s) (131) scaling helpers, no range setters, no bias calibration, no self-test (SelfTest regs :15-18 unused), no FIFO, no fusion (complementary/Madgwick would belong in [[filters]]); `Configure` does not check WHO_AM_I; PLL clock source (PWR_MGMT_1=0x01) recommended over internal oscillator [INFERENCE]; 5 Hz DLPF adds ≈19 ms lag [INFERENCE] — too slow for legged balance (W3). Allocation per read (:114, :128, :142, :163); `Connected` fails for MPU6500/9250 clones (WHO_AM_I 0x70/0x71 [INFERENCE]).

## Build targets and dependencies

- Pure Go, no CGO; only std + `x/devices`, `x/math/mat`, `x/math/vec`, `x/devices/tca9548a`. No reflection, no float64 in these packages.
- TinyGo suitability [INFERENCE, TinyGo not run]: mpu6050/vl53l0x/adns3080 are small but allocate per read; as734x allocates per read and uses `time.Now()` timestamps; cr30 allocates per packet and uses `context`, `time.After`, `strings`; the array lidar starts a goroutine with `time.Ticker` and `sync.Mutex` (fine on TinyGo with scheduler) and allocates a view per scan. Board-tag variants (rp2040/esp32/sam&&xiao) exist only in [[device-io]] bus shims, none here.
- Apps: `cmd/cr30` and `cmd/spectrometer` use cr30 via `devices.NewSerialWithConfig` (linux/windows only, baseline F2) and import GoCV → `go build -o /dev/null ./cmd/cr30` fails on darwin (opencv4 pkg-config missing, 2026-11-09).

## Tests and evidence

All with `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly`, 2026-11-09, packages `./x/devices/{adns3080,as734x,cr30,mpu6050,vl53l0x,lidar/vl53l0x}` + `./x/devices`:
- `go build` → ok; `go vet` → ok; `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build` → ok.
- `go test -count=1` → `[no test files]` for all seven.
- `go build -o /dev/null ./cmd/cr30` → fail (gocv: opencv4 not found).
- Probe `~/src/tmp/easyrobot/devices/sensors/` (own go.mod, `go run .`; `ascopy/` holds verbatim copies of unexported as734x functions): results quoted in each algorithm note and the MPU6050 section above. `go run -race .` → no race reported (array scan; single run).
- Hardware/serial not touched; all runtime behaviour on real sensors is unverified.

## API notes (ease of use, composability)

- Constructors are uniform (`New(bus, addr)` with 0 → default address) but configuration is not: `Configure(init bool)` (adns3080, vl53l0x, array), `Configure()` (mpu6050), `Configure(Config)` (as734x), `Connect()` (cr30).
- Return styles differ: pointer-allocating results (`*AccelerometerData`, `*MotionData`), value structs (`RawMeasurement`), destination matrices (`Spectrometer`, `lidar.Device`). Destination-style APIs type-assert to the concrete `mat.Matrix`/`vec.Vector` and panic on other implementations (cr30/device.go:218, 318; lidar/vl53l0x/array.go:161-162).
- No physical-unit outputs except VL53L0X (mm) and MPU temperature (°C); no shared sensor interface (IMU, range, spectral) beyond `Spectrometer` and `lidar.Device`.
- Errors mostly use `devices.Err*` sentinels; cr30 and as734x define their own.
- Drivers are not goroutine-safe; only the array lidar has a mutex (and still races, see its note).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Add fake-bus unit tests for every sensor package (none exist) — x/devices/mpu6050/mpu6050.go:83 #p2
- [ ] MPU6050: scaling to g and °/s, range setters, gyro bias calibration, WHO_AM_I check in `Configure` — x/devices/mpu6050/mpu6050.go:83 #p2
- [ ] MPU6050: select PLL clock and a faster DLPF for balance control (W3) — x/devices/mpu6050/mpu6050.go:105 #p2
- [ ] Remove per-read allocations (mpu6050 read8/bursts, vl53l0x read8, as734x buffers) for TinyGo — x/devices/mpu6050/mpu6050.go:114 #p3
- [ ] Unify `Configure` signatures / result styles across sensor drivers — x/devices/mpu6050/mpu6050.go:83 #p3

## Inferred objectives (proposals)

- **Proposal:** the VL53L0X array is a cheap short-range "lidar" for obstacle maps in [[W8-lidar]]; making it trustworthy needs the VL53L0X calibration work above and an XSHUT/readdress path to exceed 8 sensors.
- **Proposal:** AS734x + CR30 + [[colorscience]] form a spectral-measurement chain (CR30 as reference instrument to calibrate AS734x per `calib.md`); the calibration API described in `calib.md` is the intended next step and AS734x should implement `devices.Spectrometer`.
- **Proposal:** MPU6050 + ADNS3080 are odometry/attitude inputs for legged or wheeled robots ([[W3-legged-ik]], [[W4-k230-feature-stream]] recorder); they need unit scaling and a fusion filter in [[filters]] before use.
- **Proposal:** a common `IMU` / `RangeSensor` interface in [[device-io]] would let [[pipeline-and-chain]] / [[dndm-and-protobuf]] publish sensor data uniformly.

## Links

[[device-io]] · [[lidar]] · [[W8-lidar]] · [[colorscience]] · [[filters]] · [[build-targets]] · [[packages]]
