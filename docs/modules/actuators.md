---
type: module
group: "Devices and apps"
status: partial
tags: [easyrobot, devices, actuators]
workstream: [W3, W6]
packages: [x/devices/servo, x/devices/pca9685, x/devices/xiao, x/devices/motor, x/devices/encoder, x/devices/pcf8574, x/devices/tca9548a]
test_result: build-fail
---
# Actuators, encoders and bus expanders

**Purpose:** Drive servos and DC motors (with encoder feedback) and extend I/O: hobby-servo arrays for legged robots / manipulators, H-bridge DC motors for wheeled robots, I2C PWM and GPIO expanders, I2C mux.

**Packages:**

| Package | Files (LOC) | Tags | Role |
|---|---|---|---|
| `x/devices/servo` | `types.go` 100, `array.go` 87, `array_tinygo.go` 60, `array_stub.go` 17, README 140 | `tinygo` / `!tinygo` | angle↔µs mapping, servo array |
| `x/devices/pca9685` | `pca9685.go` 146 | — | 16-ch I2C PWM |
| `x/devices/xiao` | `xiao_pwm.go` 289, stub 29, README 75 | `sam && xiao` / `!sam \|\| !xiao` | SAMD21 TCC `PWMDevice` |
| `x/devices/motor` | `motor.go` 366, `array.go` 128, `types.go` 74, README 265 | — | DC motor PID speed control |
| `x/devices/encoder` | `encoder.go` 193, `linux.go`, `stub.go`, `tinygo.go`, example | `linux` / `!tinygo && !linux` / `tinygo` | quadrature decoder |
| `x/devices/pcf8574` | `pcf8574.go` 84 | — | 8-bit I2C GPIO expander (shadow register, write-all) |
| `x/devices/tca9548a` | `tca9548a.go` 74, `router.go` 110, example test (tinygo) | — | I2C mux + virtual-bus router |

## Public API

- `servo.NewMotorConfig(WithPin, WithMicroseconds(min,max,default,maxAngle))`, `servo.NewServoArray(pwm devices.PWMDevice, []Motor)`, `ServoArray.Set([]float32 deg)`, `Stop()`.
- `pca9685.New(bus, addr)`, `SetFrequency(hz)`, `SetPWM(ch, duty, invert)`, `SetPWMRaw`.
- `xiao.NewPWMDevice() devices.PWMDevice`.
- `motor.New(pwm devices.PWMDevice, motor.Config{Type, Dir, PWM, PinA, PinB, Encoder, PIDGains, MaxOutput, SamplePeriod, MaxRPM})`, `SetSpeed(rpm)`, `Enable/Disable`; `motor.NewMotorArray`.
- `encoder.New(pinA, pinB, Config)`, `Configure()`, `Position()`, `RPM()`.
- `pcf8574.New(bus, addr)`, `SetPin/GetPin/WriteAll/ReadAll`; `tca9548a.NewRouter(bus, addr).Channel(n) devices.I2C`.

Composition: `PWMDevice` (xiao, linux sysfs) → `servo.ServoArray` / `motor.Motor`; `encoder.Device` → `motor.Motor`; `tca9548a.Router` → any I2C driver. Upstream, `drivers/manipulator` (XIAO) feeds `vaj` motion profiles ([[motion-and-gait]]) into `ServoArray` over dndm ([[apps]]).

## Algorithms

<!-- vault:algorithms h=aa799a6fbdf9 -->

Counts: 0 tested · 1 implemented · 4 partial · 0 stub · 1 unverified (6 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[tca9548a-i2c-mux-router]] | implemented | none | none (probe) |
| [[dc-motor-pid-speed-control]] | partial | build-fail | Ticker race, encoder dependency |
| [[pca9685-pwm-driver]] | partial | none | ON/OFF registers swapped |
| [[quadrature-encoder-decoding]] | partial | build-fail | Table correct, RPM gate 1000× off, F1 |
| [[servo-angle-mapping-and-array]] | partial | none | `Stop` deadlocks, TinyGo-only |
| [[xiao-samd21-tcc-pwm]] | unverified | none | Suspect period units |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, no CGO. Host: servo array not constructible off TinyGo (`array_stub.go:14-15`); motor and encoder fail on darwin and windows (F1). Linux: all build (cross-build 2026-11-09).
- TinyGo: xiao (`sam && xiao`) and servo (`tinygo`) target boards. `servo` builds under tinygo 0.43 since the `TinyGoPin` adapter ([[device-io]], 2026-11-12); `encoder`/`motor` fail (`encoder/tinygo.go` uses undefined `devices.PinConfig` and a non-existent `Pin.Configure`); `xiao` fails on `-target=xiao` and its `Channel` rejects the adapter ([[xiao-samd21-tcc-pwm]]). Motor/encoder use goroutines, `time.Ticker`, `sync`, float32 — TinyGo-friendly; PCA9685 allocates per write.

## Tests and evidence

- `go test -count=1 ./x/devices/...` (darwin, 2026-11-09): `encoder`, `motor` FAIL [build failed]; `servo`, `pca9685`, `pcf8574`, `tca9548a`, `xiao` no test files. `GOOS=windows` build: encoder fails (F1 also on windows). `GOOS=linux` build OK; vet flags `encoder/example_test.go:11`.
- Probe `~/src/tmp/easyrobot/devices/actprobe` (fake I2C): PCA9685 bytes `06 e8 03 00 00` for value 1000 (swapped), prescale 121 @50 Hz OK; TCA9548A select 0x08/0x20 + caching OK; servo 0/90/180/200° → 1000/1500/2000/2000 µs; `ServoArray.Stop` deadlock. Probe `encprobe`: quadrature table ∓99 for 100 Gray states; RPM gate needs 100 s.

## API notes (ease of use, composability)

- Good: everything takes interfaces; `ServoArray.Set([]float32)` is the right shape for IK output (one vector per tick); functional options for servo config.
- Friction for legged robots (W3): no joint calibration layer (offset, direction, limits, µs/deg) between IK angles and servo pulses; servo angles are 0…MaxAngle (no signed joint angles around a zero pose); PCA9685 is not a `PWMDevice`, so `ServoArray` cannot drive the 16-channel board most hexapods use; no synchronous "commit all channels" (PCA9685 auto-increment write of 16×4 bytes would give one I2C transaction per tick).
- Motor: `Config` mixes wiring, control gains and limits; `PIDGains` anonymous struct is awkward to fill; `motor.Config.Encoder` is a concrete `*encoder.Device` — an `interface{ RPM() float64 }` would allow other speed sources (hall, back-EMF, simulation).
- Lifecycle naming differs across drivers (`Configure(bool)`, `Configure()`, `Enable()`, `Start()`).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Make PCA9685 a `devices.PWMDevice` with `SetMicroseconds` and batch update — x/devices/pca9685/pca9685.go #p2
- [ ] Joint calibration layer for IK → servo (offset/sign/limits) — x/devices/servo/types.go #p2
- [ ] Accept an `RPM()` interface in motor config — x/devices/motor/types.go #p3
- [ ] Fake-bus/fake-pin unit tests for all actuator packages — x/devices/{servo,pca9685,motor,encoder,pcf8574,tca9548a} #p2

## Inferred objectives (proposals)

- **Proposal:** "servo bus" for hexapod/quadruped: `PCA9685 PWMDevice` + `servo.Joint` calibration + `ServoArray.Set` fed by the gait engine in [[motion-and-gait]] at a fixed tick (50 Hz). One host test per layer with fakes.
- **Proposal:** wheeled ESP32 rover (W6): fix encoder + motor, add feed-forward, expose `motor.Array` speeds to [[kinematics]] differential drive.

Related: [[device-io]], [[pid-control]], [[kinematics]], [[motion-and-gait]], [[W3-legged-ik]], [[apps]]
