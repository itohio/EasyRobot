---
type: module
group: "Control and planning"
status: tested
tags: [easyrobot, navigation, dead-reckoning]
workstream:
packages: [x/math/nav/pdr, x/math/nav/strapdown]
test_result: pass
---
# Navigation (x/math/nav)

**Purpose:** dead reckoning from body-worn or vehicle sensors: position from motion events and a heading, or from the accelerometer and an attitude. First pieces are for the Tufty badge:

- its walking path: MPU6050 → [[mahony-ahrs]] → steps → track;
- its 3D motion path: MPU6050 → Mahony → strapdown with zero-velocity updates.

**Packages:** `x/math/nav/pdr` (pdr.go 163, pdr_test.go 352) and `x/math/nav/strapdown` (strapdown.go 336, strapdown_test.go 695); no build tags.

## Public API

- `pdr.NewDetector(pdr.DefaultDetectorConfig(rateHz)) (Detector, error)`; `Detector.Update(a [3]float32) bool`; `Steps()`.
- `pdr.Heading(q [4]float32, axis [3]float32) (heading, ok)`: the horizontal direction of a body axis from an attitude quaternion; any mounting.
- `pdr.Track{StepLength}`; `Step(heading)`; fields `X, Y, Steps`.
- Composition (badge: walking axis = IMU −Z): `mpu6050.Read` → `ahrs.MahonyAHRS.Update(dt, &in)` → `pdr.Heading([4]float32(mahony.Output()), [3]float32{0, 0, -1})` → `Track.Step(heading)` when `Detector.Update(accel)` is true. The detector does not need the filter; any heading source works.
- `strapdown.New(strapdown.DefaultConfig()) (Nav, error)`; `Nav.Update(dt, q, accel, gyro)` with `q = [4]float32(mahony.Output())` and the MPU6050's m/s² and rad/s; read `Nav.P` (m, world, z up; Pong's paddle follows `P[2]`, gentle and slow strokes included) and `Nav.Still`; `Reset()` returns to the origin. Start at rest: the first rest teaches the bias, and nothing moves before it. Give Mahony an integral term or calibrate the gyro at rest: a gyro bias tilts the attitude, and the tilt leaks into the horizontal path.

## Algorithms

<!-- vault:algorithms h=455981d3de8e -->

Counts: 2 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[pedestrian-dead-reckoning]] | tested | pass | steps from \|a\| + heading of a chosen body axis + track; synthetic-signal tests, no hardware yet |
| [[strapdown-inertial-navigation]] | tested | pass | strapdown translation from an attitude quaternion + accelerometer: gravity removed, integrated to velocity and position, zero-velocity updates at rest with bias learning; float32, no alloc |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, float32, `errors`; value types, no allocation per sample. `pdr` uses `chewxy/math32`; `strapdown` needs no math library. TinyGo 0.43 `-target=tufty2040`:

- with mpu6050 + Mahony + pdr: allocations only at construction (2026-11-12);
- with mpu6050 + Mahony + strapdown (`motiontufty`): no allocation in `strapdown` (2026-11-13).

## Tests and evidence

`go test -race -count=1 ./x/math/nav/...` pass, 2026-11-13; details in [[pedestrian-dead-reckoning]] and [[strapdown-inertial-navigation]].

## TODOs

None at module level.
