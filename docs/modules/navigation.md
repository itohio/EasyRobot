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

**Packages:** `x/math/nav/pdr` (pdr.go 163, pdr_test.go 389) and `x/math/nav/strapdown` (strapdown.go 970, strapdown_test.go 1482); no build tags.

## Public API

- `pdr.NewDetector(pdr.DefaultDetectorConfig(rateHz)) (Detector, error)`; `Detector.Update(a [3]float32) bool`; `Steps()`.
- `pdr.Heading(q [4]float32, axis [3]float32) (heading, ok)`: the horizontal direction of a body axis from an attitude quaternion; any mounting.
- `pdr.Track{StepLength}`; `Step(heading)`; fields `X, Y, Steps`.
- Composition (badge: walking axis = IMU −Z): `mpu6050.Read` → `ahrs.MahonyAHRS.Update(dt, &in)` → `pdr.Heading([4]float32(mahony.Output()), [3]float32{0, 0, -1})` → `Track.Step(heading)` when `Detector.Update(accel)` is true. The detector does not need the filter; any heading source works.
- `strapdown.New(strapdown.DefaultConfig()) (Nav, error)`; `Nav.Update(dt, q, accel, gyro)` with `q = [4]float32(mahony.Output())` and the MPU6050's m/s² and rad/s; `Reset()` returns to the origin. After every sample the `Nav` is the badge's whole motion state, one value for every consumer (path, paddle, water, fire, boids): `Q` (the attitude used: the filter's until the first rest, then the Nav's own, gyro-carried and levelled at rest), `P`, `V` (m, m/s, world, z up), `PB`, `VB` (the same in the badge frame: each sample's displacement rotated into that sample's badge frame and summed, so a tilt in place adds nothing; Pong's paddle follows `PB[1]`), `A` (world linear acceleration, gravity and bias removed, every sample), `F` (specific force, badge frame: gravity plus the badge's own acceleration; water and fire), `Still`. Gentle and slow strokes are tracked in every direction. Start at rest: the first rest teaches the biases, and nothing moves before it. Tilting in place does not move the path (fpv-gate's 60 tilt cases with an MPU6050's noise and offsets up to past the datasheet's: 0.0 mm). A turn's offset error is taken out of the motion at its rest, and a stroke made while turning that never shows itself jumps into place at the rest after it. Every stroke of 5 cm or more, and every 3 cm stroke of 1 s or less, registers. Known failures (fpv-gate ER-067 §r10, 144 runs; r11's targets): 3 cm over 1.5 s 5 times unregistered and 3 times read the wrong way (−109 %, −736 %, −781 %); 3 cm in 2 s with no turn ended at −39 % once; slow 2 s strokes jump back at their late rest (10 cm reads 48 % in one of three draws). fpv-gate measured these as mostly design, not the sensor. Mahony's integral term no longer matters after the first rest.

## Algorithms

<!-- vault:algorithms h=d40d0b5ac88d -->

Counts: 2 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[pedestrian-dead-reckoning]] | tested | pass | steps from \|a\| + heading of a chosen body axis + track; synthetic-signal tests, no hardware yet |
| [[strapdown-inertial-navigation]] | tested | pass | strapdown translation from an IMU: the Nav carries its own attitude on the gyro after the first rest (levelled at rest, gyro bias learned at rest), rotates the accelerometer's force into the world, removes gravity and integrates to velocity and position; zero-velocity updates at rest in 3D; a rotation in place moves nothing while it lasts and its rest takes up any translation it carried; every turned motion's rest takes the turned offsets' error out of the whole motion; bias learning in the world and the sensor frame; the Nav is the badge's whole motion state (Q, P, V, PB, VB, A, F); float32, no alloc per sample |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, float32, `errors`; value types, no allocation per sample. `pdr` and `strapdown` use `chewxy/math32`. TinyGo 0.43 `-target=tufty2040`:

- with mpu6050 + Mahony + pdr: allocations only at construction (2026-11-12);
- with mpu6050 + Mahony + strapdown (`motiontufty`): one allocation in `strapdown`, the 736-byte `Nav` from `New`; none per sample; 94,264 B flash (2026-11-13).

## Tests and evidence

`go test -race -count=1 ./x/math/nav/...` pass, 2026-11-13; details in [[pedestrian-dead-reckoning]] and [[strapdown-inertial-navigation]].

## TODOs

None at module level.
