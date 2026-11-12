---
type: module
group: "Control and planning"
status: tested
tags: [easyrobot, navigation, dead-reckoning]
workstream:
packages: [x/math/nav/pdr]
test_result: pass
---
# Navigation (x/math/nav)

**Purpose:** dead reckoning from body-worn or vehicle sensors: position from motion events and a heading. First piece: pedestrian dead reckoning for the Tufty badge's walking path (MPU6050 → [[mahony-ahrs]] → steps → track).

**Packages:** `x/math/nav/pdr` (pdr.go 163, pdr_test.go 352; no build tags).

## Public API

- `pdr.NewDetector(pdr.DefaultDetectorConfig(rateHz)) (Detector, error)`; `Detector.Update(a [3]float32) bool`; `Steps()`.
- `pdr.Heading(q [4]float32, axis [3]float32) (heading, ok)`: the horizontal direction of a body axis from an attitude quaternion; any mounting.
- `pdr.Track{StepLength}`; `Step(heading)`; fields `X, Y, Steps`.
- Composition (badge: walking axis = IMU −Z): `mpu6050.Read` → `ahrs.MahonyAHRS.Update(dt, &in)` → `pdr.Heading([4]float32(mahony.Output()), [3]float32{0, 0, -1})` → `Track.Step(heading)` when `Detector.Update(accel)` is true. The detector does not need the filter; any heading source works.

## Algorithms

<!-- vault:algorithms h=5953661b1411 -->

Counts: 1 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (1 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[pedestrian-dead-reckoning]] | tested | pass | steps from \|a\| + heading of a chosen body axis + track; synthetic-signal tests, no hardware yet |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, float32 (`chewxy/math32`), `errors`; value types, no allocation per sample. TinyGo 0.43 `-target=tufty2040` with mpu6050 + Mahony: allocations only at construction (2026-11-12).

## Tests and evidence

`go test -race -count=1 ./x/math/nav/pdr/` pass, 2026-11-12; details in [[pedestrian-dead-reckoning]].

## TODOs

None at module level.
