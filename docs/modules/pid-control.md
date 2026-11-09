---
type: module
group: "Control and planning"
status: tested
tags: [easyrobot, control, pid]
workstream: [W3, W8]
packages: [x/math/control/pid]
test_result: pass
---
# PID control (x/math/control/pid)

**Purpose:** Scalar and vector PID controllers for speed and position loops. Users:
- lidar spin-rate control (`x/devices/lidar/ld06`, `xwpftb`) → [[lidar]]
- DC motor speed (`x/devices/motor`) → [[actuators]]
- the path-tracking controller in [[motion-and-gait]]

**Packages:** `x/math/control/pid`:
- `pid1d.go` (39 LOC): `PID1D`
- `pid.go` (49 LOC): `PID` over `vec.Vector`
- tests: `pid_test.go` (176 LOC)

No build tags.

## Public API

- `New1D(p, i, d, min, max) PID1D`: value type. Set `Input` and `Target`, call `Update(dt)`, read `Output`. `Reset()` syncs `lastInput` and clears the integral. Gains are public fields; limits are private and fixed at construction.
- `New(p, i, d, min, max vec.Vector) PID`: element-wise version. Panics (`panic(-1)`) on length mismatch.
- Composition: embed one `PID1D` per loop. Chained updates return `*PID1D`.

## Algorithms

<!-- vault:algorithms h=173f05437365 -->

Counts: 1 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (1 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[pid-controller]] | tested | pass | Positional PID with derivative on measurement and integral clamping: `I += Ki·e·dt` clamped to… |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Deps: `x/math` (`Clamp`), `x/math/vec`. float32, no CGO, no reflection.
- `PID1D` is allocation-free and fine for TinyGo/MCU loops.
- `PID.Update` allocates four vector clones per call (`pid.go:41-45`). Avoid it in tight embedded loops.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/control/pid` (2026-11-09): ok, 8 tests.
- Tests cover: term values, component-wise operation, integral clamping, reset, and a panic on mismatch.

## API notes (ease of use, composability)

- The small value type with public fields is easy to embed and inspect.
- Derivative on measurement avoids setpoint kick, but the first `Update` without `Reset()` kicks from `lastInput = 0`. Document this, or initialise on the first call.
- Integral and output share the same `[min, max]`, so there is no separate anti-windup bound.
- No output-rate limit, no derivative filter, no feed-forward term. Each user re-implements these; the lidar driver clamps duty itself.
- `dt = 0` divides by zero (`pid1d.go:35`).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Optional derivative low-pass and separate integral limits — pid/pid1d.go:7 #p3

## Inferred objectives (proposals)

- **Proposal:** keep `PID1D` as the one MCU loop primitive and add an optional D-filter and feed-forward term for servo and motor loops on ESP32.
