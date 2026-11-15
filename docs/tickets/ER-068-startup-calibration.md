---
id: ER-068
title: "Startup calibration of the motion sensor at rest — x/math/nav/calib"
type: ticket
status: active
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, navigation, tufty]
created: 2026-11-13
updated: 2026-11-13
---
# ER-068 — Startup calibration of the motion sensor at rest — x/math/nav/calib

[[tickets/index|Ticket index]] · [[navigation]] · [[startup-calibration]]

**Brick:** the badge's first page calibrates its motion sensor at rest and shows how good it is. The user lays the badge flat or stands it on a side and keeps it still; a ring fills while it measures; six strips show each axis's spread and drift; at the end a forecast says what drift to expect in motion. The badge app draws the page; this brick is the Calibrator behind it: fed one sample at a time, it finds the face that is down, waits until the sensor is still, averages, restarts when moved, and returns the gyro bias, the accelerometer's readings at rest, the noise, and the forecast; and a way to give the result to `strapdown.Nav`.

## Goal and evidence

- Pure Go, float32, no I/O, no allocation per sample; one value per calibrator.
- `Progress()`, `State()` (waiting, measuring, moved, done), `Result()`; live per-axis mean and spread (standard deviation, exponentially weighted) for both sensors, the stillness thresholds and the accelerometer's rest target.
- Forecast for the Done meter: heading drift (°/min) and position stray (cm/min), each split into systematic drift and noise random walk, with an acceptable line; the model documented, checked against a simulation of the strapdown path.
- `strapdown.Nav.Seed(gyroBias)` (the accelerometer's noise was in the sketch; seeding it changed nothing measurable).
- Hardware offset registers: not used (see the note).

## Acceptance evidence required later

- [x] The face found flat and on each side (six faces): `TestFindsTheFaceThatIsDown`.
- [x] Gyro bias and the accelerometer's rest readings recovered within tolerance from synthetic biased, noisy samples; the noise measured: `TestRecoversBiasOffsetsAndNoise` (5e-4 rad/s, 0.01 m/s², 15 %).
- [x] Motion restarts the measurement (moved, then waiting, then measuring from 0): `TestMovingRestartsTheMeasurement`, and slow motions too: `TestSlowMotionsRestartTheMeasurement`.
- [x] Zero allocations per sample: `TestUpdateDoesNotAllocate`, 0 allocs in the benchmark, none in calib under TinyGo.
- [x] Seeding: a seeded Nav keeps the calibration's gyro bias through its first rest: `TestSeedKeepsACalibrationsGyroBias`.
- [x] Forecast within a factor of two of what the strapdown simulation shows at the same noise: heading 1.2 times, position 1.05 times (0.75–1.5 asserted).
- [x] Accepted: six faces, gyro bias within 0.00017 rad/s, a nudge restarts, 0 allocations, heading forecast 0.208 against 0.196 °/min measured, position about 3.3 against 2.5 cm/min. The first-turn stray was already 0.0 mm in r10 without calibration. Seeding widened the after-turn strokes (22 of 36 within 10 % against 26, over-reads up to 222 %; failures 7 → 5), so the badge uses the Calibrator for the page and the forecast only, without seeding, until strapdown r11 (seeding may not make any after-turn probe worse).
- [ ] On the badge : the page runs the Calibrator; a hand check.

## Effort uncertainty and revision trigger

Size guess: **S**. Revisit if the user wants the accelerometer's offsets themselves (a several-pose routine: one pose cannot tell an offset from the table's tilt).

## Implementation evidence

- `x/math/nav/calib/calib.go` (373 lines) and `calib_test.go` (500): `go test -race ./x/math/nav/calib/` pass, 15 tests, 2026-11-13. Red first: the tests were written before the package; `Seed`'s test before `Seed`.
- `strapdown.Nav.Seed(gyroBias)`: the first rest keeps the seed and the rest after it starts its running averages at their longest (the first try left them at 0.5 s and the seed's benefit went: 0.024 rad against 0.013 expected). On a noisy gyro, held in motion after a 1.5 s first rest: 0.013 rad of attitude error after 20 s seeded with a 10 s calibration, 0.036 rad unseeded. The accelerometer's noise is not seeded (no measurable effect; a mutant showed it).
- Forecast checked against simulation: the heading's systematic drift 0.48 °/min against 0.40 °/min rms of the bias 40 calibrations left; the position 3.33 cm/min against 3.18 cm rms of 20 minutes of strapdown play (15 strokes of 1 s, 3 s at rest between). The model's two fixes came from these checks: the halves' wander counted only past the noise (summed, it read 1.8 times), and strapdown's 0.3 s to see the rest added to each motion (without it, 0.64 times). Strapdown finds no rest after 1 s strokes with 2 s rests (the leftover speed waits for `MaxGlide`); the play model leaves 3 s.
- Mutants: 28 of 29 of calib.go fail a test (the survivor is the forecast's noise term, 16 % of the total); 2 of 2 of `Seed`.
- Host: 10 ns per `Update`, 0 allocs; Calibrator 216 bytes (212 on tufty2040).
- TinyGo 0.43 tufty2040 (a badge build: MPU6050 → Calibrator → seeded strapdown and Mahony): 97,512 B flash, no allocation in calib; the strapdown-only build 94,296 B (94,264 before `Seed`).
- Notes: [[startup-calibration]], [[strapdown-inertial-navigation]] (Seeding), [[navigation]]; `x/math/SPEC.md` (strapdown, which was missing, and calib).
