---
id: ER-067
title: "Strapdown integrator with ZUPT for the badge's 3D motion path — x/math/nav/strapdown"
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
# ER-067 — Strapdown integrator with ZUPT for the badge's 3D motion path — x/math/nav/strapdown

[[tickets/index|Ticket index]] · [[navigation]] · [[strapdown-inertial-navigation]] · Brief: Leader m-a196c06b6546 (owner, decisions.md 2026-11-13)

**Brick:** the badge's Motion page must "track PATH, meaning translation and rotation … actual motion of the badge in real 3d space". Rotation comes from the attitude filter (ahrs Mahony); this brick adds translation:

1. rotate the accelerometer reading into the world by the filter's quaternion and subtract gravity;
2. integrate the result into velocity and position;
3. hold velocity at zero while the badge is at rest (zero-velocity updates, from a stillness detector);
4. optionally damp velocity while it moves.

tinygui consumes it. It also drives Pong (owner, Leader m-4635cbfcd77b): "move the board up or down and the paddle move up or down with the same speed and motion". So vertical displacement and speed must be right over 0.5–1 s strokes, including strokes without a pause between them.

## Goal and evidence

From Leader's brief:

- **Arithmetic:** fixed point or float32, stated. This is **float32**, like the badge's Mahony filter and `pdr`; it is a few dozen operations per sample.
- **Allocation:** zero per step.
- **Synthetic tests:** a 10 cm move and back returns near the start under ZUPT; a pure rotation gives no translation.

## Dependencies and scope

Package `x/math/nav/strapdown`, beside `pdr` in [[navigation]].

**API:** `New(Config) (Nav, error)`, `Nav.Update(dt, q, accel, gyro)`, `Reset`; state in `P`, `V`, `A`, `Bias`, `Still`. Inputs:

- `q`: the filter's quaternion, `[w,x,y,z]`, sensor to world, z up;
- `accel`: the accelerometer's specific force in m/s² (as `mpu6050.Read` gives it);
- `gyro`: rad/s.

**Not in scope:**

- magnetometer heading: without one, yaw is relative and drifts;
- a Kalman filter: the bias is learned at rest instead;
- drawing the path: that is tinygui's.

No hardware or flashing by easyrobot. `tmp/easyrobot/simalgo/motiontufty` is the badge program (MPU6050 → Mahony → strapdown), for tinygui.

## Acceptance evidence required later

- [x] A 10 cm move and back returns near the start under ZUPT (`TestMoveAndBackReturnsUnderZUPT`).
  - Conditions: badge held at a tilt; accelerometer bias of about 0.1 m/s², a 1 % scale error, noise, and a 0.05° attitude error.
  - Result: the middle rest is at 10.2 cm and the end within 0.3 cm of the start.
  - A plain double integration of the same samples ends 3.2 m away.
- [x] A pure rotation gives no translation (`TestPureRotationGivesNoTranslation`): four axes up to 3 rad/s, |P| and |V| ≤ 8e-6.
  - With Mahony supplying the attitude, tilting 45° and back strays at most 3.3 cm, from the filter's own error during the turn, and ends at 0.05 cm (`TestTiltAndBackWithMahony`).
- [x] Pong: vertical strokes tracked in displacement and speed (`TestVerticalMovesTrackedForPong`).
  - Setup: badge upright; Mahony supplies the attitude; biased accelerometer with 0.02 m/s² rms noise (about an MPU6050's at 44 Hz).
  - Motion: six 10 cm strokes of 0.5 s without a pause, then 1 s strokes up and down.
  - Result over three noise draws: every stroke's displacement within 1 cm (measured 2.0–3.6 mm) and the vertical speed within 3 cm/s (4.4–6.8 mm/s) of up to 37.5 cm/s. The badge ends at rest within 3 cm (0.05–1.6 cm, the noise's random walk).
- [x] Pong: gentle and slow strokes move the paddle too (fpv-gate `verification/ER-067-strapdown.md`: the stillness measures were ORed and chained across a smooth stroke, so rest came mid-stroke at up to 372 mm/s, and slow strokes tracked 0 % to −24 %).
  - `TestGentleAndSlowStrokesTrack` (fpv-gate's grid): 3–30 cm over 0.3–2 s, ideal sensor through Mahony: every stroke within 10 % (worst 96.2 %, 3 cm in 2 s), at rest after. Red first: the previous version failed 16 of the 42.
  - `TestGentleStrokesTrackWithNoise`: 3 and 5 cm over 1 and 1.2 s with accelerometer bias, 0.03 m/s² noise and gyro bias 0.02 rad/s, five draws: within 10 % + 3 mm (−4.0 to +2.9 mm).
  - `TestSlowStrokesWithExactAttitude`: within 1 %; 3 cm in 2 s within 5 % (96.2 %).
  - Waving without a pause has nothing to correct it: up to 7 mm off after 2 s, 2 cm after 4 s, 8 cm after 10 s (documented).
- [x] ZUPT and stillness, pinned by tests:
  - rest needs a quiet gyro, no acceleration and a vertical speed under `StillSpeed` at once for StillTime; a glide after a push is at rest StillTime later, back where the push ended (`TestZUPTAfterPush`); after `MaxGlide` of looking still, any glide is (`TestLongGlideCountsAsRest`);
  - the bias is learned at rest (exact mean over the first rest, from the rest only), follows a slow drift, and the start of a motion does not leak into it;
  - nothing is integrated before the first rest;
  - a bias learned while the attitude was still wrong is learned again, also when found while moving, and the stretch's drift is taken back (`TestStaleBiasIsRelearned`, `TestStaleBiasFoundWhileMoving`).
- [x] The damping option, invalid samples ignored, config refused, `Reset` keeps the bias, 0 allocations per `Update`.
- [x] TinyGo 0.43 tufty2040 build (`motiontufty`, MPU6050 + Mahony + strapdown): `-print-allocs=strapdown` empty.
- [ ] On the badge with a hand (the owner): strokes, gentle and slow ones included, move the paddle the badge's distance; moved 10 cm and back, the path returns near the start; tilted in place, it stays put. At rest it is already checked (tinygui, previous version): 30 s read 0 0 0 mm with `Still` throughout.

## Effort uncertainty and revision trigger

easyrobot's guess: **S**. Revisit if the badge trace shows drift the synthetic tests don't model, for example a gyro bias that Mahony's integral term has not yet removed.

## Implementation evidence

- `x/math/nav/strapdown/strapdown.go` and `strapdown_test.go`: `go test -race ./x/math/nav/strapdown/` pass, 21 tests, 2026-11-13 (`strapdown-test-race-r3.txt`).
- The first tests failed to build without the package (`ER-SIMALGO-logs/strapdown-fail-before.txt`); they were written alongside the code. The gentle and slow stroke tests were red first against the previous version (`strapdown-gentle-fail-before.txt`).
- 24 mutants all fail a test (`strapdown-mutants-r3.txt`). They cover the measures chained without the speed gate, the speed judged in 3D, each part of the wake-up (threshold, older shadow, its distance, its bias), the take-back, the glide fallback, stale recalibration and its take-back, both limits of the bias average, gravity, trapezoidal velocity, the gyro check, the first-rest mean and its samples, damping and config validation.
- Host: 7–8 ns per `Update`, 0 allocs (`strapdown-bench-host-r3.txt`).
- TinyGo: `motiontufty` is 87,480 B flash in all; `strapdown` allocates nothing (`strapdown-tinygo-tufty2040-r3.txt`).
- Tufty 2040 (tinygui, 2026-11-13, previous version): 87–88 µs per `Update`, 0 heap allocations (`docs/evidence/er-timing/motiontufty-serial.log`).
