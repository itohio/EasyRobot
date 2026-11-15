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

[[tickets/index|Ticket index]] · [[navigation]] · [[strapdown-inertial-navigation]]

**Brick:** the badge's Motion page must "track PATH, meaning translation and rotation … actual motion of the badge in real 3d space". Rotation comes from the attitude filter (ahrs Mahony); this brick adds translation:

1. rotate the accelerometer reading into the world by an attitude and subtract gravity (the filter's quaternion until the first rest, then the Nav's own: gyro-carried, levelled at rest; a filter leans into any acceleration, so slow horizontal strokes cancel themselves on its attitude);
2. integrate the result into velocity and position;
3. hold velocity at zero while the badge is at rest (zero-velocity updates, from a stillness detector);
4. optionally damp velocity while it moves.

The badge UI consumes it. It also drives Pong: moving the board up or down must move the paddle up or down with the same speed and motion. So vertical displacement and speed must be right over 0.5–1 s strokes, including strokes without a pause between them.

Requirement (2026-11-13): the badge needs one shared motion state (orientation, position through real 3D space, and the specific force: gravity plus the badge's own acceleration) for boids, paddle, path, water and fire; and tilting the badge must not show motion when there is none. So the Nav exposes, per sample, the world linear acceleration and the body-frame specific force beside position, and a tilt in place must not move the path. Correction: the paddle uses vertical position in the badge frame and water and fire the force in the badge frame, so the Nav also gives displacement and velocity in the body frame.

## Goal and evidence

From the brief:

- **Arithmetic:** fixed point or float32, stated. This is **float32**, like the badge's Mahony filter and `pdr`; it is a few dozen operations per sample.
- **Allocation:** zero per step.
- **Synthetic tests:** a 10 cm move and back returns near the start under ZUPT; a pure rotation gives no translation.

## Dependencies and scope

Package `x/math/nav/strapdown`, beside `pdr` in [[navigation]].

**API:** `New(Config) (Nav, error)`, `Nav.Update(dt, q, accel, gyro)`, `Reset`; after every sample the `Nav` is the whole motion state: `Q` (the attitude used: the filter's until the first rest, then the Nav's own), `P`, `V` (world), `PB`, `VB` (body frame: each sample's world displacement rotated into that sample's sensor frame and summed, and the velocity in the sensor frame now; Pong's paddle follows `PB[1]`), `A` (world linear acceleration, gravity and bias removed, every sample), `F` (the specific force given, sensor frame), `Bias`, `Still`. Proposed to the badge UI. Inputs:

- `q`: the filter's quaternion, `[w,x,y,z]`, sensor to world, z up;
- `accel`: the accelerometer's specific force in m/s² (as `mpu6050.Read` gives it);
- `gyro`: rad/s.

**Not in scope:**

- magnetometer heading: without one, yaw is relative and drifts;
- a Kalman filter: the bias is learned at rest instead;
- drawing the path: that belongs to the badge UI.

No hardware or flashing in this ticket. The badge program is a build of MPU6050 → Mahony → strapdown, for the badge UI.

## Acceptance evidence required later

- [x] A 10 cm move and back returns near the start under ZUPT (`TestMoveAndBackReturnsUnderZUPT`).
  - Conditions: badge held at a tilt; accelerometer bias of about 0.1 m/s², a 1 % scale error, noise, and a 0.05° attitude error.
  - Result: the middle rest is at 10.0 cm and the end within 6 mm of the start.
  - A plain double integration of the same samples ends 3.2 m away.
- [x] A pure rotation gives no translation (`TestPureRotationGivesNoTranslation`): four axes up to 3 rad/s, |P| and |V| ≤ 8e-6.
- [x] Tilting in place does not move the path (`TestTiltOnlyGivesNoTranslation`).
  - Setup: 60 cases: 15°, 30°, 45°, 60°, 90° over 1, 0.5 and 0.2 s, and back; upright about the screen normal and the screen's x, flat about x and y; Mahony KP 2, KI 0.005 after 5 s; an MPU6050's noise (accelerometer 0.03 m/s² Gaussian, gyro 0.002 rad/s), gyro bias 0.005 rad/s; offsets of 0.09 m/s², an uncalibrated MPU6050's (0.32, −0.48, 0.40) and 1.5× that; 30 seeds each.
  - Bound: |P| and the paddle's |ΔPB[1]| under 1 mm, at rest after (the required bar is 1 cm). Measured: 0.0 mm in all runs; 30 seeds from six ranges at all three offset levels and at gyro bias 0.02 rad/s (37,800 runs): median, p90 and max 0.0 in every case; the independent harness 60/60 at 0.0. r8 strayed up to 3 cm here, r5 58 cm (4/60 under 1 cm), r3 2–75 cm.
  - How: a motion that begins with the gyro alone is a rotation and moves nothing while it lasts; a translation it shows is taken up from a shadow at once, and one it never shows is taken up at its rest if it stands out of the noise (4σ), of what unread offsets could make, and of 1 cm. Every turned motion's rest reads what is left as the turned offsets' error δb and takes it out of the whole motion (m2·δb from P), which is why a tilt moves nothing and a 5–20 cm stroke right after a turn neither overshoots nor runs away (3 cm strokes: below).
- [ ] Every stroke registers (r9 left up to 7 of 36 unmoved and a stroke right after a turn read 149–405 % and then drifted 1.3 m in 30 s). **Not yet**: every stroke of 5 cm or more and every 3 cm stroke of 1 s or less registers, but of the 144 independent stroke runs, 3 cm over 1.5 s did not register 5 times (wrist wobbling at 0.1 rad/s once, at 1 rad/s three times, right after a 45° turn once) and read the wrong way 3 times (−109 % wobbling; −736 % and −781 % right after a 45° turn). With no turn, 3 cm in 2 s ended at −39 % in one draw, and slow 2 s strokes jump back at their late rest (one of three draws each; 10 cm reads 106–110 % live, then 48 %; 5 cm, 95 % then 54 %). Decision: r10 goes to main as an interim with these limits; r11 must reach 0 unregistered, 0 wrong-way and no snap-back on the independent sets.
  - Wrist wobbling (`TestStrokesTrackWhileTheWristTurns`, exact sensor model): all 36 within 15 % at 0.3 rad/s, 30 % at 1 rad/s (0.9 % and 0.3 % short on average), those of 10 cm or more half there while they move; with an MPU6050's noise and offsets (`TestStrokesWhileTheWristTurnsWithNoise`, the independent set, ten seeds) 72–140 %.
  - Right after a turn: exact attitude within 1 %, 3–20 cm (`TestStrokeRightAfterATurn`); with noise, 5–20 cm with no pause or 0.25 s, 59–153 % over 180 runs, at rest, PB within 1.3 mm (`TestStrokesAfterATurnWithNoise`); the independent runaway case: 50–150 %, then 30 s held without a wake, under 1 mm (`TestTurnThenStrokeComesToRest`, red against r9: 306–335 %, 11 wakes, 1.7 m).
  - 3 cm right after a turn, measured: −250 % to 170 % over my 60 runs, and what a misread rest leaves drifts up to 20 cm over 10 s held, with 24 wakes over the 60 runs (`TestGentleStrokeAfterATurnStaysBounded` bounds it at 30 cm). Review: mostly design, the registration threshold and how a turn's offset error is attributed; the sensor's part is a standard deviation of about 23 % at an MPU6050's noise. Gentle strokes with no turn (3–5 cm over 1–2 s) leave up to 2 cm/s of noise, so their rest waits for `MaxGlide`.
  - After a turn and a rest the gentlest strokes track within 10 % (`TestGentleStrokeAfterATurnTracks`).
- [x] The motion state per sample (`TestLinearAccelerationOnEverySample`): `A` is the true world linear acceleration to 2 mm/s² on every sample with a biased sensor, at rest from the first rest on, and through a stroke including the samples before the Nav sees it.
- [x] The body frame (`TestBodyFrameMotion`, exact attitude, upright): 10 cm up gives `PB` (0, 0.10, 0) within 2 mm; tilting 40° in place leaves it there (P rotated by the new attitude would swing 6.4 cm); 5 cm along the tilted +Y gives (0, 0.15, 0) within 3 mm, and `VB` reads the stroke's speed along +Y to 5 mm/s once the Nav sees the motion.
- [x] Pong: vertical strokes tracked in displacement and speed (`TestVerticalMovesTrackedForPong`).
  - Setup: badge upright; Mahony supplies the attitude; biased accelerometer with 0.02 m/s² rms noise (about an MPU6050's at 44 Hz).
  - Motion: six 10 cm strokes of 0.5 s without a pause, then 1 s strokes up and down.
  - Result over three noise draws: every stroke's displacement within 1 cm (measured 2.0–3.4 mm) and the vertical speed within 3 cm/s (4.4–7.2 mm/s) of up to 37.5 cm/s. The badge ends at rest within 3 cm (0.03–1.6 cm, the noise's random walk).
- [x] Gentle and slow strokes move the paddle and the path, in every direction (review: the stillness measures were ORed and chained across a smooth stroke, so rest came mid-stroke at up to 372 mm/s, and slow strokes tracked 0 % to −24 %; accepted for Pong vertically; r7's fix was vertical-only, horizontal strokes 2–68 % through Mahony).
  - `TestGentleAndSlowStrokesTrack` (the review grid): 3–30 cm over 0.3–2 s vertically, ideal sensor through Mahony: every stroke within 10 % (worst 96.2 %, 3 cm in 2 s), at rest after.
  - `TestSlowHorizontalStrokesTrack` (exact attitude): 5 cm in 2 s, 3 cm in 1.5 s, 10 and 30 cm in 2 s along x within 1 % (measured 0.2 %), at rest after. The slowest, 3 cm in 2 s, ends at 88 % (its wake-up at `StillSpeed`, two axes of noise). Through Mahony with a noisy sensor, 41 of 42 of the grid's strokes horizontally within 10 % (worst 111 %), 36 of 42 vertically.
  - `TestGentleStrokesTrackWithNoise`: 3 and 5 cm over 1 and 1.2 s with accelerometer bias, 0.03 m/s² noise and gyro bias 0.02 rad/s, five draws: within 10 % + 3 mm.
  - `TestSlowStrokesWithExactAttitude`: within 1 %; 3 cm in 2 s within 5 % (96.2 %).
  - Waving without a pause has nothing to correct it: up to 4 mm off after 2 s, 2 cm after 4 s, 11 cm after 9.5 s (documented).
- [x] ZUPT and stillness, pinned by tests:
  - rest needs a quiet gyro, no acceleration and a speed under `StillSpeed` at once for StillTime (vertically always, horizontally over the first `MaxGlide` of a motion); a horizontal glide after a push is at rest within `MaxGlide`, back where the push ended (`TestZUPTAfterPush`); after `MaxGlide` of looking still, any glide is (`TestLongGlideCountsAsRest`);
  - the bias is learned at rest (exact mean over the first rest, from the rest only), follows a slow drift, and the start of a motion does not leak into it;
  - nothing is integrated before the first rest;
  - a bias learned while the attitude was still wrong is learned again, also when found while moving, and the stretch's drift is taken back (`TestStaleBiasIsRelearned`, `TestStaleBiasFoundWhileMoving`); a 0.4 m/s² bias step at rest is relearned within 2 s, under 10 cm off (`TestBiasStepAtRestIsRelearned`; without it, 5 m in 5 s); a glide's rest allows for what is left in its acceleration (`TestGlideTakesBackWhatABiasBuilt`);
  - the attitude stays level at rest: a gyro bias step during a long rest is levelled back out (`TestAttitudeStaysLevelAtRest`); fed from power-on while Mahony still converges, the first rest waits for it (`TestPowerOnWhileTheFilterSettles`, red against r7: 1.6 m at 3 s).
- [x] The damping option, invalid samples ignored, config refused, `Reset` keeps the biases and the attitude, 0 allocations per `Update`.
- [x] TinyGo 0.43 tufty2040 build (MPU6050 + Mahony + strapdown): `-print-allocs=strapdown` lists only the 736-byte `Nav` that `New` returns (TinyGo heap-allocates values over 256 bytes); `Update` allocates nothing.
- [ ] On the badge with a hand: strokes, gentle and slow ones included, move the paddle the badge's distance; moved 10 cm and back, the path returns near the start; tilted in place, it stays put. At rest it is already checked (previous version): 30 s read 0 0 0 mm with `Still` throughout.

## Effort uncertainty and revision trigger

Estimate: **S**. Revisit if the badge trace shows drift the synthetic tests don't model, for example gyro noise well above an MPU6050's (the attitude is the gyro's between rests) or accelerometer offsets that change between rests.

## Implementation evidence

- `x/math/nav/strapdown/strapdown.go` and `strapdown_test.go`: `go test -race ./x/math/nav/strapdown/` pass, 38 tests (with ER-068's `Seed`), 2026-11-13.
- The first tests failed to build without the package; they were written alongside the code. The gentle and slow stroke tests were red first against an earlier version; the tilt-only test against r3; the power-on test against r7; the 48-case tilt test against r8; the turn-then-stroke test against r9.
- 51 of 52 mutants of r10's code (the turn's error terms and corrections, the rotation's registration, the turned rest's checks, the glide, the stale rule, the offsets) fail a test, one is a constant; r9's 31, r8's 29, r7's 23 and r3's 24 were run against those versions.
- Host: 27 ns per `Update`, 0 allocs.
- TinyGo: the badge build is 94,296 B flash in all (94,264 before `Seed`, ER-068); one 736-byte allocation in `New`.
- Tufty 2040 (measured on the badge, 2026-11-13, an earlier version): 87–88 µs per `Update`, 0 heap allocations; r10 not yet timed on the badge.
