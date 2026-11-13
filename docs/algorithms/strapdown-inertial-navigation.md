---
type: algorithm
status: tested
tags: [easyrobot, navigation, imu, tinygo]
workstream:
module: "[[navigation]]"
summary: "strapdown translation from an attitude quaternion + accelerometer: gravity removed, integrated to velocity and position, zero-velocity updates at rest with bias learning; float32, no alloc"
source: [x/math/nav/strapdown/strapdown.go]
tests: [x/math/nav/strapdown/strapdown_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race ./x/math/nav/strapdown pass (21 tests, 2026-11-13): gentle and slow Pong strokes (3–30 cm over 0.3–2 s, Mahony attitude) end within 10 %, worst 96.2 %; with noise and gyro bias 3–5 cm strokes within −4.0…+2.9 mm; with an exact attitude within 1 % (3 cm in 2 s: 96.2 %); 10 cm Pong strokes through Mahony with a noisy biased accelerometer within 3.6 mm and 6.8 mm/s; 10 cm move and back ends within 0.3 cm (plain integration: 3.2 m); pure rotation ≤ 8e-6 m; tilt 45° and back through Mahony strays ≤ 3.3 cm; 24 mutants all fail (ER-SIMALGO-logs/strapdown-mutants-r3.txt); 0 allocs, 7–8 ns per Update; tinygo 0.43 tufty2040 (motiontufty) no allocation in strapdown; badge (tinygui, previous version): 87–88 µs per Update, still at 0 0 0 mm for 30 s at rest"
---
# Strapdown integrator with zero-velocity updates (strapdown)

**What it does:** gives the translation half of a body's 3D path from an IMU:

- the accelerometer's specific force is rotated into the world by an attitude filter's quaternion;
- gravity is removed;
- the rest is integrated into velocity and position.

The attitude filter gives the rotation half. Built for the Tufty badge (ER-067): its Motion page shows the path, and Pong moves the paddle with the badge's vertical motion. It is generic for any IMU with an attitude estimate.

**Where:** `x/math/nav/strapdown/strapdown.go`:

- `Config`, `DefaultConfig` (:67-82), `Nav` (:89-108), `New` (:114), `Reset` (:126);
- `Update` (:138-271), `rest` (:274), `atRest` (:291), `shadow.step` (:310), `rotate` (:321).

**Inputs → outputs:** `New(Config)` returns a `Nav` at the origin. `Update(dt, q, f, w)` takes:

- `q`: the attitude, `[w,x,y,z]`, sensor to world, z up (e.g. `ahrs.MahonyAHRS.Output()`);
- `f`: the specific force, sensor frame, m/s², +g up at rest;
- `w`: the gyro rate, rad/s.

It updates:

- `P` (m) and `V` (m/s), world frame;
- `A`, the linear acceleration with the bias removed;
- `Bias`, the learned world bias;
- `Still`.

For Pong, the paddle follows `P[2]` (up) or `V[2]`.

**Method (per sample):**

1. **World acceleration:** r = R(q)·f − (0, 0, G).
2. **Measures:** the gyro is quiet (< `StillGyro`); no linear acceleration is left (|r − Bias| < `StillAccel`, "calm"); the specific force is steady (within `StillAccel` of its running average over `AvgTime`).
3. **First rest:** quiet and steady for `StillTime`, then `AvgTime` more; the bias is the plain mean of its samples. Nothing is integrated before it; the origin holds.
4. **Rest** (from motion): quiet, calm and the vertical speed under `StillSpeed`, all at once and without a break for `StillTime`.
   - Low acceleration alone is not rest: 3 cm in 2 s peaks at 0.043 m/s², under any usable `StillAccel`. In the previous version the measures were ORed and chained across a smooth stroke (the force steady at the acceleration peak, the acceleration nil at mid-stroke), so rest came mid-stroke at up to 372 mm/s (fpv-gate) and slow strokes tracked 0 % to −24 %.
   - Speed is judged vertically only: an attitude error θ puts g·sin θ into the horizontal acceleration (as much as a gentle stroke's) but only g·(1 − cos θ) into the vertical. Gated on |V| in 3D, the Mahony filter still converging after power-on kept waking the Nav and the path drifted 0.75 m sideways in the Pong test.
   - **Glides:** quiet, calm and steady for `MaxGlide` is rest whatever the speed. This bounds the drift after a wrong velocity.
5. **ZUPT:** on entering rest, velocity is zero, and the velocity left over is error that was there through the whole quiet stretch (its acceleration was nil): its distance over the stretch is taken back. A real stroke's tail has slowed to nothing and is kept. (The previous version snapped back to where the body began to look still; that cut slow strokes' tails and made the paddle jump.)
6. **At rest:** bias learning by a running average whose time constant grows from `AvgTime` to 4·`AvgTime` over the rest. A rally without pauses integrates whatever bias error it starts with: with a 0.5 s average the Pong test ended up to 3.3 cm off, with the growing one 1.6 cm.
7. **Leaving rest:** a rotation; a strong acceleration that isn't steady; or, while calm, a vertical shadow speed of `StillSpeed`/2.
   - Two shadows integrate r during the rest, each against the bias of its snapshot (one every `StillTime`). On waking, the older one, which began at least `StillTime` before, gives the velocity, the last acceleration and the distance to add, and the bias goes back to its snapshot (the first samples of a motion are learned into the bias before the motion is seen).
   - With an exact attitude, 3 cm in 2 s reaches the wake-up speed a sixth of the way in and ends at 96.2 %; waking at `StillSpeed` instead gives 81.9 %, dropping the shadow's distance 93.3 %, a shadow against the current bias 90.4 %.
8. **Stale bias:** quiet and steady with a strong acceleration (not calm) for `StillTime` + `AvgTime` is no hand but a bias gone stale (the attitude filter converging at power-on). The bias is learned again as a first rest, and the position goes back to where that stretch began.
9. **Motion:** `A` = r − Bias; velocity and position both by the trapezoid; velocity times (1 − `Damping`·dt). The samples are of a smooth motion: taking each acceleration for the whole step before it adds half a step of it to the velocity (1 cm/s on a 2 m/s² Pong stroke).

**Numbers:** float32, matching the badge's Mahony filter and `pdr`. One quaternion rotation and a few dozen operations per sample; no square root or trig. Non-finite samples and a non-positive dt are ignored.

**Limits:**

- Accelerometer noise random-walks the position: at about an MPU6050's 0.02 m/s² rms, a few millimetres per second of motion. Rests stop the velocity, not that offset. In the Pong test the end was up to 1.6 cm off vertically after 6 s of strokes; each stroke stays right.
- Waving without a pause has no rest to correct it (10 cm strokes of 0.5 s, 0.02 m/s² noise, five draws): off by up to 7 mm after 2 s, 2 cm after 4 s, 8 cm after 10 s.
- An attitude error θ leaks g·sin θ, 0.17 m/s² per degree, into the horizontal acceleration. A turn through a filter that lags or leads moves the path: through Mahony at KP 2 with an exact gyro, 45° and back strayed 3.3 cm; with a gyro bias of 0.02 rad/s and no integral term, 90° and back strayed up to 0.6 m sideways (fpv-gate), vertically within 1 mm. Give the filter an integral term (KI) or calibrate the gyro at rest; a real MPU6050's gyro bias is well above 0.02 rad/s.
- Gentle horizontal strokes are not tracked: horizontal speed isn't trusted (above), so a horizontal stroke under `StillAccel` stays at rest. A horizontal glide is stopped after `StillTime`.
- Sustained motion without rest drifts quadratically. `Damping` bounds it at the cost of short motions.
- No magnetometer: yaw, and with it the path's heading, is relative and drifts with the gyro bias.

## Completeness review

- Complete: no. Everything in ER-067 except moved-badge checks on the badge (strokes, move and back, tilt in place), which need the owner's hand.
- Tested (`strapdown_test.go`, 100 Hz synthetic sensors):
  - **Acceptance (Pong):**
    - `TestGentleAndSlowStrokesTrack` (fpv-gate's grid; ideal sensor, Mahony attitude, badge upright): every stroke of 3, 5, 7.5, 10, 15, 20 and 30 cm over 0.3, 0.5, 0.75, 1, 1.5 and 2 s ends within 10 % (worst 96.2 %, 3 cm in 2 s) and at rest. Red first: the previous version failed 16 of the 42 (0 % to −24 %; `ER-SIMALGO-logs/strapdown-gentle-fail-before.txt`).
    - `TestGentleStrokesTrackWithNoise`: 3 and 5 cm over 1 and 1.2 s, accelerometer bias and 0.03 m/s² noise, gyro bias 0.02 rad/s, five draws: within 10 % + 3 mm (−4.0 to +2.9 mm). The previous version moved −13 to +1 mm.
    - `TestVerticalMovesTrackedForPong` (bias, 0.02 m/s² noise, three draws): six 0.5 s strokes of 10 cm without a pause, then 1 s strokes; every stroke within 1 cm (measured 2.0–3.6 mm) and 3 cm/s (4.4–6.8 mm/s); ends still within 3 cm vertically (0.05–1.6 cm).
    - `TestMoveAndBackReturnsUnderZUPT` (tilted, bias, 1 % scale error, noise, 0.05° attitude error): the middle rest at 10.2 cm, the end within 0.3 cm; plain integration ends 3.2 m away.
    - `TestPureRotationGivesNoTranslation`: four axes, 1–3 rad/s; |P|, |V| ≤ 8.3e-6.
  - **With the badge's filter:** `TestTiltAndBackWithMahony`: 45° and back; strays ≤ 3.3 cm, ends 0.05 cm away, still.
  - **Integration and rest:**
    - `TestSlowStrokesWithExactAttitude`: 3 cm in 2 s within 5 % (96.2 %); 5 cm in 2 s, 3 cm in 1.5 s and 10 cm in 2 s within 1 %.
    - `TestZUPTAfterPush`: exact double-trapezoid values for a push; the glide after it is at rest StillTime + 1 sample later, back where the push ended (within 0.1 mm).
    - `TestLongGlideCountsAsRest`: a vertical glide at 5 cm/s is at rest after `MaxGlide`, back where it began.
    - `TestSlowStartIsNotLost`: a 1 s move ends 10 cm away to the millimetre.
    - `TestFirstRestLearnsTheBias` (within 1 %), `TestFirstBiasFromTheRestOnly`, `TestBiasNotLearnedFromTheStartOfAMotion`, `TestBiasFollowsASlowDriftAtRest` (lags < 0.01 m/s²), `TestStaleBiasIsRelearned`, `TestStaleBiasFoundWhileMoving` (back where the steady stretch began), `TestHoldsOriginUntilFirstRest`, `TestResetKeepsBias`.
  - **Options and API:** `TestDamping` (exact (1 − D·dt)ⁿ), `TestBadSamplesAreIgnored`, `TestNewRejectsConfig`, `TestUpdateDoesNotAllocate`.
  - The first tests failed to build without the package (`ER-SIMALGO-logs/strapdown-fail-before.txt`); they were written alongside the code.
- Correct: yes on these properties; 24 mutants fail a test (`ER-SIMALGO-logs/strapdown-mutants-r3.txt`), among them the measures chained without the speed gate, the speed judged in 3D, each part of the wake-up, the take-back, the glide fallback, the stale take-back and both limits of the bias average.
- Speed: 7–8 ns per `Update` on the host (Apple M5), 0 allocs (`strapdown-bench-host-r3.txt`). Tufty 2040 (tinygui, 2026-11-13, previous version): 87–88 µs per `Update` (Mahony 97 µs), 0 heap allocations, and 30 s at rest read 0 0 0 mm with `Still` throughout (`docs/evidence/er-timing/motiontufty-serial.log`).
- TinyGo 0.43 `-target=tufty2040` (`motiontufty`: MPU6050 → Mahony → strapdown): 87,480 B flash, `-print-allocs=strapdown` empty (`strapdown-tinygo-tufty2040-r3.txt`).

## TODOs

- [ ] Badge check with a hand (the owner, `motiontufty`): Pong strokes, gentle and slow ones included, move the paddle the badge's distance; 10 cm and back returns near the start; tilting in place stays put — x/math/nav/strapdown/strapdown.go:138 #p1 #ER-067

## Links

- Module: [[navigation]] · Related: [[pedestrian-dead-reckoning]], [[mahony-ahrs]], [[mpu6050-imu-driver]] · Ticket: [[ER-067-strapdown-motion-path]]
