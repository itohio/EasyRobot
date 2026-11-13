---
type: algorithm
status: tested
tags: [easyrobot, navigation, imu, calibration, tinygo]
workstream:
module: "[[navigation]]"
summary: "startup calibration of an IMU at rest: finds the face that is down, waits for stillness, averages for 3 s, restarts when moved; gyro bias, readings at rest, noise, live per-axis statistics and a forecast of heading and position drift checked against simulation; seeds strapdown's gyro bias"
source: [x/math/nav/calib/calib.go]
tests: [x/math/nav/calib/calib_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race ./x/math/nav/calib pass (15 tests, 2026-11-13): six faces found; gyro bias within 5e-4 rad/s, readings at rest within 0.01 m/s², noise within 15 % at an MPU6050's noise; a nudge, a slow lean and a slow turn restart the run; tremor and gyro jitter never start it; the heading forecast 1.2 times the bias 40 runs left, the position forecast 1.05 times a minute of strapdown play (strokes of 0.5–2 s: 1.2–1.75); a warming gyro over the heading line; 28 of 29 mutants fail a test (calib-tg3-mutants.txt); 0 allocs, 10 ns per Update; Calibrator 216 B (212 on tufty2040); tinygo 0.43 tufty2040 (calibtufty, with Mahony and strapdown) 97,512 B flash, no allocation in calib"
---
# Startup calibration at rest (calib)

**What it does:** calibrates a motion sensor (a 3-axis accelerometer and gyro) at rest, for the badge's first page (TG-3): the owner lays the badge flat or stands it on a side and keeps it still; a ring fills while it measures; six strips show each axis's spread and drift; at the end a forecast says what drift to expect in motion. Fed one sample per call, the Calibrator finds the face that is down, waits until the sensor is still, averages, starts again when moved, and gives the gyro bias, the accelerometer's readings at rest, the noise and the forecast. The gyro bias seeds [[strapdown-inertial-navigation]]. Generic for any IMU.

**Where:** `x/math/nav/calib/calib.go`:

- `StandardGravity` (:75), `seeRest` (:80), `ErrConfig` (:83), `State` (:86-107), `Config`, `DefaultConfig` (:110-128), `Stats` (:131), `Forecast` (:136), `Result` (:142), `Calibrator` (:151), `New` (:172), `Reset` (:183);
- `State`, `Progress`, `Accel`, `Gyro`, `Thresholds`, `Up`, `AccelRest`, `Result` (:189-231), `Update` (:237), `still` (:302), `begin` (:316), `restart` (:332), `result` (:339: the Result and the forecast).

**Inputs → outputs:** `New(DefaultConfig())` returns a Calibrator, waiting. `Update(dt, accel, gyro)` takes the accelerometer's specific force (m/s², +g up at rest) and the gyro rate (rad/s), sensor frame. After each call:

- `State()`: waiting, measuring, moved, done; `Progress()`: 0–1 while measuring, 1 at done, 0 otherwise;
- `Accel()`, `Gyro()`: per axis, the mean and the standard deviation over the last `Window` (exponentially weighted), in sensor units: the page's strips;
- `Thresholds()`: the stillness thresholds; `AccelRest()`: the accelerometer's target at rest, ±G on the axis that points up, 0 on the others; `Up()`: that axis and its sign;
- `Result()`, at done: `GyroBias`, `AccelOffset` (the readings at rest less `AccelRest`), `AccelNoise`, `GyroNoise` (standard deviations), `Forecast` (heading drift and noise in °/min, position drift and noise in cm/min, each with its acceptable line).

`Reset()` starts again (A on the page). **Until strapdown r11, the badge uses the Calibrator for the page and the forecast only:** Mahony and the Nav take the raw samples, and `strapdown.Nav.Seed` is not called. fpv-gate (`roster/verification/ER-TG3-calibrator.md`) found that seeding widens the strokes made right after a turn: 22 of 36 within 10 % against 26 unseeded, over-reads up to 222 %, though failures drop from 7 to 5. Leader's rule: seeding may not make any fpv-gate probe worse. r11 looks into it; then the app will subtract `GyroBias` from the gyro samples and call `Seed(GyroBias)`.

**Method:**

1. **Live statistics:** per axis, a running mean and variance; a plain one of all samples until they span `Window`, then exponentially weighted with time constant `Window`. (Started from the first sample alone, the mean was off by its noise, 0.05 m/s², and settled through the first measurement: its reference drifted by the threshold and measuring restarted at random.)
2. **Still:** every axis's spread under its threshold (0.06 m/s², 0.01 rad/s: twice and five times an MPU6050's noise at 100 Hz) and its mean within the threshold of where it was when it began to look still; never before the statistics span `Window`. The mean test catches what the spread misses: a slow lean (0.5 °/s: the gyro's 0.009 rad/s passes its threshold) moves the force's mean, and a slow turn on a swivelling desk (0.015 rad/s) the gyro's.
3. **Waiting → measuring** after `Window` still (from power-on: 2·`Window`, 1 s). The face that is down is the axis with the largest mean force then.
4. **Measuring:** a plain mean and variance (Welford) of every sample over `Measure` (3 s); the gyro's mean at half time is kept. Not still → **moved** for `MovedHold` (0.5 s, Progress 0), then waiting.
5. **Done:** the sums stay; `Result` reads them. The live statistics keep running.

**One pose cannot tell an offset from the surface's tilt:** on a table tilted by 1°, a horizontal axis reads its offset plus g·sin 1° (0.17 m/s², more than an MPU6050's typical offset); the up axis reads its offset plus the scale error at 1 g (±3 %: 0.29 m/s²). So `AccelOffset` is for display, not for correcting samples, and it is not seeded into strapdown, which learns the offsets across turns (r10: no stray on the first turn without calibration). The gyro bias, a rate at rest, has no such ambiguity. The sensor's hardware offset registers are not used: the result applies to any IMU, needs no bus writes, is testable on the host, and the registers would hold the same tilt-confounded figures.

**Forecast** (for the Done meter):

- **Heading:** the gyro bias left about the up axis turns the heading at a constant rate. The mean of N samples is off by σg/√N; the bias may wander, which half the change between the measurement's halves shows, but that holds the same noise, so it counts only where larger: HeadingDrift = max(σg/√N, |m₂ − m₁|/2)·60 s. (Added together, the two counted the noise twice: 1.8 times the bias the runs left.) HeadingNoise = σg·√(dt·60 s), the angle random walk after a minute.
- **Position:** strapdown holds the position at rest and integrates only while the badge moves. A minute of play is `MotionsPerMin` motions (15) of `MotionTime` (1 s), at rest between for long enough that strapdown takes the rest: it waits up to `MaxGlide` (2.5 s) after a stroke whose speed left over is above `StillSpeed`, and with 2 s rests between 1 s strokes at an MPU6050's noise it found none after the second stroke and strayed by metres. Strapdown integrates each motion and the `StillTime` (0.3 s) it takes to see the rest after it: T = `MotionTime` + 0.3 s (without it, the forecast read 0.64 of the stray). PosNoise = √(3M)·σa·√(dt·T³/3) (twice-integrated noise per axis per motion, adding as a random walk); PosDrift = M·½·(σa/√n)·T², n the samples of strapdown's longest rest average (4·0.5 s), the biases learned at the rests taken as adding up.
- **Lines:** 2 °/min of heading (tinygui's scale runs to 6) and 5 cm/min of position (to 10): about four and 1.5 times what an MPU6050 at 100 Hz gives (0.48 °/min; 3.3 cm/min).
- **Checked against simulation** (the tests below): the heading drift is 1.2 times the rms of the bias left by 40 calibrations; the position total is 1.05 times the rms stray of 20 minutes of strapdown play (exact attitude, the MPU6050's noise and offsets). Other plays: 20 × 0.5 s 1.18, 12 × 1.5 s 1.49, 10 × 2 s 1.75: the drift term overstates longer strokes, where the rests' biases do not all add.

## Completeness review

- Complete: no. The badge has not run it: tinygui draws the page (TG-3) and the owner's hand check is to come; fpv-gate's check of the seeded harness is open.
- Tested (`calib_test.go`, 100 Hz synthetic MPU6050: offsets (0.04, −0.06, 0.05) m/s², gyro bias (5, −3, 4) mrad/s, noise 0.03 m/s² and 0.002 rad/s):
  - **Acceptance:** `TestFindsTheFaceThatIsDown` (six faces, done in 2·`Window` + `Measure`), `TestRecoversBiasOffsetsAndNoise` (gyro bias within 5e-4 rad/s, four standard deviations of the 3 s mean; readings at rest within 0.01 m/s²; noise within 15 %), `TestMovingRestartsTheMeasurement` (a nudge at half time: moved for `MovedHold`, waiting, measuring from 0, the result untouched), `TestUpdateDoesNotAllocate`.
  - **Stillness:** `TestWaitsForStillness` (a hand's tremor, 0.1 m/s², and a motor's gyro jitter, 0.03 rad/s, never start it), `TestSlowMotionsRestartTheMeasurement` (the lean and the turn), `TestProgressRisesToDone` (rises steadily, 1 at done, measuring from 2·`Window`).
  - **Figures:** `TestOnePoseReadsTheSurfaceTilt` (1° of table tilt read as offset), `TestStatsAreMeanAndStandardDeviation`.
  - **Forecast:** `TestHeadingForecastMatchesWhatCalibrationsLeave` (0.8–1.5 times), `TestPositionForecastMatchesTheStrapdownPath` (0.75–1.5 times), `TestHeadingForecastSeesAWanderingBias` (a gyro bias growing 1 mrad/s each second goes over the line; a steady one stays under).
  - **API:** `TestResetStartsAgain`, `TestBadSamplesAreIgnored` (NaN, zero and negative dt), `TestNewRejectsConfig`.
- Correct: yes on these properties. 29 mutants (`ER-SIMALGO-logs/calib-tg3-mutants.txt`): 28 fail a test: the warm-up, the plain start of the statistics, each spread and mean test, the waiting reference, the still window, ignoring a move, `MovedHold`, Welford's variance, the mean's divisor, finishing at half time, the face's axis and sign, the rest target's sign, the noise's divisor, each heading term, the position drift, `seeRest`, Progress at done, Result before done, the dt check, the rest target in the offsets, the bias from the live mean. The survivor: the position forecast without its noise term (16 % of the total in quadrature, under what a simulation comparison resolves). Removed after mutants survived them: a second clearing of the measurement at moved and a reference taken at moved→waiting (both redundant).
- Speed: 10 ns per `Update` on the host (Apple M5), 0 allocs; not yet timed on the badge.
- TinyGo 0.43 `-target=tufty2040` (`calibtufty`: MPU6050 → Calibrator → seeded strapdown with Mahony): 97,512 B flash; `-print-allocs='calib|strapdown'` lists only strapdown's 736-byte `Nav` from `New`; the 212-byte Calibrator stays off the heap.

## TODOs

- [ ] tinygui: the calibration page (TG-3) on the badge; the owner's hand check (flat and on a side, a nudge restarts, the forecast against how the motion pages drift) — x/math/nav

## Links

- Module: [[navigation]] · Related: [[strapdown-inertial-navigation]], [[mahony-ahrs]], [[mpu6050-imu-driver]] · Ticket: [[ER-068-startup-calibration]]
