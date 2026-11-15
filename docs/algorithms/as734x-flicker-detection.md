---
type: algorithm
status: partial
tags: [easyrobot, devices, sensors, spectral, flicker, dsp]
workstream: W1
module: "[[sensors]]"
summary: "partial (decoder and frequency estimator wrong per probe)"
source: [x/devices/as734x/as7341.go, x/devices/as734x/as7343.go, x/devices/as734x/flicker_measure.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe on verbatim copies (2026-11-09): calculateFrequency(sine 100 Hz @1 kHz)=200.00, 120 Hz→239.58, 50 Hz→0.00; AS7341 decodeFlicker(0x2C 'valid, no flicker')=100, (0x01 unvalidated)=100; AS7343 decode 0x2C→0, 0x2D→100, 0x2E→120."
---
# AS734x flicker detection (hardware FD + software zero-crossing)

**What it does:** two paths. (1) Hardware flicker detector: AS7341 loads the FD SMUX table, sets FDEN, sleeps 200 ms and decodes FD_STATUS into 100/120 Hz; AS7343 reads FD_STATUS after enabling via ENABLE bit. (2) `MeasureFlickerFrequency` samples a "clear" channel via repeated `Read()` and estimates frequency by mean-crossing intervals.

**Where:**
- `x/devices/as734x/as7341.go:158-177` `detectFlickerAS7341`; `:179-197` `startFlickerAS7341`; `:199-206` `finishFlickerAS7341` (fixed 200 ms sleep, FD_STATUS 0xDB); `:208-227` `decodeFlickerFrequency`.
- `x/devices/as734x/as7343.go:43-54` `configureFlickerAS7343` (sets ENABLE bit 0x20); `:100-114` `readFlickerAS7343` (FD_STATUS 0xE3); `:116-125` `decodeAS7343Flicker`.
- `x/devices/as734x/flicker_measure.go:18-74` `MeasureFlickerFrequency`; `:79-151` `calculateFrequency`; `:155-157` `MeasureFlickerFrequencyQuick` (200 ms, 1000 S/s).

**Inputs → outputs:** `Flicker{Frequency uint16 (0, 1=unknown, 100, 120), Valid, Saturated, Raw}` inside `RawMeasurement`; `MeasureFlickerFrequency(duration, sampleRate) (float32 Hz, error)`.

FD_STATUS bit map used for review [INFERENCE: AS7341 datasheet/Adafruit from memory]: b0 100 Hz detected, b1 120 Hz detected, b2 100 Hz valid, b3 120 Hz valid, b4 saturation, b5 measurement valid. Adafruit maps 0x2C→0 (no flicker), 0x2D→100, 0x2E→120, else unknown.

## Completeness review

- Complete: no — no FD_TIME/FD_GAIN configuration on either variant; AS7343 path never waits for an FD measurement; software path cannot reach the requested sample rate.
- Tested: no test files. Probe (verbatim copies of the three pure functions; `go run .` 2026-11-09):
  - `calculateFrequency`, 200 samples of `1000+500·sin(2πft+0.3)` at 1 ms: 100 Hz → **200.00**, 120 Hz → **239.58**, 50 Hz → **0.00**.
  - `decodeFlickerFrequency` (AS7341): 0x00→0, **0x2C→100** (expected 0), 0x2D→100, 0x2E→120, 0x20→0, **0x01→100** (not validated), 0x30→0.
  - `decodeAS7343Flicker`: 0x2C→0, 0x2D→100, 0x2E→120, 0x01→0 — matches the bit map above.
- Correct: no (probe).

**Defects**
- AS7341 decoder treats the *valid* bits as detections: `case status&0x04 → 100`, `case status&0x08 → 120` (as7341.go:214-217), so "measurement valid, no flicker" (0x2C) reports 100 Hz; detection bits are accepted without their valid bits (:210-213). Cases `status&0x0C` (:218) are unreachable; return value 1 = "unknown" is undocumented.
- `calculateFrequency` counts both rising and falling crossings, so the mean interval is a half period → frequency ×2 (flicker_measure.go:123-143; probe 100→200).
- Hysteresis is not latched: `prevAbove/prevBelow` reset when a sample falls inside the ±2 % band, so crossings through the band are lost (flicker_measure.go:100-112; probe 50 Hz→0).
- Uses the nominal `interval`, not measured timestamps (flicker_measure.go:124), while each `Read()` takes ≈100–300 ms on AS7341 (two SMUX passes + 200 ms flicker sleep with default config) → real rate ≈3–10 S/s, heavy aliasing; `MeasureFlickerFrequencyQuick` (200 samples) blocks ≈60 s [INFERENCE from timings in code].
- `sampleRate <= 0` → integer divide by zero panic (flicker_measure.go:24).
- AS7343 "clear" channel taken as `Channels[0]` (flicker_measure.go:54-57); in auto-SMUX-18 order index 0 is FZ, VIS is at 4/5/10/11/16/17 [INFERENCE].
- AS7343 FD enable uses ENABLE bit 0x20 (as7343.go:49); FDEN is bit 6 (0x40) as on AS7341 (`regFlickerEnable`, registers.go:20) [INFERENCE] → hardware FD likely never enabled.
- AS7341 FD wait is a fixed 200 ms sleep (as7341.go:200), Adafruit uses ~500 ms [INFERENCE]; blocking, no FD-ready poll.

## TODOs

- [ ] Fix AS7341 FD_STATUS decoding (require b5 + detect&valid pairs; 0x2C→0) — x/devices/as734x/as7341.go:208 #p1 #backlog
- [ ] Fix ×2 error and latch hysteresis state in `calculateFrequency` — x/devices/as734x/flicker_measure.go:100 #p1 #backlog
- [ ] Use measured sample timestamps and reject `sampleRate<=0` — x/devices/as734x/flicker_measure.go:24 #p1 #backlog
- [ ] Verify AS7343 FDEN bit (0x20 vs 0x40) against datasheet — x/devices/as734x/as7343.go:49 #p1 #backlog
- [ ] Unit tests: decode tables and synthetic-sine `calculateFrequency` (50/100/120 Hz) — x/devices/as734x/flicker_measure.go:79 #p2
- [ ] Use a VIS/clear channel for AS7343 in `MeasureFlickerFrequency` — x/devices/as734x/flicker_measure.go:54 #p2
- [ ] Software path needs a fast single-channel mode (short ATIME, no SMUX reload) to be meaningful — x/devices/as734x/flicker_measure.go:36 #p3

## Links

- Module: [[sensors]] · Related: [[as734x-smux-and-spectral-read]] · DSP alternatives: [[dsp-core]]
