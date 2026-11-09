---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, filter, dsp, kalman, ahrs, slam]
workstream: [W7, W8, W3]
packages: [x/math/filter, x/math/filter/agc, x/math/filter/ahrs, x/math/filter/ekalman, x/math/filter/fir, x/math/filter/gaussian, x/math/filter/iir, x/math/filter/kalman, x/math/filter/ma, x/math/filter/median, x/math/filter/pll, x/math/filter/savgol, x/math/filter/slam]
test_result: fail
---
# Filters (x/math/filter)

**Purpose:** Signal and state estimation building blocks: 1-D streaming filters (FIR/IIR/MA/median/Gaussian/Savitzky–Golay/AGC), carrier PLL for the modems, Kalman/EKF, IMU attitude (Madgwick/Mahony), and a ray-cast EKF localizer with log-odds mapping.

**Packages** (source LOC / test LOC; no build tags; pure Go, float32 via `github.com/chewxy/math32`):

| Package | Src | Tests | Notes |
|---|---|---|---|
| `filter` | 21 | – | interfaces `Resetter`, `Updater[T]`, `Filter[T,K]`, `Processor[T]` (filter.go:3-21) |
| `agc` | 58 | none | used by `dsp/modulation/{mpsk,qam}` demodulators |
| `ahrs` | 596 | 525 | Madgwick, Mahony, functional options |
| `ekalman` | 505 | 347 | EKF, numerical Jacobians |
| `fir` | 259 | 133 | used by `dsp/modulation/internal/fir.go` |
| `gaussian` | 139 | none | |
| `iir` | 283 | 221 | |
| `kalman` | 384 | 214 | |
| `ma` | 129 | 151 | |
| `median` | 101 | none | |
| `pll` | 109 | 595 | used by `dsp/modulation/{mfsk,mpsk,qam}` |
| `savgol` | 200 | none | |
| `slam` | 753 | 362 | depends on `ekalman`, `x/math/grid` |

## Public API

- Two composition styles coexist: **stream processors** `Process(float32) float32` (+`Reset`) — fir, iir, ma, median, gaussian, savgol, agc (only ma declares `Processor`); and **state filters** `Update(dt, input)` + `Input()/Output()` implementing `filter.Filter[T,K]` — kalman/ekalman (`vec.Vector`), ahrs (`matTypes.Matrix` 3×3 rows accel/gyro/mag → `vec.Quaternion`), slam (2×N [angles; distances] → `mat.Matrix3x3`).
- Constructors: positional `New(...)` (kalman, fir, iir, ma…), functional options (ahrs `WithKP/WithKI/WithMagnetometer`, slam `WithMap/WithResolution/WithOnline`). Designers: `fir.NewLowPass/HighPass/BandPass/BandStop/GeneralBand`, `iir.NewButterworth*`, `iir.NewChebyshev*`, `pll.NewFromLoopBandwidth`, `ma.AlphaFromHalfLife`.
- Buffer helpers: `ProcessBuffer(in, out)` (fir/iir, caller-provided) vs `ProcessBuffer(in) vec.Vector` (median/gaussian/savgol, allocate and reset state).

## Algorithms

<!-- vault:algorithms h=38ab62c2f766 -->

Counts: 2 tested · 2 implemented · 12 partial · 1 stub · 0 unverified (17 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[madgwick-ahrs]] | tested | pass | Matches reference equations, converges |
| [[moving-average-filters]] | tested | pass | `Filter` = running-sum SMA over a ring buffer (averages over the samples seen so far until the… |
| [[fir-windowed-sinc-lowpass-highpass]] | implemented | pass | Correct by probe |
| [[iir-rbj-biquad-bandpass-bandstop]] | implemented | pass | Correct by probe |
| [[automatic-gain-control]] | partial | none | Instantaneous level, inverted α |
| [[ekf-raycast-localization]] | partial | pass | Zero Jacobian, no motion model |
| [[extended-kalman-filter]] | partial | pass | Same aliasing; Jacobian at predicted state |
| [[fir-band-design]] | partial | pass | Wrong band impulse response |
| [[gaussian-smoothing-filter]] | partial | none | Rotated kernel, passthrough |
| [[iir-butterworth-filters]] | partial | pass | LP unstable, HP wrong, order ignored |
| [[linear-kalman-filter]] | partial | pass | Covariance aliasing kills filter after 1 update |
| [[log-odds-occupancy-mapping]] | partial | pass | Hits rarely marked, diagonal ranges wrong |
| [[mahony-ahrs]] | partial | pass | 6-DOF error sign flipped (converges upside down) |
| [[median-filter]] | partial | none | Periodic passthrough |
| [[phase-locked-loop]] | partial | fail | Single-wrap phase error |
| [[savitzky-golay-filter]] | partial | none | Right coefficients, wrong application |
| [[iir-chebyshev-type1]] | stub | pass | Falls back to Butterworth |

<!-- /vault:algorithms -->

## Build targets and dependencies

- No CGO, no build tags, no goroutines, no reflection except `sort.Slice` in median. All math float32 (`math32`) except `agc` (`math.Abs` via float64, agc.go:37) — good fit for ESP32's single-precision FPU [INFERENCE: not built with TinyGo].
- Allocation-free per sample: fir, iir, ma, pll, ahrs, agc. Allocating per sample/update: median (2 slices + sort), kalman (innovation), ekalman (innovation + 2 vectors per state dim for numerical Jacobians + user callbacks returning new slices), slam (ray-cast vector per `h` call, map clone for non-`mat.Matrix` maps). `mat.Matrix` is `[][]float32`; `Mul` silently writes into a temporary when the destination is non-contiguous (`x/math/mat/mat.go:68-93,496-516`) — a trap for user-built matrices.
- Embedded suitability: ahrs, fir, iir core, ma, pll are MCU-ready once the defects are fixed; kalman/ekalman need fixed-size variants (e.g. `Matrix3x3`-based) to avoid heap churn.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/filter/...` (2026-11-09; raw `tmp/easyrobot/filters/test-2026-11-09.json`): ahrs 9 pass, ekalman 6, fir 8, iir 13, kalman 6, ma 8, slam 12 — ok; **pll FAIL 9 pass / 2 fail** (`TestLoop_FrequencyTracking` impl defect, `TestLoop_Rotate` test defect; `TestLoop_NoiseLowSNR/5dB_Medium` flaky — unseeded global rand); agc, gaussian, median, savgol have no tests. Same as baseline 2026-11-08 §4.
- Probes (all under `~/src/tmp/easyrobot/filters/`): `kfprobe` (KF/EKF vs scalar reference), `ahrsprobe` (convergence), `dspprobe` (FIR/IIR magnitude responses, Gaussian impulse, median spikes), `sgprobe`, `pllprobe`, `slamprobe`. Results in the algorithm notes.
- Test quality: most filter tests only assert "not NaN" or loose bounds after one step; none would have caught the KF/EKF covariance collapse, the unstable Butterworth LP, or the Mahony sign error.

## API notes (ease of use, composability)

- **Good:** small constructors, per-sample `Process` that composes trivially into chains; `filter.Filter[T,K]` gives a uniform `Update/Input/Output/Reset` for estimators; ahrs uses fixed-size types (no allocation) and functional options; PLL exposes both gain and bandwidth APIs and IQ rotation helpers that the modems reuse.
- **Friction:**
  - Two incompatible streaming shapes: `ProcessBuffer(in, out []float32)` (fir/iir) vs `ProcessBuffer(vecTypes.Vector) vec.Vector` that allocates *and resets the filter* (median/gaussian/savgol). Only `ma` asserts `filter.Processor`. Proposal: one `Processor` interface with `Process(x)` and `ProcessBlock(dst, src)`, asserted by every stream filter.
  - Estimators overload "all-zero measurement" as "no measurement" (kalman.go:353, ekalman.go:474) and ignore `timestep` (kalman) — make missing data explicit (`Predict(dt)` / `Correct(z)` split).
  - AHRS input packed as a 3×3 matrix (rows accel/gyro/mag) through `matTypes.Matrix` interface with a type switch (madgwick.go:40-59) — a typed `IMUSample{Accel, Gyro, Mag vec.Vector3D}` would be clearer; mag presence is a constructor flag, not per-sample.
  - Designers silently ignore arguments (`order` in iir, `Weight` in `BandSpec`, `rippleDb` in Chebyshev) — should error or be removed.
  - Kalman exposes `P`, `F`, `H`, `Q`, `R` as public fields and also copies via setters — unclear ownership (aliasing bug is a symptom).
  - No shared delay-line/ring-buffer type: four packages reimplement the same ring with the same off-by-rotation bug (gaussian, savgol, median warm-up).
- **Composability for W7 (voice codecs):** the modems already depend on `pll`, `agc`, `fir`; fixing [[phase-locked-loop]] wrapping and [[automatic-gain-control]] is a prerequisite for noise-figure work ([[W7-filters-codecs-voice]]).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Replace NaN-only tests with frequency-response / convergence tests (probe code is a starting point) #p2
- [ ] Unified `Processor` interface + shared ring buffer; drop allocating `ProcessBuffer` variants #p3

## Inferred objectives (proposals)

- **proposal:** make the stream filters a verified, allocation-free "DSP block" layer for ESP32 audio (FIR/IIR-SOS/AGC/PLL) feeding the voice modems (W7), with golden frequency-response tests generated offline (scipy values committed as constants).
- **proposal:** a fixed-size attitude/odometry estimator for the legged robot (Madgwick or fixed Mahony + small EKF on `Matrix3x3`/`Matrix4x4`) to give body pitch/roll to the gait controller ([[W3-legged-ik]]).
- **proposal:** replace the ray-cast EKF with scan-matching or a particle filter over a shared occupancy grid for the lidar ([[W8-lidar]]).
