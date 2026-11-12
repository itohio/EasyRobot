---
id: ER-019
title: "Filter correctness"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W7
milestone: M-e
tags: [easyrobot, ticket, w7, filters]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-019 — Filter correctness

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Wrong-result bugs with wide reach in `x/math/filter` ([[filters]]); queued per O7, with the **PLL/AGC part before [[ER-022-audio-modem-acquisition|ER-022]]**:

- Kalman/EKF covariance aliasing — `kalman.go:338`, `ekalman.go:459`; EKF linearises after overwriting x — `ekalman.go:336-340`, `:368-372` ([[linear-kalman-filter]], [[extended-kalman-filter]]).
- Mahony 6-DOF error sign — `mahony.go:225-229` ([[mahony-ahrs]]).
- IIR Butterworth LP/HP design — `iir.go:135-191`; FIR band impulse response — `fir.go:216` ([[iir-butterworth-filters]], [[fir-band-design]]).
- Ring-buffer indexing/warm-up — `median.go:58`, `gaussian.go:65,74`, `savgol.go:72,79`.
- **PLL** full phase wrap — `x/math/filter/pll/pll.go:100-108`, `:64`; `TestLoop_Rotate`, unseeded noise `pll_test.go:44,513` ([[phase-locked-loop]]).
- **AGC** envelope detector and α = 1−exp(−1/(τ·fs)) — `agc.go:28,37` ([[automatic-gain-control]]).

## Dependencies and scope

Dependencies: None blocking. Stage A (PLL, AGC) before ER-022 (O7); stage B (the rest) any time. Serves M-e stage 2, W8 SLAM and any AHRS.

Not in scope: SLAM measurement Jacobian / motion model / occupancy mapping (`slam.go:203-229`, `mapping.go:131,173`) stay with W8 unless a lidar stage pulls them in.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Stage A: PLL test across the ±π seam and with |phase| > 3π fails first, then passes; `TestLoop_Rotate` fixed and noise seeded; AGC step-response test (attack/release time constants within 10 %) fails first, then passes.
- [ ] KF/EKF: multi-step test vs a scalar/float64 reference (P and x after N updates) fails first, then passes.
- [x] Mahony: convergence from a tilted static pose for both 6- and 9-DOF paths.
- [ ] IIR/FIR: response tests against known coefficient sets (LP/HP −3 dB at fc; BP/BS in-band ≈ 0 dB, out-of-band < −40 dB).
- [ ] Ring-buffer filters: output over the first w samples uses only the samples seen; matches a naive reference afterwards.
- [ ] `go test -count=1 ./x/math/filter/...` passes.

## Effort uncertainty and revision trigger

Product's guess: **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

- **Mahony (2026-11-12, ER-SIMALGO, for the badge's 6-axis MPU6050):** `TestMahonyConvergesToAStaticTilt` failed first (6-DOF settled at the mirror tilt), then passes with the 6-DOF error `a × v` (`mahony.go:224-231`); plus bias, yaw-rate, `Euler` and no-allocation tests in `convergence_test.go`, and the 6-DOF reference step in `ahrs_test.go` corrected to `a × v`. `go test -race -count=1 ./x/math/filter/ahrs/` pass. Details: [[mahony-ahrs]]. The other items are open.
