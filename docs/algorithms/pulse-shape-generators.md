---
type: algorithm
status: tested
tags: [easyrobot, dsp, generator, pulse-shaping]
workstream: W7
module: "[[dsp-core]]"
summary: "RC matches formula 1e-5; RC (not RRC) only"
source: [x/math/dsp/generators_trig.go, x/math/dsp/generators.go]
tests: [x/math/dsp/generators_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -v ./x/math/dsp/ → ok; TestRaisedCosineGeneratorMatchesFormula, TestSincGenerator_*, TestGaussianGenerator_* PASS (2026-11-09)"
---
# Pulse-shape generators: sinc, raised cosine, Gaussian

**What it does:** stateless sampled pulse shapes on a sample grid x = i − center:
- Sinc: `A·sin(π·s·x)/(π·s·x)`.
- Raised cosine: `A·sinc(s·x)·cos(πα·s·x)/(1−(2α·s·x)²)`, singularity at |2α s x| = 1 replaced by the analytic limit `A·(α/2)·sin(π/(2α))`.
- Gaussian: `A·exp(−x²/(2σ²))`; σ ≤ 0 → zeros.

**Where:**
- `SincGenerator` x/math/dsp/generators_trig.go:340-427.
- `RaisedCosineGenerator` x/math/dsp/generators_trig.go:431-634 (`NewRaisedCosineGenerator` clamps α to [0,1], scale 0→1; sin/cos by angle-addition recurrence :554-564; `Correlate` :592-634).
- `GaussianGenerator` x/math/dsp/generators.go:564-669 (incremental x² update :649-651).

**Inputs → outputs:** `(center, scale, [rolloff,] amplitude)` float32; `Generate/Add/GenerateWithOp(dst, reverse, op)` fill `dst[0..n)`; no time state (each call restarts at i=0). `reverse` mirrors index order.

## Completeness review

- Complete: yes. Not root-raised-cosine (RRC) — matched-filter pairs need RRC; consumers use RC on both ends [INFERENCE from grep: `NewRaisedCosineGenerator` in modulation qam/mpsk/mfsk/mask/dsss waveform files; filter pairing not checked here].
- Tested (PASS 2026-11-09):
  - `TestRaisedCosineGeneratorMatchesFormula` — 64 samples vs an independent per-sample formula, tol 1e-5 (α=0.35, scale 1/32): checks the recurrence does not drift over 64 samples.
  - `TestSincGenerator_Generate/_ZeroAtCenter/_Add/_GenerateWithOp` — peak 1 at center, zeros at ±1, symmetry, Add/op.
  - `TestGaussianGenerator_Generate/_ZeroAtCenter/_SigmaEffect/_InvalidSigma/_Add/_GenerateWithOp` — peak, symmetry, monotone, σ effect, zeros for σ=0.
- Correct: yes within tested lengths. Recurrence drift for long pulses (thousands of samples) is not tested; float32 angle-addition error grows ~linearly with n [INFERENCE].
- Sinc compares `xScaled == 0` exactly after incremental accumulation (generators_trig.go:381,401); a near-zero non-zero x still evaluates fine (sin x/x), so harmless.

## TODOs

- [ ] Add root-raised-cosine for TX/RX matched filtering — x/math/dsp/generators_trig.go:429-459 #p2
- [ ] Test RC over ≥1024 samples to bound recurrence drift — x/math/dsp/generators_test.go:653-665 #p3
- [ ] `RaisedCosineGenerator.Correlate` is O(N²) time-domain; allocates when dst < 2N−1 and N > 64 — x/math/dsp/generators_trig.go:602-613 #p3

## Links

- Module: [[dsp-core]]
- Consumers: [[modulation]] (waveform shaping), [[line-coding-and-fec]] (symbol shaping context)
- Siblings: [[sine-oscillator-generators]], [[chirp-generators]], [[noise-generators]]
