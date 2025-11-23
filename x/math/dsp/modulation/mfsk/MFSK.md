# Spec: Coherent M-FSK Demodulator Implementation (Engineering / Mathematical)


## 0. Overview & assumptions

- Encoder has produced coherent MFSK encoded signal
- Receiver works at complex baseband samples `r[n]` sampled at `F_s` (samples/sec). If input from ADC is real passband, assume it is first front-end downconverted to complex baseband (IQ).  
- Transmitter uses M-FSK: a symbol chooses one of `M` center frequencies `f_k = f_center + (k - (M-1)/2)·Δf` for `k = 0..M-1`. Symbols are transmitted for `T_sym` seconds (symbol interval).  
- Coherent detection: receiver estimates and compensates carrier frequency offset (CFO) and phase so that coherent correlation integrates signal energy constructively. This yields better performance than noncoherent detection, at cost of tracking complexity.  
- Sampling choices: let `N = round(T_sym * F_s)` be samples per symbol. Prefer `F_s` such that `N` is integer. If not, use resampling or fractional-timing handling.  
- Each tone is assumed narrowband during `T_sym` (i.e., frequency modulation within symbol negligible).

---

## 1. Signal model (baseband)

Transmitted complex baseband for symbol interval `k` carrying tone index `m = s[k]`:

\[
s(t) = A \, e^{j(2\pi f_m t + \phi_0)}
\quad\text{for } t\in[kT_{sym},(k+1)T_{sym})
\]

Receiver sees (with CFO `f_\Delta` and noise `w(t)`):

\[
r(t) = s(t)\,e^{j 2\pi f_\Delta t} + w(t)
= A\,e^{j(2\pi (f_m + f_\Delta) t + \phi_0)} + w(t)
\]

Discrete-time samples `r[n] = r(nT_s)`.

Objective: for each symbol interval, estimate `m` (tone index) reliably using coherent correlation after removing `f_\Delta` and phase drift.

---

## 2. High-level architecture

1. **Front-end & buffering**: accumulate incoming complex samples in ring buffer; provide symbol-length windows.
2. **Coarse acquisition**:
   - Detect presence (energy/preamble) and coarse symbol timing.
   - Coarse estimate of CFO `f_\Delta` (using long-window FFT or correlation across multiple symbols).
3. **Fine synchronization & tracking**:
   - Carrier phase/frequency tracking (single PLL on center frequency or per-tone trackers).
   - Symbol timing tracking (e.g., Gardner, Mueller & Müller, or symbol synchronizer via early/late correlators).
4. **Coherent correlators / matched filters**:
   - For each possible tone `k`, multiply `r[n]` by `e^{-j 2π f_k nT_s}` (i.e., complex mixing) and integrate across symbol interval → complex correlation `Z_k`.
   - Choose `\hat m = arg max_k |Z_k|` (or likelihood-based decision).
5. **Decision & framing**:
   - Map symbol indices to bits; assemble into frames using preamble & CRC; handle loss-of-sync re-acquisition.
6. **Optional: Viterbi or sequence decoding** for channels with coded sequences.

---

## 3. Key mathematical blocks

### 3.1 Coherent correlator (per symbol)
Given symbol window samples `r[n]`, `n = 0..N-1`, local oscillator at tone `k` with frequency `f_k` (Hz):

\[
Z_k = \sum_{n=0}^{N-1} r[n] \cdot e^{-j 2\pi f_k n T_s} \cdot w[n]
\]

- `w[n]` = chip/symbol pulse samples or sample window taper (e.g., rectangular = 1, or Hann to reduce spectral leakage).  
- `Z_k` is complex; with perfect synchronization, `Z_k` aligns phases and amplitude adds coherently.  
- Decision: `\hat m = \arg\max_k |Z_k|`. Optionally use `|Z_k|^2` as metric.

### 3.2 Likelihood & soft metrics
If noise is AWGN with variance `σ^2`, the log-likelihood for tone `k` is proportional to `|Z_k|^2`. Use `LLR` for coded decisions:

\[
LLR(m) \propto \frac{1}{σ^2} (|Z_{m_0}|^2 - |Z_{m_1}|^2)
\]

For soft-decision decoding, compute LLRs between symbol hypotheses.

---

## 4. Carrier frequency & phase estimation (coherent tracking)

Two-layer approach recommended:

### 4.1 Coarse CFO estimator (before coherent detection)
Use a long block `B` samples (several symbol durations) and compute FFT of `r` to find dominant tone ridge (or multi-tone energy center). Estimate coarse offset `f_\Delta^{coarse}`.

Alternative: use phase difference across repeated preamble tone symbols. If preamble contains repeated same-tone symbols separated by `T_rep`, estimate `f_\Delta` from phase rotation:

\[
\hat f_\Delta = \frac{\angle\left( \sum_{n \in \text{sym window}} r[n] \bar r[n+N_{rep}] \right)}{2\pi T_{rep}}
\]

### 4.2 Fine carrier tracking (PLL)
Use a single PLL to track residual carrier phase `θ[n]` after coarse correction. Feed the PLL with an error metric derived from the selected correlator:

- After mixing with `e^{-j 2\pi f_k n T_s}`, take the phase of `Z_k`: `\phi = arg(Z_k)`.
- Phase error (for BPSK-like carrier) can be `ε = arg(Z_k)` (or improved via Costas for phase ambiguity).
- PLL update (discrete integrator form):
  - `ω[n+1] = ω[n] + K_i * ε`
  - `θ[n+1] = θ[n] + K_p * ε + ω[n+1]`
- Convert `ω` to frequency offset estimate; use both to correct subsequent mixing.

Tuning: choose `K_p`, `K_i` based on loop bandwidth relative to symbol rate and expected Doppler dynamics.

### 4.3 Alternative per-tone phase tracking
If tones have phase continuity (coherent FSK), you can track phase for each tone: maintain complex phase `φ_k` updated per symbol by `Z_k`. But a single PLL on carrier center frequency is more common and simpler.

---

## 5. Symbol timing recovery (coherent)

Timing ensures integration windows align with actual symbol boundaries.

### 5.1 Use a symbol synchronizer (non-data-directed or data-directed)

- **Non-data-directed**: use energy envelope or autocorrelation (M&M or Gardner for bandpass signals). Computes timing error from early/late sample energy difference:
  - Early correlator `E = ∑ r[n + early] e^{-j2π f_k...}`
  - Late correlator `L = ∑ r[n + late] ...`
  - Timing error `ε_t = (|E| - |L|) / (|E| + |L|)`
  - Update timing phase `τ` with a DLL-like integrator.

- **Data-directed using preamble**: during preamble, correlate at candidate offsets and pick the best offset. For steady-state, use early/late loops.

### 5.2 Sample-precise timing
If fractional sample timing exists, implement interpolation (Farrow or polyphase) to resample and align windows. Alternatively, oversample (e.g., `Fs` >> symbol rate) and perform fractional delay via FIR interpolation.

---

## 6. Acquisition algorithm (detection & alignment)

1. **Preamble design**: use deterministic sequence of tones (e.g., repeated tone patterns or known symbol sequence) that simplifies acquisition: e.g., alternate tone A/B or a PN sequence of tone indices.
2. **Sliding correlation**:
   - Slide a window of `K* N` samples; for each offset `τ` inside one symbol period, compute sequence of `Z_k` for the preamble symbols and measure how well they match expected tone indices.
   - Compute score `S(τ) = Σ_j |Z_{expectedTone(j)}(τ + jN)|^2` (normalized).
3. **Peak detection**:
   - Find `τ* = argmax_τ S(τ)`. If `S(τ*)` exceeds threshold (`T_detect` based on noise estimate), declare acquisition with symbol boundary aligned at `τ*`.
4. **Coarse CFO**:
   - From the complex phases across repeated preamble symbols, compute coarse `f_Δ`.
5. **Initialize PLL & timing loops**:
   - Use `τ*` to set initial timing, set PLL frequency to `–f_Δ^{coarse}`, set loop integrators to zero.
6. **Switch to tracked mode**:
   - Decode symbols coherently; update loop filters continuously.

---

## 7. Decision & symbol mapping

- After coherent integration for the symbol interval `j`, compute `Z_k` for all `k`.
- Pick `\hat m_j = argmax_k |Z_k|`.
- Map `\hat m_j` to `log2(M)` bits via predetermined Gray mapping.
- Offer soft metrics per bit if needed: compute LLR for bit `b_i` as:

\[
LLR(b_i) \approx \ln\frac{\sum_{m \in S_1} e^{|Z_m|^2/σ^2}}{\sum_{m \in S_0} e^{|Z_m|^2/σ^2}}
\]

where `S_1` and `S_0` are symbol sets with `b_i = 1` or `0`.

---

## 8. Structs / API (abstract)

### 8.1 `CoherentMFSKDecoder` (stateful)

Fields (state):
- `Fs`, `T_sym`, `N` (int) — samples per symbol
- `M`, `f_center`, `delta_f` (freq grid)
- `tone_freqs[k]` = `f_center + (k - (M-1)/2)*delta_f`
- `ring_buffer` (complex float buffer) length ≥ `L_acq * N`
- `state` ∈ {SEARCH, ACQUIRE, TRACK}
- **Timing state**:
  - `symbol_offset` (fractional sample offset)
  - `sample_counter` (index modulo N)
- **Carrier PLL state**:
  - `theta` (phase), `omega` (frequency integrator), `Kp`, `Ki`
- **Noise estimator**: `noise_var`
- **Acquisition params**: preamble pattern, `acq_threshold`, `acq_window`
- `last_Z[k]` (complex) last symbol correlator outputs
- `soft_buffer` / `frame_assembler`
- `metrics` (SNR, correlation_peak)

Methods:
- `Init(params)` — set Fs, symbol timing, tone grid, loop gains, preamble
- `FeedSamples(samples[])` — push complex samples into ring buffer, call `ProcessStream()`
- `ProcessStream()` — main loop: if SEARCH => run acquisition; if TRACK => perform symbol-level demod: correct CFO & phase, compute `Z_k`, decide symbol, update PLL & timing
- `ComputeCorrelator(window_start)` — returns `Z_k` for k=0..M-1 (use efficient method: Goertzel bank or precomputed sin/cos tables and vector dot product)
- `GetDecodedBytes()` — returns assembled bytes/frames
- `Reset()`

### 8.2 `CoherentMFSKEncoder` (stateful) — optional

Fields:
- `Fs`, `T_sym`, `N`, `M`
- `tone_freqs[k]`
- `symbol_queue`
- `phase_accumulators` per tone (allow phase continuity if desired)
- `pulse_window` (optional)
Methods:
- `Init(params)`
- `FeedBits(bits[])`
- `NextSamples(n)` — generate complex samples (mix carrier with `e^{j2π f_k t}` for symbol k)
- `Reset()`

---

## 9. Efficient correlator implementation

- **Goertzel per tone**: for computing DFT at specific frequencies; complexity O(M·N) per symbol but with small M and moderate N it's efficient.
- **Precompute complex exponentials**: `e^{-j2π f_k n T_s}` for `n=0..N-1, k=0..M-1` to speed dot-products. Store per-symbol phase table and apply windowing.
- **FFT approach**: take N-point FFT of symbol window; map tone frequencies to bins (or interpolate between bins). Use when M is large or N is power-of-two and optimized FFT is available.
- **Overlap-save** / streaming FFT for acquisition over long windows.

---

## 10. Timing & PLL parameter guidelines

- PLL loop bandwidth `B_loop` choose small fraction of symbol rate (e.g., `B_loop ≈ 0.01–0.1 * (1/T_sym)`), trade-off between tracking agility and noise. Use standard control theory formulas to set `Kp`, `Ki` from desired natural frequency and damping ratio.
- Timing DLL early/late spacing `δ` = ±0.25–0.5 symbol (or fractional sample); use small loop bandwidth for stability.
- For mobile or doppler heavy channels, increase `B_loop` but accept higher jitter.

---

## 11. Acquisition performance & thresholds

- Determine `acq_threshold` from noise estimate: set such that false alarm probability is low given window length and #trials. Roughly:

\[
T_{acq} = \mu_{noise} + \kappa \sigma_{noise}
\]

with κ ~ 6–10 depending on desired P_FA.

- Evaluate via simulation: AWGN channel, varying Eb/N0, measure Probability of Detection vs False Alarm.

---

## 12. Handling impairments

- **Doppler / CFO**: coarse estimate via FFT over several symbols; fine via PLL. For large CFO, rotate samples by coarse `e^{-j2π f_{coarse} t}` before symbol-wise correlators.
- **Multipath / ISI**: if severe, use matched filtering and equalization; M-FSK is relatively resilient if symbol duration >> delay spread.
- **Phase noise**: reduces coherent gain; measure and possibly switch to noncoherent detector if phase noise is too large.
- **Timing jitter / fractional sample offsets**: use interpolation (Farrow filter) for fractional delaying of windows to optimize alignment.

---

## 13. Frame & bit alignment (practical)

- Use robust preamble: e.g., Barker-coded sequence of tone indices or PN tone sequence. Preamble length tradeoff: longer gives faster, more reliable acquisition at low SNR.
- After preamble detection and initial CFO estimate, mark that sample position as symbol boundary. Then decode fixed-length payload and CRC. If CRC fails, allow small timing adjustments (±1 sample) and re-evaluate, else re-enter acquisition.

---

## 14. Testing / verification plan

- Unit tests:
  - Correlator correctness with synthetic tone + noise.
  - PLL step response with synthetic constant tone + known offset.
  - Timing loop convergence on simulated symbol transitions.
- End-to-end tests:
  - AWGN channel: BER vs Eb/N0 plot, compare to coherent detection theoretical curves.
  - CFO test: inject offset ±x Hz, measure acquisition success & residual error.
  - Multipath test: two-tap channel; measure BER.
  - Real recorded samples (if available).
- Regression: stress tests for long runs, buffer wrap, event of lost sync.

---

## 15. Implementation tips & optimizations

- Use fixed-point only if CPU/embedded constraints require it; otherwise double/float simplifies loop tuning.
- Vectorize correlator computations (SIMD) or use BLAS-like operations for dot products.
- Precompute and cache `e^{-j2π f_k nT_s}` tables for all `k` and `n` (or generate on the fly with phase accumulator).
- Keep ring buffer contiguous for vector ops (use circular indexing).
- Measure CPU budget: correlator cost = O(M·N) mul-adds per symbol. Choose M and N accordingly.

---

## 16. Example parameter recommendation (starter)

- Symbol rate: `R_sym = 1000` sps (symbols/sec) → `T_sym = 1 ms`.  
- Fs = 48 kHz → `N = 48` samples/symbol.  
- M = 8 (8-FSK), Δf = 1/T_sym = 1 kHz (orthogonal spacing) → tone spacing = 1 kHz.  
- Use rectangular window (simple) with Goertzel for each tone; use coarse FFT over 128-256 ms for acquisition.

---

## 17. Summary of processing pipeline

1. Acquire samples into buffer.
2. Search for preamble & estimate coarse CFO and symbol offset.
3. Initialize PLL & timing loops.
4. For each symbol:
   - Correct samples by `e^{-j(2π \hat f_\Delta t + \hat θ)}`.
   - Compute `Z_k` correlators for k=0..M-1.
   - Select `\hat m = argmax |Z_k|`.
   - Map to bits, assemble frames.
   - Update PLL (phase/frequency) and DLL (timing) using chosen correlator outputs.
5. Validate frames via CRC; if fail, adapt or re-acquire.

---

### End of specification

Use this spec to implement `CoherentMFSKDecoder` and (optionally) `CoherentMFSKEncoder`. If you want, I can next:

- produce a **reference pseudocode** / flowchart for the `ProcessStream()` loop,  
- or generate a **Go skeleton** with key inner loops (correlator, PLL update, acquisition) to jumpstart implementation. Which would you prefer?
::contentReference[oaicite:0]{index=0}
