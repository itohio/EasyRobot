# Spec: Coherent M-ASK Demodulator — Software Implementation (Engineering / Mathematical)

## 0. Scope & assumptions

- Input: real passband or baseband samples `r[n]` sampled at `F_s` (samples/sec). M-ASK can work with real signals (unlike QAM/PSK which typically require complex IQ).
- Transmitter: M-ASK modulation with constellation size `M` (M = 2, 4, 8, 16, ...). Gray bit mapping assumed for bit error performance.
- Symbols: symbol period `T_sym`, symbol rate `R_sym = 1/T_sym`. Samples per symbol `N = round(F_s * T_sym)`. Prefer integer `N`; otherwise use resampling or fractional-timing interpolation.
- Channel model: AWGN + possible carrier frequency offset (CFO) and phase noise, possible slight Doppler, and multipath/ISI if specified.
- Coherent demodulation requires carrier phase recovery and symbol timing recovery. M-ASK is amplitude-based, so envelope detection is also possible (non-coherent).
- Constellation: amplitude levels `A_m` for `m = 0, 1, ..., M-1`. Typically evenly spaced or optimized for power efficiency.

---

## 1. Signal model

Transmitted (real passband or baseband) for symbol index `k`, symbol amplitude `A_{m_k}` corresponding to symbol `s[k]`:

\[
s(t) = A_{m_k} \cdot \cos(2\pi f_c t + \phi_0), \quad t \in [kT_{sym},(k+1)T_{sym})
\]

where:
- `A_{m_k}` = amplitude level for symbol `m_k` ∈ {0, 1, ..., M-1}
- `f_c` = carrier frequency (for passband) or 0 (for baseband)
- `φ_0` = initial phase (constant)

For baseband (no carrier):
\[
s(t) = A_{m_k}, \quad t \in [kT_{sym},(k+1)T_{sym})
\]

**Amplitude levels**:
- **Evenly spaced**: `A_m = m \cdot A_{max} / (M-1)` for `m = 0, 1, ..., M-1`
  - M=2 (OOK): `A_0 = 0`, `A_1 = A_{max}`
  - M=4: `A_0 = 0`, `A_1 = A_{max}/3`, `A_2 = 2A_{max}/3`, `A_3 = A_{max}`
- **Power-efficient (non-zero minimum)**: `A_m = A_{min} + m \cdot (A_{max} - A_{min}) / (M-1)`
  - All levels non-zero (better for synchronization)

Receiver observes with CFO `f_Δ`, phase noise `φ(t)`, and AWGN `w(t)`:

\[
r(t) = A_{m_k} \cdot \cos(2\pi (f_c + f_\Delta) t + \phi_0 + \phi(t)) + w(t)
\]

For baseband (after downconversion):
\[
r(t) = A_{m_k} + w(t)
\]

Discrete-time: `r[n] = r(nT_s)`.

Objective: recover symbol indices `m_k` → bits, using coherent integration per symbol after correcting CFO/phase and aligning timing, or using envelope detection (non-coherent).

---

## 2. High-level processing chain

1. **Front-end preprocessing**: DC removal, AGC, anti-alias/anti-imaging filter.
2. **Coarse acquisition**:
   - Frame/preamble detection and coarse symbol timing estimate.
   - Coarse CFO estimation (if passband) via FFT of long block or phase differences.
   - AGC initialization for amplitude normalization.
3. **Fine tracking**:
   - Carrier phase & frequency tracking (Costas loop or decision-directed PLL) for passband.
   - Symbol timing recovery (timing DMA / DLL / Mueller & Müller / Gardner).
   - AGC for amplitude tracking.
4. **Symbol demodulation**:
   - Matched filtering over symbol interval.
   - Envelope detection or coherent amplitude estimation.
   - Compute decision metric and soft LLRs or hard bits.
5. **Bit mapping & frame handling**:
   - Gray mapping, framing (preamble, CRC), error handling, re-acquisition.

---

## 3. Key mathematical blocks

### 3.1 Matched filter / symbol integrator

For symbol window samples `r[n]`, `n = 0..N-1` (after timing & CFO correction), matched-filter output:

\[
Y = \sum_{n=0}^{N-1} r[n] \cdot g[n]
\]

- `g[n]` is matched filter pulse (conjugate of transmit pulse), e.g., rectangular (1) or RRC samples.
- If `g[n] = 1` (rectangular), `Y` is simply the sum of samples over the symbol.
- For passband signals, `Y` is complex (from IQ demodulation); for baseband, `Y` is real.

### 3.2 Carrier removal (mixing) - Passband only

Before matched filtering, mix the input by the estimated carrier:

\[
\tilde r[n] = r[n] \cdot e^{-j(2\pi \hat f_\Delta n T_s + \hat\phi_n)}
\]

or for real signals, use quadrature demodulation:
\[
I[n] = r[n] \cdot \cos(2\pi \hat f_c n T_s + \hat\phi_n)
\]
\[
Q[n] = r[n] \cdot \sin(2\pi \hat f_c n T_s + \hat\phi_n)
\]

Then envelope: `|Y| = \sqrt{I^2 + Q^2}`

### 3.3 Envelope detection (non-coherent)

For passband signals, envelope detection is simpler and doesn't require phase lock:

\[
Y_{env} = \sqrt{\left(\sum_{n=0}^{N-1} r[n] \cos(2\pi f_c n T_s)\right)^2 + \left(\sum_{n=0}^{N-1} r[n] \sin(2\pi f_c n T_s)\right)^2}
\]

Or simpler: `Y_{env} = |Y|` where `Y` is the complex matched filter output.

### 3.4 Decision rule (hard)

The ML decision for M-ASK in AWGN (coherent) is:

\[
\hat m = \arg\min_m |Y - A_m|^2
\]

For baseband (real `Y`):
\[
\hat m = \arg\min_m (Y - A_m)^2
\]

**Quantization approach**:
1. Normalize: `Y' = Y / A_{max}` (or use AGC-normalized value)
2. Quantize to nearest level:
   \[
   \hat m = \begin{cases}
     0 & \text{if } Y' < A_1 / (2 A_{max}) \\
     m & \text{if } (A_{m-1} + A_m) / (2 A_{max}) \leq Y' < (A_m + A_{m+1}) / (2 A_{max}) \\
     M-1 & \text{if } Y' \geq A_{M-2} / (2 A_{max})
   \end{cases}
   \]

For evenly spaced levels with `A_m = m \cdot A_{max} / (M-1)`:
\[
\hat m = \text{round}\left(\frac{Y' \cdot (M-1)}{A_{max}}\right)
\]
clamped to `[0, M-1]`.

### 3.5 Soft metrics (LLR)

For bit-level soft decisions (for FEC), compute LLR for bit `b_i` of the symbol:

\[
LLR(b_i) \approx \ln\frac{\sum_{m \in S_1} \exp\!\big(\frac{-(Y - A_m)^2}{σ^2}\big)}{\sum_{m \in S_0} \exp\!\big(\frac{-(Y - A_m)^2}{σ^2}\big)}
\]

For practical implementation, approximate with max-log:

\[
LLR(b_i) \approx \frac{1}{σ^2}\big( \min_{m\in S_0}(Y - A_m)^2 - \min_{m\in S_1}(Y - A_m)^2 \big)
\]

where `S_1`/`S_0` are symbol sets with bit `i` = 1/0.

---

## 4. Carrier recovery (PLL / Costas loop) - Passband only

M-ASK with passband requires phase lock for coherent detection. Use a decision-directed Costas loop or simpler envelope detection.

### 4.1 Phase error detector (PED) - Coherent

For coherent detection:

1. Compute matched filter output `Y` (complex from IQ demodulation).
2. Make hard decision: `\hat A = A_{\hat m}`
3. Error estimate:

- **Decision-directed**:
  \[
  e_\phi = \Im\{ \hat A^* \cdot Y \} = \hat A \cdot \Im\{Y\}
  \]
  This is proportional to sine of phase error.

- **Simpler for ASK**: Use I component only:
  \[
  e_\phi = \Im\{Y\} \quad \text{(if I component should be dominant)}
  \]

### 4.2 Envelope detection (non-coherent alternative)

For non-coherent detection, use envelope:

\[
Y_{env} = |Y| = \sqrt{\Re\{Y\}^2 + \Im\{Y\}^2}
\]

Then quantize `Y_{env}` to nearest amplitude level. This avoids phase tracking complexity but loses ~3dB performance vs coherent.

### 4.3 Loop update (coherent only)

Discrete-time loop integrator form:

\[
\omega[n+1] = \omega[n] + K_i \, e_\phi[n]
\]
\[
\theta[n+1] = \theta[n] + K_p \, e_\phi[n] + \omega[n+1]
\]

- `θ` is phase estimate; `ω` is frequency integrator.
- `K_p`, `K_i` chosen per desired loop bandwidth and damping.

---

## 5. Timing recovery (symbol synchronization)

Goal: align integration windows to symbol boundaries and correct fractional timing.

### 5.1 Approaches

- **Non-data-directed**: Gardner TED, Mueller & Müller algorithms.
- **Data-directed**: use known preamble symbols to compute timing offset via correlation; then switch to tracking.
- **Decision-directed**: use decisions to compute early/late energy error.

### 5.2 Gardner TED (for real or complex baseband)

For oversampled signal (samples per symbol `Ns` > 2):

\[
e_t[n] = r(t+\tfrac{T}{2}) \cdot (r(t) - r(t+T))
\]

For complex signals:
\[
e_t[n] = \Re\{ r(t+\tfrac{T}{2}) \cdot (r^*(t) - r^*(t+T)) \}
\]

Discrete form uses interpolated samples at `±T/2`. Update timing phase `τ` with small loop gain `K_t`.

### 5.3 Mueller & Müller

Uses consecutive symbol-spaced samples and previous decisions to produce timing error; robust and widely used.

### 5.4 Fractional delay interpolation

If timing offset not integer samples, use interpolation (Farrow filter or polyphase) to produce symbol-aligned samples prior to matched filtering.

---

## 6. Acquisition (frame & coarse CFO)

### 6.1 Preamble & frame detection

- Use a known preamble sequence of symbols (e.g., alternating high/low amplitudes) to correlate and detect frame boundaries.
- Slide a window of `L_preamble * N` samples; compute correlation metric:

\[
S(\tau) = \left| \sum_{k=0}^{L_p-1} Y(\tau + kN) \cdot A_{preamble}[k] \right|
\]

where `Y(·)` is matched-filter output at candidate symbol offsets. Peak exceeding threshold → acquisition.

### 6.2 Coarse CFO estimation (passband only)

- Methods:
  - FFT of long block to find peak frequency shift.
  - Use phase differences across repeated preamble symbols:

\[
\hat f_\Delta = \frac{1}{2\pi T_{rep}} \angle\bigg( \sum_{n\in W} r[n] r^*[n + T_{rep,\text{samples}}] \bigg)
\]

- Apply coarse correction (multiply `r[n]` by `e^{-j2π \hat f_Δ n T_s}`) and initialize PLL.

### 6.3 AGC initialization

- Estimate signal amplitude from preamble:
  \[
  \hat A_{max} = \frac{1}{L_p} \sum_{k=0}^{L_p-1} |Y_{preamble}[k]|
  \]
- Normalize received signal: `r'[n] = r[n] / \hat A_{max}`

---

## 7. Symbol decision & mapping

- After timing & carrier correction (or envelope detection), compute `Y` per symbol. Decision:
  - Hard: quantize `Y` (or `|Y|` for envelope) to nearest amplitude level.
  - Soft: compute LLRs per bit using max-log approx from `(Y - A_m)²` metrics.
- Gray mapping reduces bit errors for adjacent symbol errors.

**Gray mapping for M-ASK**:
- For each amplitude level, assign Gray-coded bit pattern.
- Example (4-ASK): Levels {0, 1, 2, 3}
  - Level 0 → bits 00
  - Level 1 → bits 01
  - Level 2 → bits 11
  - Level 3 → bits 10
  - (Gray code: 00, 01, 11, 10)

---

## 8. Frame & CRC handling

- Use preamble for sync, then fixed-length payload + CRC.
- On decode, verify CRC; on failure, allow small timing adjustments (±1 sample) and re-try; else re-enter acquisition.
- Maintain counters for packet loss, SNR metrics, and re-acquisition stats.

---

## 9. Structs / API (abstract)

### 9.1 `CoherentMASKDecoder` (stateful)

**State fields**
- `Fs`, `T_sym`, `N` (samples/symbol)
- `M`, `amplitude_levels` (array of `A_m` values)
- `ring_buffer` (real or complex); buffer length ≥ `L_acq*N + margin`
- `state` ∈ {SEARCH, ACQUIRE, TRACK}
- **Timing state**:
  - `timing_phase` (fractional offset), `sample_index`
  - interpolator coefficients (for fractional delays)
- **Carrier PLL state** (passband only):
  - `theta`, `omega`, `Kp`, `Ki`
- **AGC state**:
  - `amplitude_estimate`, `agc_gain`
- **Acquisition**:
  - `preamble` (symbol array), `acq_threshold`, `acq_window`
- **Noise estimator**: `noise_var`, SNR estimator
- **Detection mode**: `coherent` (true) or `non_coherent` (false, envelope detection)
- **Decoder**:
  - `frame_assembler`, `soft_buffer`, CRC checker
- **Metrics**: SNR, correlation peaks, bit error counters

**Primary methods**
- `Init(params)` — sets rates, amplitude levels, loop gains, preamble
- `FeedSamples(samples[])` — push to ring buffer and run `ProcessStream()`
- `ProcessStream()` — handles state machine: SEARCH→ACQ→TRACK
- `ComputeMatchedFilter(start)` — compute `Y` for window at start (using interpolation if needed)
- `QuantizeAmplitude(Y)` — quantize amplitude to nearest level
- `CarrierLoopUpdate(e_phi)` — update `theta`, `omega` (coherent only)
- `TimingLoopUpdate(e_t)` — update `timing_phase`
- `AGCUpdate(Y)` — update amplitude estimate and gain
- `GetDecodedFrames()` — return collected frames
- `Reset()`

### 9.2 `CoherentMASKEncoder` (stateful)

**State**
- `Fs`, `T_sym`, `N`, `M`
- `amplitude_levels` (array of `A_m` values)
- `symbol_queue`
- `phase_accumulator` (for continuous phase if passband)
- `pulse_shape` (RRC etc.)

**Methods**
- `Init(params)`
- `FeedBits(bits[])` → converts to symbols and queues
- `NextSamples(n)` → generate `n` real or complex samples (apply pulse shaping & carrier if passband)
- `Reset()`

---

## 10. Numerical & loop tuning guidance

- PLL loop bandwidth `B` typically 1–5% of symbol rate for low Doppler; increase as needed for dynamics. Use 2nd-order loop formulas to set `Kp`, `Ki` for `ζ≈0.707`.
- Timing loop bandwidth smaller than PLL; use lower gains to avoid interaction.
- Symbol integration length `N` large gives noise averaging but reduces agility; choose `N` per system bandwidth & channel.
- AGC: M-ASK requires accurate amplitude estimation. Use moving average or exponential filter on `|Y|` to estimate signal amplitude. AGC bandwidth should be slower than symbol rate but fast enough to track channel variations.

---

## 11. Implementation details & optimizations

- **Matched-filter efficiency**: precompute `g[n]` and use vectorized dot-product for `Y`.
- **Carrier mixing**: for passband, implement phase accumulator `φ[n+1] = φ[n] + 2π (f_center + f_est) / F_s` and compute `e^{-jφ[n]}` efficiently with recursive trig or complex multiply.
- **Envelope detection**: for non-coherent, use `|Y| = \sqrt{I² + Q²}` or approximate with `max(|I|, |Q|) + 0.5 min(|I|, |Q|)` for lower complexity.
- **Interpolation**: implement Farrow structure for fractional sample delays, or polyphase filterbank if oversampling factor small.
- **Soft outputs**: compute LLRs with max-log approximation for FEC decoders (LDPC, turbo).
- **Quantization**: precompute quantization thresholds and use lookup table.
- **Complexity**: per-symbol cost dominated by N mul-adds for matched filter; optimize with SIMD.

---

## 12. Acquisition & performance considerations

- Preamble length tradeoff: longer preamble → faster acquisition at low SNR but overhead increases.
- CFO tolerance before demod: ensure coarse CFO estimator handles expected Doppler range; otherwise use envelope detection (non-coherent).
- Phase noise and residual frequency error degrade performance—estimate and compensate promptly, or use envelope detection.
- AGC: M-ASK is sensitive to amplitude errors. Ensure AGC converges before symbol decisions. Use robust amplitude estimation (median filter) to reject outliers.

---

## 13. Handling impairments

- **Large CFO**: perform coarse FFT-based search across freq bins, correct, then PLL. Or switch to envelope detection.
- **Phase ambiguity**: M-ASK has 2-fold phase ambiguity (0° vs 180°) for passband. Resolve via preamble or use envelope detection.
- **Amplitude errors**: use AGC to normalize received signal amplitude. M-ASK is very sensitive to amplitude errors.
- **DC offset**: remove DC component before amplitude quantization. Use high-pass filter or DC estimation and subtraction.
- **Multipath**: if severe, implement equalizer (MMSE) using known pilots or sequence-based channel estimation.
- **Burst interference**: detect using energy threshold; mute decoder or hold previous phase/timing for short bursts.

---

## 14. Testing & validation

- **Unit tests**: matched filter with noiseless amplitude level; PLL step response; timing loop convergence on synthetic offsets; AGC convergence.
- **AWGN tests**: BER vs Eb/N₀ curves for chosen M; compare to theoretical coherent and non-coherent M-ASK curves.
- **CFO & Doppler tests**: inject offsets ±X Hz and verify acquisition & tracking (coherent) or envelope detection (non-coherent).
- **AGC tests**: verify amplitude normalization under varying signal levels.
- **DC offset tests**: verify DC removal and amplitude quantization under DC bias.
- **Real data tests**: capture real samples from transmitter, run decoder end-to-end.
- **Stress tests**: buffer wrap, reconnection, re-acquisition under packet loss.

---

## 15. Example parameter sets (starter)

- **2-ASK (OOK)**: `R_sym = 1e3` sps, `Fs = 16e3` → `N = 16`. PLL BW = 0.01·R_sym (10 Hz). Preamble 64 symbols. Levels: `A_0 = 0`, `A_1 = 1.0`.
- **4-ASK**: same sampling, but expect ~4 dB more Eb/N₀ for same BER. Use longer preamble (128 symbols) in low-SNR. Levels: `A_0 = 0`, `A_1 = 1/3`, `A_2 = 2/3`, `A_3 = 1.0`.
- **8-ASK**: expect ~6–8 dB more Eb/N₀ than 4-ASK. Levels: evenly spaced from 0 to 1.0.
- **16-ASK**: very sensitive to amplitude errors; requires excellent AGC. Levels: evenly spaced from 0 to 1.0.

---

## 16. Amplitude level design

### 16.1 Evenly spaced levels

For `M` levels with maximum amplitude `A_{max}`:

\[
A_m = m \cdot \frac{A_{max}}{M-1}, \quad m = 0, 1, ..., M-1
\]

**Average power** (assuming uniform symbol distribution):
\[
P_{avg} = \frac{1}{M} \sum_{m=0}^{M-1} A_m^2 = \frac{A_{max}^2}{M(M-1)^2} \sum_{m=0}^{M-1} m^2 = \frac{A_{max}^2 (2M-1)}{6(M-1)}
\]

**Normalization** (to achieve average power = 1):
\[
A_{max} = \sqrt{\frac{6(M-1)}{2M-1}}
\]

### 16.2 Power-efficient levels (non-zero minimum)

For better synchronization, use non-zero minimum:

\[
A_m = A_{min} + m \cdot \frac{A_{max} - A_{min}}{M-1}, \quad m = 0, 1, ..., M-1
\]

where `A_{min} > 0` (e.g., `A_{min} = A_{max} / M`).

### 16.3 Optimal levels (maximize minimum distance)

For maximum error performance, optimize levels to maximize minimum distance subject to power constraint. Typically solved numerically.

---

## 17. Coherent vs Non-coherent Detection

### 17.1 Coherent Detection

- Requires phase lock (Costas loop or PLL)
- Better performance: ~3dB gain over non-coherent
- More complex: requires carrier recovery
- Sensitive to phase errors

### 17.2 Non-coherent (Envelope) Detection

- No phase lock required
- Simpler implementation
- ~3dB performance loss vs coherent
- Robust to phase errors
- Good for low-complexity receivers

**Recommendation**: Use envelope detection for simplicity, coherent for maximum performance.

---

## 18. Deliverables for engineer

- Implement `CoherentMASKDecoder` with API described above. Maintain state for timing & carrier loops (if coherent).
- Include acquisition module (preamble-based), coarse CFO estimator (if passband), PLL & timing loops, matched filter demodulator, AGC, and soft/hard decision outputs.
- Support both coherent and non-coherent (envelope) detection modes.
- Provide testbench to simulate AWGN, CFO, timing offset, phase noise, amplitude errors, DC offset; generate BER curves and acquisition statistics.
- Provide parameter set recommendations and tuning guide.

---

## 19. References (conceptual)

- Digital Communications theory (coherent and non-coherent M-ASK performance)  
- Costas loop and amplitude-based phase detectors  
- Gardner / Mueller & Müller timing recovery  
- Matched filtering and Nyquist pulse shaping (RRC)  
- Soft metrics and max-log LLR approximations
- AGC (Automatic Gain Control) for amplitude normalization
- Envelope detection for non-coherent demodulation

---

### End of specification

