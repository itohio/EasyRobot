# Spec: Coherent M-PSK Demodulator — Software Implementation (Engineering / Mathematical)


## 0. Scope & assumptions

- Input: complex baseband samples `r[n]` sampled at `F_s` (samples/sec). If ADC is real passband, assume an external front-end downconverts to complex IQ or that a software downconversion stage exists.
- Transmitter: M-PSK modulation with constellation size `M` (M = 2,4,8,16,...). Gray bit mapping assumed for bit error performance.
- Symbols: symbol period `T_sym`, symbol rate `R_sym = 1/T_sym`. Samples per symbol `N = round(F_s * T_sym)`. Prefer integer `N`; otherwise use resampling or fractional-timing interpolation.
- Channel model: AWGN + possible carrier frequency offset (CFO) and phase noise, possible slight Doppler, and multipath/ISI if specified.
- Coherent demodulation requires carrier phase recovery (phase ambiguity handling for PSK) and symbol timing recovery. Optionally support differential PSK (DPSK) to resolve exact phase ambiguity.

---

## 1. Signal model

Transmitted (complex baseband) for symbol index `k`, symbol phase `θ_k` corresponding to symbol `s[k]`:

\[
s(t) = A \cdot \exp\!\big(j\theta_k\big), \quad t \in [kT_{sym},(k+1)T_{sym})
\]

with

\[
\theta_k = \frac{2\pi}{M} \cdot m_k + \theta_{const}
\quad\text{for } m_k\in\{0,\dots,M-1\}
\]

Receiver observes with CFO `f_\Delta`, phase noise `φ(t)`, and AWGN `w(t)`:

\[
r(t) = s(t)\,e^{j(2\pi f_\Delta t + \phi(t))} + w(t)
\]

Discrete-time: `r[n] = r(nT_s)`.

Objective: recover symbol indices `m_k` → bits, using coherent integration per symbol after correcting CFO/phase and aligning timing.

---

## 2. High-level processing chain

1. **Front-end preprocessing**: DC removal, AGC, anti-alias/anti-imaging filter.
2. **Coarse acquisition**:
   - Frame/preamble detection and coarse symbol timing estimate.
   - Coarse CFO estimation (via FFT of long block or phase differences across repeated known symbols).
3. **Fine tracking**:
   - Carrier phase & frequency tracking (Costas loop or decision-directed PLL).
   - Symbol timing recovery (timing DMA / DLL / Mueller & Müller / Gardner).
4. **Symbol demodulation**:
   - Matched filtering over symbol interval.
   - Phase rotation removal using current carrier phase estimate.
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

- `g[n]` is matched filter pulse (conjugate of transmit pulse), e.g., rectangular (1) or RRC samples. For normalized energy, scale `g` so that signal amplitude is consistent.
- If `g[n] = 1` (rectangular), `Y` is simply the sum of samples over the symbol.

### 3.2 Carrier removal (mixing)

Before matched filtering, mix the input by the estimated carrier:

\[
\tilde r[n] = r[n] \cdot e^{-j(2\pi \hat f_\Delta n T_s + \hat\phi_n)}
\]

where `\hat f_\Delta` is current estimate and `\hat\phi_n` is per-sample phase correction from PLL. Typically, mixing is applied per-sample or by phase accumulator.

### 3.3 Decision rule (hard)

The ML decision for M-PSK in AWGN (coherent) is:

\[
\hat m = \arg\min_m |Y - A e^{j 2\pi m / M}|^2
\quad\Longrightarrow\quad
\hat m = \arg\max_m \Re\{Y e^{-j 2\pi m / M}\}
\]

Equivalently, compute `θ_Y = arg(Y)` and quantize:

\[
\hat m = \left\lfloor \frac{M}{2\pi} \big( \operatorname{wrap}(θ_Y) \big) \right\rceil
\]

(Where wrap maps to `(-π, π]` and quantization rounds to nearest constellation angle.)

### 3.4 Soft metrics (LLR)

For bit-level soft decisions (for FEC), compute LLR for bit `b_i` of the symbol:

\[
LLR(b_i) \approx \ln\frac{\sum_{m \in S_1} \exp\!\big(\frac{|Y|^2 + 2\Re\{Y A e^{-j 2\pi m / M}\}}{σ^2}\big)}{\sum_{m \in S_0} \exp\!\big(\frac{|Y|^2 + 2\Re\{Y A e^{-j 2\pi m / M}\}}{σ^2}\big)}
\]

For practical implementation, approximate with max-log:

\[
LLR(b_i) \approx \frac{1}{σ^2}\big( \max_{m\in S_1} \Re\{Y A e^{-j 2\pi m / M}\} - \max_{m\in S_0} \Re\{Y A e^{-j 2\pi m / M}\} \big)
\]

where `S_1`/`S_0` are symbol sets with bit `i` = 1/0.

---

## 4. Carrier recovery (PLL / Costas loop)

M-PSK requires phase lock. Use a decision–directed Costas loop generalization for M-PSK.

### 4.1 Phase error detector (PED)

A general PED for M-PSK:

1. Compute matched filter output `Y`.
2. Rotate `Y` by `-arg(Y_ref)` (or compare to nearest constellation point `s_hat = e^{j2π \hat m / M}`).
3. Error estimate:

- **Decision-directed**:
  \[
  e_\phi = \Im\{ s_{\hat m}^* \cdot Y \}
  \]
  This is proportional to sine of phase error.

- **Blind (for acquisition)**:
  - Use non-linear transformation, e.g., raise `Y` to power `M` and extract phase:
    \[
    e_\phi = \frac{1}{M}\arg(Y^M)
    \]
  - This removes constellation ambiguity during coarse acquisition (helps resolve carrier phase of order 2π/M).

### 4.2 Loop update

Discrete-time loop integrator form:

\[
\omega[n+1] = \omega[n] + K_i \, e_\phi[n]
\]
\[
\theta[n+1] = \theta[n] + K_p \, e_\phi[n] + \omega[n+1]
\]

- `θ` is phase estimate; `ω` is frequency integrator.
- `K_p`, `K_i` chosen per desired loop bandwidth and damping.
- Use first-order or second-order loops tuned with control-theory formulars (natural frequency, damping).

### 4.3 Handling phase ambiguity

- M-PSK has `M`-fold phase ambiguity after PLL acquisition if using blind methods. Resolve ambiguity via:
  - Preamble with known symbol phases (pilot symbols).
  - Differential encoding at transmitter (DPSK): receiver can decode without absolute phase.
  - Data-aided estimation: initial preamble provides absolute alignment.

---

## 5. Timing recovery (symbol synchronization)

Goal: align integration windows to symbol boundaries and correct fractional timing.

### 5.1 Approaches

- **Non-data-directed**: Gardner TED, Mueller & Müller algorithms (work for many modulations).
- **Data-directed**: use known preamble symbols to compute timing offset via correlation; then switch to tracking.
- **Decision-directed**: use decisions to compute early/late energy error.

### 5.2 Gardner TED (for complex baseband)

For oversampled signal (samples per symbol `Ns` > 2):

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

- Use a known preamble sequence of symbols (e.g., specific phases) to correlate and detect frame boundaries.
- Slide a window of `L_preamble * N` samples; compute correlation metric:

\[
S(\tau) = \left| \sum_{k=0}^{L_p-1} Y(\tau + kN) \cdot s_{preamble}[k]^* \right|
\]

where `Y(·)` is matched-filter output at candidate symbol offsets. Peak exceeding threshold → acquisition.

### 6.2 Coarse CFO estimation

- Methods:
  - FFT of long block to find peak frequency shift.
  - Use phase differences across repeated preamble symbols:

\[
\hat f_\Delta = \frac{1}{2\pi T_{rep}} \angle\bigg( \sum_{n\in W} r[n] r^*[n + T_{rep,\text{samples}}] \bigg)
\]

- Apply coarse correction (multiply `r[n]` by `e^{-j2π \hat f_\Delta n T_s}`) and initialize PLL.

---

## 7. Symbol decision & mapping

- After timing & carrier correction, compute `Y` per symbol. Decision:
  - Hard: quantize `θ_Y = arg(Y)` to nearest constellation angle `2π m / M`.
  - Soft: compute LLRs per bit using max-log approx from `Re{Y s_m^*}` metrics.
- Gray mapping reduces bit errors for adjacent symbol errors.

---

## 8. Frame & CRC handling

- Use preamble for sync, then fixed-length payload + CRC.
- On decode, verify CRC; on failure, allow small timing adjustments (±1 sample) and re-try; else re-enter acquisition.
- Maintain counters for packet loss, SNR metrics, and re-acquisition stats.

---

## 9. Structs / API (abstract)

### 9.1 `CoherentMPSKDecoder` (stateful)

**State fields**
- `Fs`, `T_sym`, `N` (samples/symbol)
- `M`, `constellation` (complex phasors `s_m = e^{j2π m / M}`)
- `ring_buffer` (complex); buffer length ≥ `L_acq*N + margin`
- `state` ∈ {SEARCH, ACQUIRE, TRACK}
- **Timing state**:
  - `timing_phase` (fractional offset), `sample_index`
  - interpolator coefficients (for fractional delays)
- **Carrier PLL state**:
  - `theta`, `omega`, `Kp`, `Ki`
- **Acquisition**:
  - `preamble` (symbol array), `acq_threshold`, `acq_window`
- **Noise estimator**: `noise_var`, SNR estimator
- **Decoder**:
  - `frame_assembler`, `soft_buffer`, CRC checker
- **Metrics**: SNR, correlation peaks, bit error counters

**Primary methods**
- `Init(params)` — sets rates, constellation, loop gains, preamble
- `FeedSamples(samples[])` — push to ring buffer and run `ProcessStream()`
- `ProcessStream()` — handles state machine: SEARCH→ACQ→TRACK
- `ComputeMatchedFilter(start)` — compute `Y` for window at start (using interpolation if needed)
- `CarrierLoopUpdate(e_phi)` — update `theta`, `omega`
- `TimingLoopUpdate(e_t)` — update `timing_phase`
- `GetDecodedFrames()` — return collected frames
- `Reset()`

### 9.2 `CoherentMPSKEncoder` (optional)

**State**
- `Fs`, `T_sym`, `N`, `M`
- `constellation` phasors
- `symbol_queue`
- `phase_accumulator` (for continuous phase if desired)
- `pulse_shape` (RRC etc.)

**Methods**
- `Init(params)`
- `FeedBits(bits[])` → converts to symbols and queues
- `NextSamples(n)` → generate `n` complex samples (apply pulse shaping & carrier)
- `Reset()`

---

## 10. Numerical & loop tuning guidance

- PLL loop bandwidth `B` typically 1–5% of symbol rate for low Doppler; increase as needed for dynamics. Use 2nd-order loop formulas to set `Kp`, `Ki` for `ζ≈0.707`.
- Timing loop bandwidth smaller than PLL; use lower gains to avoid interaction.
- Symbol integration length `N` large gives noise averaging but reduces agility; choose `N` per system bandwidth & channel.

---

## 11. Implementation details & optimizations

- **Matched-filter efficiency**: precompute `g[n]` and use vectorized dot-product for `Y`.
- **Carrier mixing**: implement phase accumulator `φ[n+1] = φ[n] + 2π (f_center + f_est) / F_s` and compute `e^{-jφ[n]}` efficiently with recursive trig or complex multiply (avoid expensive `exp` calls).
- **Interpolation**: implement Farrow structure for fractional sample delays, or polyphase filterbank if oversampling factor small.
- **Soft outputs**: compute LLRs with max-log approximation for FEC decoders (LDPC, turbo).
- **Complexity**: per-symbol cost dominated by N mul-adds for matched filter; optimize with SIMD or use FFT-based correlation if large N and many channels.

---

## 12. Acquisition & performance considerations

- Preamble length tradeoff: longer preamble → faster acquisition at low SNR but overhead increases.
- CFO tolerance before demod: ensure coarse CFO estimator handles expected Doppler range; otherwise noncoherent fallback or larger acquisition FFT needed.
- Phase noise and residual frequency error degrade performance—estimate and compensate promptly.

---

## 13. Handling impairments

- **Large CFO**: perform coarse FFT-based search across freq bins, correct, then PLL.
- **Phase ambiguity**: if no preamble/pilot, use differential encoding or constellation rotation detection with pilot symbols.
- **Multipath**: if severe, implement equalizer (MMSE) using known pilots or sequence-based channel estimation.
- **Burst interference**: detect using energy threshold; mute decoder or hold previous phase/timing for short bursts.

---

## 14. Testing & validation

- **Unit tests**: matched filter with noiseless tone; PLL step response; timing loop convergence on synthetic offsets.
- **AWGN tests**: BER vs Eb/N₀ curves for chosen M; compare to theoretical coherent M-PSK curves.
- **CFO & Doppler tests**: inject offsets ±X Hz and verify acquisition & tracking.
- **Real data tests**: capture real IQ data from transmitter, run decoder end-to-end.
- **Stress tests**: buffer wrap, reconnection, re-acquisition under packet loss.

---

## 15. Example parameter sets (starter)

- **QPSK (4-PSK)**: `R_sym = 1e3` sps, `Fs = 16e3` → `N = 16`. PLL BW = 0.01·R_sym (10 Hz). Preamble 64 symbols.
- **8-PSK**: same sampling, but expect ~6–8 dB more Eb/N₀ for same BER. Use longer preamble (128 symbols) in low-SNR.
- **16-PSK**: large sensitivity to phase noise; prefer using QAM instead if amplitude precision is available.

---

## 16. Deliverables for engineer

- Implement `CoherentMPSKDecoder` with API described above. Maintain state for timing & carrier loops.
- Include acquisition module (preamble-based), coarse CFO estimator, PLL & timing loops, matched filter demodulator, and soft/hard decision outputs.
- Provide testbench to simulate AWGN, CFO, timing offset, phase noise; generate BER curves and acquisition statistics.
- Provide parameter set recommendations and tuning guide.

---

## 17. References (conceptual)

- Digital Communications theory (coherent M-PSK performance)  
- Costas loop and general M-PSK phase detectors  
- Gardner / Mueller & Müller timing recovery  
- Matched filtering and Nyquist pulse shaping (RRC)  
- Soft metrics and max-log LLR approximations

---

### End of specification
