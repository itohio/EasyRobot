# Spec: Coherent M-QAM Demodulator — Software Implementation (Engineering / Mathematical)

## 0. Scope & assumptions

- Input: complex baseband samples `r[n]` sampled at `F_s` (samples/sec). If ADC is real passband, assume an external front-end downconverts to complex IQ or that a software downconversion stage exists.
- Transmitter: M-QAM modulation with constellation size `M` (M = 4, 16, 64, 256, ...). Gray bit mapping assumed for bit error performance.
- Symbols: symbol period `T_sym`, symbol rate `R_sym = 1/T_sym`. Samples per symbol `N = round(F_s * T_sym)`. Prefer integer `N`; otherwise use resampling or fractional-timing interpolation.
- Channel model: AWGN + possible carrier frequency offset (CFO) and phase noise, possible slight Doppler, and multipath/ISI if specified.
- Coherent demodulation requires carrier phase recovery and symbol timing recovery. QAM is more sensitive to phase errors than PSK due to amplitude information.
- Constellation: typically square (M = 4, 16, 64, 256) or cross (M = 32, 128) constellations. Square constellations are most common.

---

## 1. Signal model

Transmitted (complex baseband) for symbol index `k`, symbol value `s[k] = a_k + j b_k` corresponding to constellation point:

\[
s(t) = (a_k + j b_k) \cdot A, \quad t \in [kT_{sym},(k+1)T_{sym})
\]

where:
- `a_k` = in-phase (I) component ∈ {±1, ±3, ±5, ...} for square constellations
- `b_k` = quadrature (Q) component ∈ {±1, ±3, ±5, ...} for square constellations
- `A` = amplitude scaling factor
- `M = L²` for square constellations, where `L` is the number of levels per dimension

For square M-QAM (M = L²):
- `L = √M` levels per dimension
- I and Q components independently take values from: `{-(L-1), -(L-3), ..., -1, +1, ..., +(L-3), +(L-1)}`
- Normalized constellation: scale so that average power = 1

Example (16-QAM, L=4):
- I, Q ∈ {-3, -1, +1, +3}
- Normalization: scale by `1/√10` to achieve average power = 1

Receiver observes with CFO `f_Δ`, phase noise `φ(t)`, and AWGN `w(t)`:

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
- `Y` is complex: `Y = Y_I + j Y_Q`

### 3.2 Carrier removal (mixing)

Before matched filtering, mix the input by the estimated carrier:

\[
\tilde r[n] = r[n] \cdot e^{-j(2\pi \hat f_\Delta n T_s + \hat\phi_n)}
\]

where `\hat f_Δ` is current estimate and `\hat\phi_n` is per-sample phase correction from PLL. Typically, mixing is applied per-sample or by phase accumulator.

### 3.3 Decision rule (hard)

The ML decision for M-QAM in AWGN (coherent) is:

\[
\hat m = \arg\min_m |Y - s_m|^2
\]

where `s_m = a_m + j b_m` is the m-th constellation point.

For square constellations, this simplifies to independent decisions on I and Q:

\[
\hat a = \arg\min_{a \in \mathcal{A}} |Y_I - a \cdot A|^2
\]
\[
\hat b = \arg\min_{b \in \mathcal{B}} |Y_Q - b \cdot A|^2
\]

where `\mathcal{A}` and `\mathcal{B}` are the sets of possible I and Q values.

**Quantization approach**:
1. Normalize: `Y_I' = Y_I / A`, `Y_Q' = Y_Q / A`
2. Quantize to nearest level: `\hat a = \text{quantize}(Y_I')`, `\hat b = \text{quantize}(Y_Q')`
3. Map `(\hat a, \hat b)` to symbol index `m`

**Quantization function** (for L levels per dimension):
- Levels: `\{-(L-1), -(L-3), ..., -1, +1, ..., +(L-3), +(L-1)\}`
- Quantize `x` to nearest level:
  \[
  \hat{x} = \begin{cases}
    -(L-1) & \text{if } x < -(L-2) \\
    \text{round to odd integer in } [-(L-2), (L-2)] & \text{if } |x| \leq (L-2) \\
    +(L-1) & \text{if } x > (L-2)
  \end{cases}
  \]

### 3.4 Soft metrics (LLR)

For bit-level soft decisions (for FEC), compute LLR for bit `b_i` of the symbol:

\[
LLR(b_i) \approx \ln\frac{\sum_{m \in S_1} \exp\!\big(\frac{-|Y - s_m|^2}{σ^2}\big)}{\sum_{m \in S_0} \exp\!\big(\frac{-|Y - s_m|^2}{σ^2}\big)}
\]

For practical implementation, approximate with max-log:

\[
LLR(b_i) \approx \frac{1}{σ^2}\big( \min_{m\in S_0}|Y - s_m|^2 - \min_{m\in S_1}|Y - s_m|^2 \big)
\]

where `S_1`/`S_0` are symbol sets with bit `i` = 1/0.

For square constellations, I and Q are independent, so LLRs can be computed separately for I and Q bits.

---

## 4. Carrier recovery (PLL / Costas loop)

M-QAM requires phase lock. Use a decision-directed Costas loop.

### 4.1 Phase error detector (PED)

A general PED for M-QAM:

1. Compute matched filter output `Y`.
2. Make hard decision: `\hat s = \arg\min_m |Y - s_m|^2`
3. Error estimate:

- **Decision-directed**:
  \[
  e_\phi = \Im\{ \hat s^* \cdot Y \}
  \]
  This is proportional to sine of phase error.

- **Blind (for acquisition)**:
  - Use constellation rotation detection:
    \[
    e_\phi = \Im\{ Y^4 \} \quad \text{(for square constellations)}
    \]
  - This removes 4-fold phase ambiguity for square QAM.

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
- Use first-order or second-order loops tuned with control-theory formulas (natural frequency, damping).

### 4.3 Handling phase ambiguity

- Square M-QAM has 4-fold phase ambiguity after PLL acquisition if using blind methods. Resolve ambiguity via:
  - Preamble with known symbol phases (pilot symbols).
  - Differential encoding at transmitter (DQAM): receiver can decode without absolute phase.
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

- Use a known preamble sequence of symbols (e.g., specific constellation points) to correlate and detect frame boundaries.
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

- Apply coarse correction (multiply `r[n]` by `e^{-j2π \hat f_Δ n T_s}`) and initialize PLL.

---

## 7. Symbol decision & mapping

- After timing & carrier correction, compute `Y` per symbol. Decision:
  - Hard: quantize `Y_I` and `Y_Q` independently to nearest constellation levels.
  - Soft: compute LLRs per bit using max-log approx from `|Y - s_m|²` metrics.
- Gray mapping reduces bit errors for adjacent symbol errors.

**Gray mapping for square QAM**:
- For each dimension (I or Q), use Gray code for the levels.
- Example (4-QAM / QPSK): I, Q ∈ {-1, +1}
  - Level -1 → bit 0
  - Level +1 → bit 1
- Example (16-QAM): I, Q ∈ {-3, -1, +1, +3}
  - Level -3 → bits 00
  - Level -1 → bits 01
  - Level +1 → bits 10
  - Level +3 → bits 11
  - (Gray code: 00, 01, 11, 10)

---

## 8. Frame & CRC handling

- Use preamble for sync, then fixed-length payload + CRC.
- On decode, verify CRC; on failure, allow small timing adjustments (±1 sample) and re-try; else re-enter acquisition.
- Maintain counters for packet loss, SNR metrics, and re-acquisition stats.

---

## 9. Structs / API (abstract)

### 9.1 `CoherentMQAMDecoder` (stateful)

**State fields**
- `Fs`, `T_sym`, `N` (samples/symbol)
- `M`, `L` (constellation size, levels per dimension)
- `constellation` (complex points `s_m = a_m + j b_m`)
- `I_levels`, `Q_levels` (arrays of possible I/Q values)
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
- **AGC state**: amplitude estimate for normalization
- **Decoder**:
  - `frame_assembler`, `soft_buffer`, CRC checker
- **Metrics**: SNR, correlation peaks, bit error counters

**Primary methods**
- `Init(params)` — sets rates, constellation, loop gains, preamble
- `FeedSamples(samples[])` — push to ring buffer and run `ProcessStream()`
- `ProcessStream()` — handles state machine: SEARCH→ACQ→TRACK
- `ComputeMatchedFilter(start)` — compute `Y` for window at start (using interpolation if needed)
- `QuantizeIQ(Y_I, Y_Q)` — quantize I and Q to nearest constellation levels
- `CarrierLoopUpdate(e_phi)` — update `theta`, `omega`
- `TimingLoopUpdate(e_t)` — update `timing_phase`
- `GetDecodedFrames()` — return collected frames
- `Reset()`

### 9.2 `CoherentMQAMEncoder` (stateful)

**State**
- `Fs`, `T_sym`, `N`, `M`, `L`
- `constellation` points
- `I_levels`, `Q_levels`
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
- AGC: QAM requires accurate amplitude estimation. Use moving average or exponential filter on `|Y|²` to estimate signal power.

---

## 11. Implementation details & optimizations

- **Matched-filter efficiency**: precompute `g[n]` and use vectorized dot-product for `Y`.
- **Carrier mixing**: implement phase accumulator `φ[n+1] = φ[n] + 2π (f_center + f_est) / F_s` and compute `e^{-jφ[n]}` efficiently with recursive trig or complex multiply (avoid expensive `exp` calls).
- **Interpolation**: implement Farrow structure for fractional sample delays, or polyphase filterbank if oversampling factor small.
- **Soft outputs**: compute LLRs with max-log approximation for FEC decoders (LDPC, turbo).
- **Quantization**: for square constellations, I and Q quantization can be done independently (simpler than full constellation search).
- **Constellation lookup**: precompute constellation points and use lookup table for quantization.
- **Complexity**: per-symbol cost dominated by N mul-adds for matched filter; optimize with SIMD or use FFT-based correlation if large N and many channels.

---

## 12. Acquisition & performance considerations

- Preamble length tradeoff: longer preamble → faster acquisition at low SNR but overhead increases.
- CFO tolerance before demod: ensure coarse CFO estimator handles expected Doppler range; otherwise noncoherent fallback or larger acquisition FFT needed.
- Phase noise and residual frequency error degrade performance—estimate and compensate promptly.
- AGC: QAM is sensitive to amplitude errors. Ensure AGC converges before symbol decisions.

---

## 13. Handling impairments

- **Large CFO**: perform coarse FFT-based search across freq bins, correct, then PLL.
- **Phase ambiguity**: if no preamble/pilot, use differential encoding or constellation rotation detection with pilot symbols.
- **Amplitude errors**: use AGC to normalize received signal amplitude. QAM is more sensitive to amplitude errors than PSK.
- **Multipath**: if severe, implement equalizer (MMSE) using known pilots or sequence-based channel estimation.
- **Burst interference**: detect using energy threshold; mute decoder or hold previous phase/timing for short bursts.

---

## 14. Testing & validation

- **Unit tests**: matched filter with noiseless constellation point; PLL step response; timing loop convergence on synthetic offsets.
- **AWGN tests**: BER vs Eb/N₀ curves for chosen M; compare to theoretical coherent M-QAM curves.
- **CFO & Doppler tests**: inject offsets ±X Hz and verify acquisition & tracking.
- **AGC tests**: verify amplitude normalization under varying signal levels.
- **Real data tests**: capture real IQ data from transmitter, run decoder end-to-end.
- **Stress tests**: buffer wrap, reconnection, re-acquisition under packet loss.

---

## 15. Example parameter sets (starter)

- **4-QAM (QPSK)**: `R_sym = 1e3` sps, `Fs = 16e3` → `N = 16`. PLL BW = 0.01·R_sym (10 Hz). Preamble 64 symbols. I, Q ∈ {-1, +1}.
- **16-QAM**: same sampling, but expect ~4 dB more Eb/N₀ for same BER. Use longer preamble (128 symbols) in low-SNR. I, Q ∈ {-3, -1, +1, +3}, normalized by `1/√10`.
- **64-QAM**: expect ~6–8 dB more Eb/N₀ than 16-QAM. I, Q ∈ {-7, -5, -3, -1, +1, +3, +5, +7}, normalized by `1/√42`.
- **256-QAM**: very sensitive to phase and amplitude errors; requires excellent synchronization. I, Q ∈ {-15, -13, ..., -1, +1, ..., +13, +15}, normalized by `1/√170`.

---

## 16. Constellation normalization

For square M-QAM with `L = √M` levels per dimension:

**Average power calculation**:
\[
P_{avg} = \frac{1}{M} \sum_{m=0}^{M-1} |s_m|^2 = \frac{1}{L^2} \sum_{a \in \mathcal{A}} \sum_{b \in \mathcal{B}} (a^2 + b^2)
\]

For symmetric constellation with levels `\{±1, ±3, ..., ±(L-1)\}`:
\[
P_{avg} = \frac{2}{3}(L^2 - 1)
\]

**Normalization factor**:
\[
A_{norm} = \frac{1}{\sqrt{P_{avg}}} = \sqrt{\frac{3}{2(L^2 - 1)}}
\]

Examples:
- 4-QAM (L=2): `A_norm = 1/√2`
- 16-QAM (L=4): `A_norm = 1/√10`
- 64-QAM (L=8): `A_norm = 1/√42`
- 256-QAM (L=16): `A_norm = 1/√170`

---

## 17. Deliverables for engineer

- Implement `CoherentMQAMDecoder` with API described above. Maintain state for timing & carrier loops.
- Include acquisition module (preamble-based), coarse CFO estimator, PLL & timing loops, matched filter demodulator, and soft/hard decision outputs.
- Implement AGC for amplitude normalization.
- Provide testbench to simulate AWGN, CFO, timing offset, phase noise, amplitude errors; generate BER curves and acquisition statistics.
- Provide parameter set recommendations and tuning guide.

---

## 18. References (conceptual)

- Digital Communications theory (coherent M-QAM performance)  
- Costas loop and general M-QAM phase detectors  
- Gardner / Mueller & Müller timing recovery  
- Matched filtering and Nyquist pulse shaping (RRC)  
- Soft metrics and max-log LLR approximations
- AGC (Automatic Gain Control) for amplitude normalization

---

### End of specification

