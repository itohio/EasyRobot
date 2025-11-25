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

---

## 18. Theoretical SNR Limits and Practical Margins

### 18.1 Coherent M-FSK SNR Requirements

For coherent M-FSK detection, the theoretical minimum Eb/N0 (energy per bit to noise power spectral density ratio) depends on the modulation order M and target BER:

**Theoretical Eb/N0 (dB) for BER = 10^-5:**
- 2-FSK (M=2): ~12.5 dB
- 4-FSK (M=4): ~14.5 dB  
- 8-FSK (M=8): ~16.5 dB
- 16-FSK (M=16): ~18.5 dB

**Implementation Margins:**
Due to implementation losses (non-ideal detection, timing errors, phase noise, etc.), practical systems require additional margin:
- **Standard margin**: 2.5 dB (recommended for 2-FSK, 4-FSK)
- **High M-FSK margin**: 3-5 dB (required for 8-FSK, 16-FSK due to frequency resolution limits)

**Practical Minimum SNR (dB) for BER = 10^-5:**
- 2-FSK: ~15 dB (theoretical 12.5 + 2.5 margin)
- 4-FSK: ~17 dB (theoretical 14.5 + 2.5 margin)
- 8-FSK: ~20-22 dB (theoretical 16.5 + 3-5 margin, higher due to Goertzel frequency resolution limits)
- 16-FSK: ~22-24 dB (theoretical 18.5 + 3-5 margin)

**Note on 8-FSK and Higher:**
For 8-FSK (256 tones) and above, the current non-coherent Goertzel-based detection has limitations:
- Tone spacing becomes very small (e.g., 109.59 Hz for 8-FSK with 96 kHz sample rate)
- Frequency resolution of Goertzel algorithm may not be sufficient with noise
- Requires higher SNR margins or coherent detection with proper phase tracking
- Current implementation may require 20+ dB SNR for reliable 8-FSK operation

**SNR Conversion:**
The relationship between Eb/N0 and SNR depends on bandwidth:
- SNR (dB) ≈ Eb/N0 (dB) - 10*log10(BitsPerSymbol) + ProcessingGain
- Processing gain from symbol integration: ~10*log10(SamplesPerSymbol)

### 18.2 Test Scenarios and Expected Performance

**Roundtrip Tests:**
- **Baseline (no noise)**: All M-FSK configurations should pass
- **2-FSK with noise**: Passes at 15+ dB SNR
- **4-FSK with noise**: Passes at 17+ dB SNR  
- **8-FSK with noise**: Requires 20+ dB SNR with current implementation

**Limitations:**
- Current implementation uses non-coherent Goertzel detection
- 8-FSK and higher may require coherent detection for optimal performance
- Frequency resolution limits affect high M-FSK performance

---

## 19. Current Implementation Status

### 18.1 What's Implemented

**Encoder (`mfsk/encoder.go`):**
- ✅ M-FSK modulation with configurable frequencies (2, 4, 8, 16-FSK)
- ✅ Phase-continuous tone generation (phase accumulator)
- ✅ Bit-to-symbol mapping (MSB first)
- ✅ Padding of incomplete symbols with zeros
- ❌ No preamble generation (tests control preamble)

**Decoder (`mfsk/decoder.go`):**
- ✅ Non-coherent detection using Goertzel algorithm
- ✅ Energy-based signal detector (`SignalDetector`) for acquisition
- ✅ Simple symbol synchronizer (fixed sample-per-symbol buffering)
- ✅ AGC (Automatic Gain Control)
- ✅ Bit extractor for symbol-to-byte conversion
- ✅ Low-pass filter (optional, currently disabled for clean signals)
- ❌ No PLL for carrier phase/frequency tracking
- ❌ No symbol timing recovery (early/late correlators)
- ❌ No CFO (Carrier Frequency Offset) estimation
- ❌ No coherent correlators (using Goertzel instead of complex mixing)

**Current Architecture:**
```
Input → AGC → SignalDetector → SymbolSynchronizer → GoertzelFrequencyDetector → BitExtractor → Bytes
```

**Issues with Current Implementation:**
1. **8-FSK/16-FSK failures**: Goertzel may not have sufficient frequency resolution for closely spaced frequencies (250Hz for 8-FSK, 100Hz for 16-FSK with 48 samples/symbol)
2. **No phase tracking**: Non-coherent detection loses ~3dB performance vs coherent
3. **No timing recovery**: Fixed sample-per-symbol assumption breaks with timing jitter
4. **No CFO compensation**: Frequency offset causes detection errors
5. **Simple acquisition**: Energy-based detection doesn't align to symbol boundaries

### 18.2 What Needs to Be Implemented

**For Proper Coherent Detection:**

1. **PLL (Phase-Locked Loop)** for carrier tracking:
   - Track residual phase after coarse CFO correction
   - Update based on detected symbol's phase
   - Use PI controller (proportional + integral)

2. **Symbol Timing Recovery**:
   - Early/late correlators or Gardner algorithm
   - Adjust symbol boundary timing dynamically
   - Handle fractional sample timing

3. **Coherent Correlators**:
   - Replace Goertzel with complex mixing: `Z_k = Σ r[n] * e^(-j2πf_k nT_s)`
   - Use precomputed LUTs for sin/cos (embedded-friendly)
   - Phase-align for coherent gain

4. **Acquisition & Synchronization**:
   - Preamble-based symbol boundary detection
   - Coarse CFO estimation from preamble
   - Proper alignment to symbol start

---

## 19. Proposed Embedded-Friendly Implementation

### 19.1 Design Principles for Embedded

- **No allocations**: Pre-allocate all buffers at initialization
- **Small LUTs**: Precompute sin/cos tables for correlators (max 256 entries)
- **Fixed-point optional**: Use float32 for simplicity, optimize later if needed
- **Ring buffers**: Use circular indexing for sample buffers
- **State machines**: Clear state transitions (SEARCH → ACQUIRE → TRACK)

### 19.2 Proposed Architecture

```
Input → AGC → SignalDetector → Acquisition → PLL → TimingRecovery → CoherentCorrelators → BitExtractor → Bytes
```

**State Machine:**
- **SEARCH**: Energy-based detection, look for preamble
- **ACQUIRE**: Coarse CFO estimation, symbol boundary alignment
- **TRACK**: Fine PLL/timing tracking, coherent detection

### 19.3 Key Components to Add

#### 19.3.1 PLL (Phase-Locked Loop)

**Location**: `x/math/filter/pll/` (shared component for M-FSK, M-PSK, QAM, M-ASK)

```go
// From x/math/filter/pll/pll.go
type Filter struct {
    kp         float32  // Proportional gain
    ki         float32  // Integral gain
    phase      float32  // Current phase estimate (radians)
    freqOffset float32  // Frequency offset accumulator (radians/symbol)
    symbolRate float32  // Symbol rate for frequency conversion
}
```

**Usage:**
```go
import "github.com/itohio/EasyRobot/x/math/filter/pll"

// Create PLL filter with loop bandwidth
filter := pll.NewFromLoopBandwidth(loopBandwidth, symbolRate, 0.707)

// Update with phase error from correlator
phaseError := math.Atan2(imag(correlatorOutput), real(correlatorOutput))
filter.Update(phaseError)

// Apply phase correction
correctedI, correctedQ := filter.RotatePhase(i, q)
```

**Update rule:**
- Phase error: `ε = arg(Z_k)` where `Z_k` is correlator output for detected symbol
- Frequency update: `freqOffset += ki * ε`
- Phase update: `phase += kp * ε + freqOffset`

#### 19.3.2 Symbol Timing Recovery

```go
type TimingRecovery struct {
    // Early/late correlators
    earlyOffset int     // Samples before nominal
    lateOffset  int     // Samples after nominal
    timingPhase float32 // Fractional sample offset
    
    // DLL (Delay-Locked Loop) state
    timingError float32
    kp, ki      float32
    
    // Pre-allocated buffers (no allocations)
    earlyBuffer []float32
    lateBuffer  []float32
}
```

**Update rule:**
- Early correlator: `E = |Z_k(early)|`
- Late correlator: `L = |Z_k(late)|`
- Timing error: `ε_t = (E - L) / (E + L)`
- Update timing phase: `timingPhase += kp * ε_t`

#### 19.3.3 Coherent Correlators with LUTs

```go
type CoherentCorrelator struct {
    frequencies []float32
    sampleRate  float32
    samplesPerSymbol int
    
    // Precomputed LUTs (embedded-friendly, no allocations)
    // sinLUT[i] = sin(2π * freq_k * i * T_s) for i=0..N-1, k=0..M-1
    // cosLUT[i] = cos(2π * freq_k * i * T_s)
    sinLUT [][]float32  // [freq_idx][sample_idx]
    cosLUT [][]float32
    
    // Or use phase accumulator approach (more memory efficient)
    phaseIncrements []float32  // 2π * freq_k / sampleRate
}
```

**Correlation computation:**
```go
// For symbol samples r[n], compute Z_k = Σ r[n] * e^(-j2πf_k nT_s)
// Using LUTs:
for n := range samples {
    idx := (n + timingOffset) % len(sinLUT[k])
    I += samples[n] * cosLUT[k][idx]
    Q += samples[n] * sinLUT[k][idx]  // Note: -sin for e^(-j...)
}
Z_k = complex(I, -Q)  // e^(-jθ) = cos(θ) - j*sin(θ)
```

**Memory-efficient alternative (phase accumulator):**
```go
// No LUTs, compute on-the-fly (slower but no memory)
for n := range samples {
    phase := 2 * math.Pi * freq_k * float32(n) / sampleRate
    I += samples[n] * float32(math.Cos(float64(phase)))
    Q += samples[n] * float32(math.Sin(float64(phase)))
}
```

#### 19.3.4 Acquisition with Preamble

```go
type Acquisition struct {
    preamblePattern []int  // Expected symbol sequence
    searchWindow    int     // Samples to search (e.g., 2 symbols)
    
    // Pre-allocated correlation buffer
    correlationBuffer []float32
}
```

**Acquisition algorithm:**
1. Slide window over samples
2. For each offset, compute correlators for preamble symbols
3. Score: `S(τ) = Σ |Z_expectedTone(j)(τ + j*N)|²`
4. Find peak: `τ* = argmax S(τ)`
5. If `S(τ*) > threshold`, declare acquisition at `τ*`

### 19.4 Memory Budget (Embedded)

**Per demodulator instance:**
- Sample buffer: `2 * samplesPerSymbol` (early/late) = ~96 samples = 384 bytes
- LUTs (if used): `M * N * 2 * 4 bytes` = `8 * 48 * 2 * 4` = 3KB for 8-FSK
- Phase accumulator (no LUTs): `M * 4 bytes` = 32 bytes
- PLL state (`x/math/filter/pll`): 20 bytes (5 float32)
- Timing recovery: 16 bytes
- **Total: ~500 bytes** (with LUTs) or **~450 bytes** (phase accumulator)

**Recommendation**: Use phase accumulator for embedded (no LUTs), optimize with LUTs if CPU allows.

### 19.5 Processing Pipeline (Proposed)

```
1. SEARCH state:
   - Update SignalDetector (energy-based)
   - If energy > threshold, switch to ACQUIRE

2. ACQUIRE state:
   - Slide window, correlate with preamble
   - Find symbol boundary offset τ*
   - Estimate coarse CFO from preamble phase rotation
   - Initialize PLL: phase = 0, freqOffset = -CFO_coarse
   - Initialize TimingRecovery: timingPhase = τ*
   - Switch to TRACK

3. TRACK state:
   - For each symbol:
     a. Apply timing recovery (adjust sample window)
     b. Apply PLL correction (rotate samples by e^(-j(phase + freqOffset*t)))
     c. Compute coherent correlators Z_k for all frequencies
     d. Detect symbol: m = argmax |Z_k|
     e. Update PLL: phase error = arg(Z_m)
     f. Update TimingRecovery: early/late correlation
     g. Extract bits from symbol m
   - If signal lost, switch back to SEARCH
```

---

## 20. Implementation Decision

### 20.1 Current Status vs Required

**Current (Non-Coherent):**
- ✅ Works for BFSK (2-FSK) with clean signals
- ❌ Fails for 8-FSK/16-FSK (frequency resolution issues)
- ❌ Loses ~3dB performance vs coherent
- ✅ Simple, low CPU usage

**Required (Coherent):**
- ✅ Better frequency resolution (coherent gain)
- ✅ Handles CFO and phase drift
- ✅ Symbol timing recovery
- ✅ Better performance at low SNR
- ❌ More complex, higher CPU usage

### 20.2 Recommendation

**Phase 1: Fix Current Issues (Quick Win)**
1. Improve Goertzel frequency resolution:
   - Use more samples per symbol (increase sample rate or reduce symbol rate)
   - Or use longer integration window (2 symbols) for acquisition
2. Add simple CFO estimation:
   - Estimate from phase difference across symbols
   - Rotate samples before Goertzel

**Phase 2: Add Coherent Detection (Proper Fix)**
1. Implement PLL for phase tracking
2. Implement symbol timing recovery (early/late)
3. Replace Goertzel with coherent correlators (phase accumulator, no LUTs initially)
4. Add preamble-based acquisition

**Phase 3: Optimize (If Needed)**
1. Add sin/cos LUTs if CPU allows
2. Optimize correlator computation (SIMD if available)
3. Fine-tune PLL/timing loop gains

### 20.3 Embedded Constraints

- **No allocations**: ✅ All buffers pre-allocated
- **Small memory**: ✅ Phase accumulator approach (~450 bytes)
- **Low CPU**: ⚠️ Coherent detection is more expensive, but necessary for 8-FSK/16-FSK
- **Real-time**: ✅ Streaming processing, no blocking operations

---

## 21. Next Steps

1. **Update decoder.go** to add:
   - PLL component
   - TimingRecovery component
   - CoherentCorrelator (phase accumulator)
   - Acquisition state machine

2. **Keep backward compatibility**:
   - Current non-coherent path for BFSK (if it works)
   - New coherent path for 8-FSK/16-FSK

3. **Test incrementally**:
   - Test PLL with known phase offset
   - Test timing recovery with known timing offset
   - Test coherent correlators vs Goertzel
   - Test full pipeline with round-trip tests

---

### End of Updated Specification
