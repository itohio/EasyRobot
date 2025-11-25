# Channel Measurement System: Mathematical Specification

## Overview

A complete channel characterization system using:
1. **Synchronization tone** for measurement start detection
2. **Delay period** for receiver preparation
3. **Logarithmic chirp** with constant energy per frequency
4. **Signal processing** to extract channel impulse response (CIR)
5. **Metrics extraction**: SNR, THD, frequency response

This enables adaptive modulation scheme selection based on measured channel conditions.

---

## SYSTEM ARCHITECTURE

```
TRANSMITTER:
┌─────────────────────────────────────────────────────────────┐
│ Generate:                                                     │
│   1. Sync Tone (duration T_sync)                            │
│   2. Silence (duration T_delay)                             │
│   3. Log Chirp (duration T_chirp)                           │
│   4. Silence (duration T_post)                              │
└─────────────────────────────────────────────────────────────┘

RECEIVER:
┌─────────────────────────────────────────────────────────────┐
│ 1. Detect sync tone → timestamp t_sync                       │
│ 2. Wait for chirp → extract from [t_sync + T_sync + T_delay] │
│ 3. Correlate with reference chirp → CIR                      │
│ 4. Compute channel metrics                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## PART 1: SYNCHRONIZATION TONE

### Purpose
Provide unambiguous measurement start marker that's easy to detect in noise.

### Mathematical Definition

**Tone Signal**:
```
s_sync(t) = A_sync × cos(2πf_sync × t + φ₀),  0 ≤ t < T_sync

where:
  A_sync = tone amplitude
  f_sync = sync frequency (Hz)
  T_sync = tone duration (seconds)
  φ₀ = initial phase (typically 0)
```

**Discrete-Time** (sampled at f_s):
```
s_sync[n] = A_sync × cos(2πf_sync × n/f_s),  n = 0, 1, ..., N_sync-1

where:
  N_sync = ⌊T_sync × f_s⌋ samples
```

### Frequency Selection

**Constraint 1**: Avoid interference with measurement band
```
f_sync ∉ [f_min, f_max]

where [f_min, f_max] = measurement bandwidth
```

**Option A**: Below measurement band
```
f_sync = f_min - Δf

where Δf ≥ 2B_tone (tone bandwidth)
```

**Option B**: Above measurement band
```
f_sync = f_max + Δf
```

**Option C**: In-band pilot (if spectrum allows)
```
f_sync = (f_min + f_max) / 2
```

**Recommendation**: 
```
Use f_min/2 if measuring high frequencies
Use 2×f_max if measuring low frequencies
```

### Duration Selection

**Minimum duration** for reliable detection:
```
T_sync ≥ 3/B_tone

where B_tone = tone bandwidth ≈ 1/T_sync

Practical: T_sync = 10 to 50 ms
```

**Trade-off**:
- Longer → better detection reliability
- Shorter → faster measurement

**Energy**:
```
E_sync = A²_sync × T_sync / 2
```

---

## PART 2: DELAY PERIOD

### Purpose
Allow receiver to:
1. Complete sync tone detection
2. Set up buffers for chirp capture
3. Transition from detection to measurement mode

### Mathematical Definition

**Silence Signal**:
```
s_delay(t) = 0,  0 ≤ t < T_delay

Discrete:
  s_delay[n] = 0,  n = 0, ..., N_delay-1
  
where N_delay = ⌊T_delay × f_s⌋
```

### Duration Selection

**Minimum delay**:
```
T_delay ≥ T_processing + T_buffer_setup

where:
  T_processing = sync detection computation time
  T_buffer_setup = buffer allocation time
```

**Practical values**:
```
Real-time systems: T_delay = 5-10 ms
Non-real-time: T_delay = 1-5 ms
Safety margin: T_delay = 20 ms
```

**Recommendation**: `T_delay = 10 ms` (covers most systems)

---

## PART 3: LOGARITHMIC CHIRP SIGNAL

### Purpose
Sweep through measurement bandwidth with constant energy per frequency, enabling accurate channel impulse response measurement.

### Why Logarithmic (Not Linear)?

**Linear Chirp**:
- Constant frequency rate df/dt
- Non-uniform energy distribution in frequency
- Poor low-frequency resolution

**Logarithmic Chirp**:
- Constant energy per octave
- Uniform frequency resolution on log scale
- Better matches human/system perception
- Equal energy per frequency ratio

### Mathematical Definition

**Instantaneous Frequency** (log sweep):
```
f(t) = f_min × (f_max/f_min)^(t/T_chirp)

where:
  f_min = starting frequency (Hz)
  f_max = ending frequency (Hz)
  T_chirp = chirp duration (seconds)
  t ∈ [0, T_chirp]
```

**Logarithmic relationship**:
```
log(f(t)) = log(f_min) + t/T_chirp × log(f_max/f_min)
```

**Phase** (integral of frequency):
```
φ(t) = 2π ∫₀ᵗ f(τ) dτ

     = 2π × f_min × T_chirp/ln(f_max/f_min) × [(f_max/f_min)^(t/T_chirp) - 1]

Let L = ln(f_max/f_min):

φ(t) = 2π × f_min × T_chirp/L × [e^(L×t/T_chirp) - 1]
```

**Time-Domain Signal**:
```
s_chirp(t) = A_chirp × cos(φ(t))

           = A_chirp × cos(2π × f_min × T_chirp/L × [e^(L×t/T_chirp) - 1])

where:
  A_chirp = chirp amplitude
  L = ln(f_max/f_min)
  0 ≤ t < T_chirp
```

**Discrete-Time Implementation**:
```
For n = 0, 1, ..., N_chirp-1:

t[n] = n / f_s

L = ln(f_max / f_min)

φ[n] = 2π × f_min × T_chirp/L × [exp(L × n/(f_s × T_chirp)) - 1]

s_chirp[n] = A_chirp × cos(φ[n])

where N_chirp = ⌊T_chirp × f_s⌋
```

### Constant Energy Property

**Energy per frequency band** [f, f+Δf]:
```
Time spent in band:
  Δt = T_chirp × Δf/f / ln(f_max/f_min)

Energy in band:
  ΔE = A²_chirp × Δt / 2
      = A²_chirp × T_chirp × Δf/f / (2 × ln(f_max/f_min))

Energy per fractional bandwidth:
  ΔE/(Δf/f) = A²_chirp × T_chirp / (2 × ln(f_max/f_min))
  
This is CONSTANT (independent of f).
```

**Total Energy**:
```
E_chirp = A²_chirp × T_chirp / 2
```

### Parameter Selection

**Frequency Range**:
```
[f_min, f_max] = measurement bandwidth

Requirements:
  f_max/f_min ≥ 2 (at least one octave)
  f_min ≥ 2/T_chirp (frequency resolution limit)
  f_max ≤ f_s/2.56 (Nyquist with safety margin)
```

**Duration**:
```
Frequency resolution:
  Δf = 1/T_chirp

Number of frequency bins:
  N_bins = (f_max - f_min) / Δf = (f_max - f_min) × T_chirp

Recommendation:
  T_chirp ≥ 10/(f_max - f_min)  (for 10 Hz resolution)
  
Typical values:
  Narrowband (f_max-f_min < 1 kHz): T_chirp = 0.1-0.5 s
  Wideband (f_max-f_min > 10 kHz): T_chirp = 0.01-0.1 s
```

**Amplitude**:
```
Peak amplitude constraint:
  A_chirp ≤ A_max (system maximum)

Energy constraint:
  E_chirp = A²_chirp × T_chirp / 2

For fixed energy budget E_target:
  A_chirp = √(2 × E_target / T_chirp)
```

### Frequency Resolution

**Time-frequency relationship**:
```
At frequency f:
  Time instant: t(f) = (T_chirp/L) × ln(f/f_min)
  
where L = ln(f_max/f_min)

Frequency resolution:
  Δf = 1/T_chirp (uniform in linear frequency)
```

---

## PART 4: SYNC TONE DETECTION

### Purpose
Locate the start of measurement sequence in received signal.

### Detection Algorithm

**Matched Filter Detection**:

**Step 1: Generate Reference**
```
Reference tone (one period):
  r_ref[n] = cos(2πf_sync × n/f_s),  n = 0, ..., N_period-1

where N_period = ⌊f_s/f_sync⌋
```

**Step 2: Cross-Correlation**
```
For each time offset m in received signal x[n]:

R[m] = Σ(k=0 to N_sync-1) x[m+k] × s_sync[k]

Normalized:
R_norm[m] = R[m] / √(E_window[m] × E_sync)

where:
  E_window[m] = Σ(k=0 to N_sync-1) x²[m+k]
  E_sync = Σ(k=0 to N_sync-1) s²_sync[k]
```

**Step 3: Peak Detection**
```
Threshold:
  γ = detection threshold (e.g., 0.7-0.9)

Detection:
  m̂ = arg max_m {R_norm[m] | R_norm[m] > γ}

Timestamp:
  t_sync = m̂ / f_s
```

### Energy Detection (Alternative)

**For simple tone**:
```
Bandpass filter around f_sync:
  BPF: [f_sync - B_tone/2, f_sync + B_tone/2]

Energy in window:
  E[m] = Σ(k=0 to N_window-1) |BPF(x[m+k])|²

Detection:
  t_sync = m̂/f_s where E[m̂] > threshold
```

### Goertzel Algorithm (Efficient)

**Single-frequency DFT**:
```
For window starting at m, compute:

w_k = 2 × cos(2πf_sync/f_s)

State update:
  s₀ = 0
  s₁ = 0
  
For n = 0 to N_window-1:
  s₂ = x[m+n] + w_k × s₁ - s₀
  s₀ = s₁
  s₁ = s₂

Magnitude:
  |X(f_sync)|² = s₀² + s₁² - w_k × s₀ × s₁

Detection:
  |X(f_sync)|² > threshold
```

**Computational advantage**: O(N) vs O(N log N) for FFT

### False Alarm Control

**Probability of false alarm**:
```
For noise-only (no signal):
  R_norm ~ N(0, 1/√N_sync)

Threshold for P_fa:
  γ = Q⁻¹(P_fa) / √N_sync

where Q⁻¹ is inverse Q-function

Example:
  P_fa = 10⁻⁶
  N_sync = 1000 samples
  γ ≈ 0.15
```

**Detection probability**:
```
With signal present (SNR = ρ):
  R_norm ~ N(√ρ, 1/√N_sync)

P_d = Q((γ - √ρ)/√(1/N_sync))

For reliable detection: ρ > 10 dB
```

---

## PART 5: CHANNEL IMPULSE RESPONSE EXTRACTION

### Principle

**Chirp-based CIR measurement** uses deconvolution:
```
Received signal:
  y(t) = h(t) ⊗ s_chirp(t) + n(t)

where:
  h(t) = channel impulse response
  ⊗ = convolution
  n(t) = noise

Goal: Recover h(t) from y(t) and known s_chirp(t)
```

### Method: Frequency-Domain Deconvolution

**Step 1: Extract Chirp Segment**
```
Start index:
  n_start = ⌊(t_sync + T_sync + T_delay) × f_s⌋

End index:
  n_end = n_start + N_chirp

Received chirp:
  y_chirp[n] = y[n_start + n],  n = 0, ..., N_chirp-1
```

**Step 2: FFT of Signals**
```
Reference chirp spectrum:
  S(f) = FFT{s_chirp[n]}

Received chirp spectrum:
  Y(f) = FFT{y_chirp[n]}
```

**Step 3: Channel Transfer Function**
```
H(f) = Y(f) / S(f)

With regularization (to avoid division by zero/noise):

H_reg(f) = Y(f) × S*(f) / (|S(f)|² + ε)

where:
  S*(f) = complex conjugate of S(f)
  ε = regularization parameter (e.g., 0.01 × max|S(f)|²)
```

**Step 4: Inverse FFT**
```
h[n] = IFFT{H(f)}

Channel impulse response in time domain.
```

### Alternative: Time-Domain Correlation

**Cross-correlation method**:
```
h[m] = (1/E_chirp) × Σ(n=0 to N_chirp-1) y_chirp[n] × s_chirp[n-m]

where E_chirp = Σ s²_chirp[n]

More robust to noise, but requires longer computation.
```

### Windowing and Truncation

**Extract significant portion**:
```
Peak location:
  n_peak = arg max_n |h[n]|

Window:
  h_window[n] = h[n] × w[n]

where w[n] is window function (e.g., Hamming):
  w[n] = 0.54 - 0.46 × cos(2πn/N_window)

Centered at peak:
  n ∈ [n_peak - N_window/2, n_peak + N_window/2]
```

**CIR length selection**:
```
N_CIR = channel delay spread × f_s × safety_factor

Typical:
  Indoor: 1-10 μs → N_CIR = 50-500 samples @ 50 kHz
  Outdoor: 10-100 μs → N_CIR = 500-5000 samples @ 50 kHz
  
Safety factor = 2-3 (capture echoes)
```

---

## PART 6: CHANNEL METRICS

### 6.1 Channel Frequency Response

**Magnitude Response**:
```
|H(f)| = |FFT{h[n]}|

In dB:
  |H(f)|_dB = 20 × log₁₀(|H(f)|)
```

**Phase Response**:
```
∠H(f) = arg(FFT{h[n]})

Group Delay:
  τ_g(f) = -dφ/df = -(∠H(f₂) - ∠H(f₁))/(f₂ - f₁)
```

**Flatness Metric**:
```
Frequency band [f_min, f_max]:

Flatness = σ(|H(f)|_dB)

where σ = standard deviation

Alternatively:
  Flatness = max|H(f)|_dB - min|H(f)|_dB

Ideal: Flatness < 3 dB
Good: Flatness < 6 dB
Poor: Flatness > 10 dB
```

### 6.2 Signal-to-Noise Ratio (SNR)

**Method A: Direct from CIR**
```
Signal power (main path):
  P_signal = Σ(n=n_peak-W to n_peak+W) h²[n]

where W = window around peak (e.g., W = 10 samples)

Noise power (tail):
  P_noise = (1/N_tail) × Σ(n=n_end-N_tail to n_end) h²[n]

where n_end = end of CIR, N_tail = noise estimation window

SNR = P_signal / P_noise

SNR_dB = 10 × log₁₀(SNR)
```

**Method B: Frequency Domain**
```
In-band power:
  P_signal = (1/N_bins) × Σ(f=f_min to f_max) |H(f)|²

Out-of-band noise:
  P_noise = (1/N_oob) × Σ(f > f_max) |H(f)|²

SNR = P_signal / P_noise
```

**Method C: Residual Error**
```
Reconstructed signal:
  ŷ[n] = h[n] ⊗ s_chirp[n]

Residual:
  e[n] = y_chirp[n] - ŷ[n]

Signal power:
  P_signal = (1/N) × Σ y²_chirp[n]

Noise power:
  P_noise = (1/N) × Σ e²[n]

SNR = P_signal / P_noise
```

### 6.3 Total Harmonic Distortion (THD)

**Purpose**: Measure nonlinear distortion in channel

**Method**: Single-tone test within chirp analysis

**Step 1: Extract tone from chirp**
```
Select frequency f₀ (e.g., 1 kHz):

Time in chirp:
  t₀ = (T_chirp/L) × ln(f₀/f_min)

Extract window around t₀:
  y_tone[n] = y_chirp[n₀ + n],  n = 0, ..., N_tone-1

where:
  n₀ = ⌊t₀ × f_s⌋
  N_tone = window length (e.g., 10 periods)
```

**Step 2: FFT and harmonic identification**
```
Spectrum:
  Y_tone(f) = FFT{y_tone[n]}

Fundamental:
  P₁ = |Y_tone(f₀)|²

Harmonics:
  P₂ = |Y_tone(2f₀)|²
  P₃ = |Y_tone(3f₀)|²
  ...
  P_n = |Y_tone(n×f₀)|²
```

**Step 3: THD calculation**
```
THD = √(P₂ + P₃ + ... + P_n) / P₁

Usually compute up to 5th harmonic:
  THD = √(P₂ + P₃ + P₄ + P₅) / P₁

In dB:
  THD_dB = 20 × log₁₀(THD)

In percent:
  THD% = 100 × THD
```

**Alternative: Full-band THD**
```
Fundamental power:
  P_fund = |H(f₀)|²

Total harmonic power:
  P_harm = Σ(k=2 to k_max) |H(k×f₀)|²

THD = √(P_harm / P_fund)
```

**THD Quality Metrics**:
```
Excellent: THD < 0.1% (-60 dB)
Good: THD < 1% (-40 dB)
Acceptable: THD < 3% (-30 dB)
Poor: THD > 5% (-26 dB)
```

### 6.4 Delay Spread

**RMS Delay Spread**:
```
Mean delay:
  τ_mean = Σ(n × |h[n]|²) / Σ|h[n]|²

RMS delay spread:
  τ_rms = √[Σ((n - τ_mean)² × |h[n]|²) / Σ|h[n]|²]

Convert to time:
  τ_rms[seconds] = τ_rms[samples] / f_s
```

**Coherence Bandwidth**:
```
B_c ≈ 1 / (5 × τ_rms)

50% coherence bandwidth (more conservative):
  B_c,50 ≈ 1 / (50 × τ_rms)
```

**Interpretation**:
```
If signal bandwidth B < B_c:
  → Flat fading (frequency-non-selective)
  → Simple equalization

If B > B_c:
  → Frequency-selective fading
  → Need complex equalization (OFDM, frequency-domain EQ)
```

### 6.5 Doppler Spread (Time-Varying Channels)

**Requires multiple measurements**:
```
Measure CIR at times t₁, t₂, ..., t_N

Cross-correlation:
  R_h(Δt) = E[h(t) × h*(t + Δt)]

Doppler power spectrum:
  S_D(f) = FFT{R_h(Δt)}

Maximum Doppler frequency:
  f_D,max = maximum frequency where S_D(f) > threshold
```

**Coherence Time**:
```
T_c ≈ 1 / (2 × f_D,max)

If measurement period T_measure < T_c:
  → Channel quasi-static
  → Single measurement valid

If T_measure > T_c:
  → Channel time-varying
  → Need tracking/re-measurement
```

---

## PART 7: COMPLETE MEASUREMENT SIGNAL

### Composite Signal Structure

**Transmitted Signal**:
```
s(t) = {
  s_sync(t),                    0 ≤ t < T_sync
  0,                            T_sync ≤ t < T_sync + T_delay
  s_chirp(t - T_sync - T_delay), T_sync + T_delay ≤ t < T_sync + T_delay + T_chirp
  0,                            t ≥ T_sync + T_delay + T_chirp
}
```

**Discrete-Time**:
```
s[n] = {
  s_sync[n],                              0 ≤ n < N_sync
  0,                                       N_sync ≤ n < N_sync + N_delay
  s_chirp[n - N_sync - N_delay],         N_sync + N_delay ≤ n < N_sync + N_delay + N_chirp
  0,                                       n ≥ N_sync + N_delay + N_chirp
}

where:
  N_sync = ⌊T_sync × f_s⌋
  N_delay = ⌊T_delay × f_s⌋
  N_chirp = ⌊T_chirp × f_s⌋
```

**Total Length**:
```
N_total = N_sync + N_delay + N_chirp + N_post

where N_post = optional post-chirp silence
```

### Amplitude Normalization

**Peak amplitude**:
```
s_peak = max(max|s_sync|, max|s_chirp|)

Normalized:
  s_norm[n] = s[n] / s_peak

Ensures |s_norm[n]| ≤ 1 (prevents clipping)
```

**Energy normalization**:
```
E_total = Σ s²[n]

s_norm[n] = s[n] / √E_total

Ensures unit energy signal.
```

---

## PART 8: RECEIVER PROCESSING FLOWCHART

### Processing Pipeline

```
1. CAPTURE
   ├─ Continuous recording: x[n]
   └─ Buffer length: N_buffer ≥ N_total + margin

2. DETECT SYNC TONE
   ├─ Cross-correlation: R[m] = x ⊗ s_sync
   ├─ Peak detection: m̂ = arg max R[m]
   └─ Timestamp: t_sync = m̂/f_s

3. EXTRACT CHIRP
   ├─ Start index: n_start = m̂ + N_sync + N_delay
   ├─ End index: n_end = n_start + N_chirp
   └─ Segment: y_chirp = x[n_start:n_end]

4. COMPUTE CIR
   ├─ FFT: Y = FFT(y_chirp), S = FFT(s_chirp)
   ├─ Deconvolution: H = Y / S (with regularization)
   └─ IFFT: h = IFFT(H)

5. EXTRACT METRICS
   ├─ Frequency response: |H(f)|, ∠H(f)
   ├─ SNR: from CIR energy distribution
   ├─ THD: from harmonic content
   ├─ Delay spread: τ_rms
   └─ Coherence bandwidth: B_c

6. DECISION
   └─ Select modulation based on metrics
```

---

## PART 9: MODULATION SELECTION CRITERIA

### Decision Rules Based on Measurements

**SNR-Based Selection**:
```
if SNR > 25 dB:
  → 16-QAM or 16-ary FSK/DSSS
  → High data rate possible

else if SNR > 15 dB:
  → QPSK or 4-ary FSK/DSSS
  → Medium data rate

else if SNR > 10 dB:
  → BPSK or binary FSK/DSSS
  → Reliable but lower data rate

else:
  → Robust coding + binary modulation
  → Maximum error protection
```

**Frequency Selectivity**:
```
if B_c > 10 × B_signal:
  → Flat fading
  → Any modulation scheme OK
  → Simple time-domain equalization

else if B_c > 2 × B_signal:
  → Moderate selectivity
  → DSSS preferred (frequency diversity)
  → Multi-carrier (OFDM) acceptable

else:
  → Severe frequency selectivity
  → OFDM strongly recommended
  → Or reduce signal bandwidth
```

**THD Consideration**:
```
if THD < 1%:
  → Channel linear
  → High-order modulation safe

else if THD < 5%:
  → Moderate distortion
  → Limit to QPSK/4-ary
  → Consider pre-distortion

else:
  → Severe distortion
  → Use binary modulation only
  → Implement linearization
```

**Combined Decision Matrix**:
```
| SNR | B_c/B_sig | THD  | Recommended Modulation |
|-----|-----------|------|------------------------|
| >25 | >10       | <1%  | 16-QAM                 |
| >20 | >10       | <2%  | 8-PSK                  |
| >15 | >5        | <3%  | QPSK                   |
| >15 | <2        | <3%  | OFDM-QPSK             |
| >10 | >3        | <5%  | BPSK                   |
| >10 | <2        | <5%  | OFDM-BPSK             |
| <10 | Any       | Any  | Binary + FEC           |
```

---

## PART 10: ADAPTIVE RE-MEASUREMENT

### Re-measurement Trigger Conditions

**Time-based**:
```
Period T_remeasure:
  - Static channel: T_remeasure = 1-10 seconds
  - Slowly varying: T_remeasure = 100-500 ms
  - Fast varying: T_remeasure = 10-50 ms

Schedule:
  Measure at t = 0, T_remeasure, 2×T_remeasure, ...
```

**Performance-based**:
```
Monitor during data transmission:
  - Bit error rate (BER)
  - Frame error rate (FER)
  - SNR estimate from constellation

Trigger re-measurement if:
  BER > threshold (e.g., 10⁻³)
  or SNR_estimated < SNR_measured - margin (e.g., 3 dB)
  or FER > threshold (e.g., 10⁻²)
```

**Change detection**:
```
Track metrics over time:
  SNR(t), |H(f,t)|

Detect change:
  Δ(t) = |metric(t) - metric(t-T)|

If Δ(t) > threshold:
  → Trigger immediate re-measurement
  → Update modulation if needed
```

### Fast Re-measurement

**Shortened chirp**:
```
Use shorter T_chirp for tracking:
  T_chirp,track = T_chirp,initial / 4

Trade-off:
  - Faster measurement
  - Lower frequency resolution
  - Less accurate CIR
  
Use for: monitoring, not initial characterization
```

**Pilot-tone tracking**:
```
Instead of full chirp:
  - Insert periodic pilot tones at key frequencies
  - Monitor |H(f_pilot)|
  - Track SNR and phase
  
Advantages:
  - Minimal overhead
  - Continuous monitoring
  - Fast adaptation
```

---

## PART 11: PRACTICAL IMPLEMENTATION PARAMETERS

### Example Parameter Set (Acoustic Channel, 1-10 kHz)

```
SYNC TONE:
  f_sync = 500 Hz (below measurement band)
  T_sync = 50 ms
  A_sync = 0.5 (normalized)
  
DELAY:
  T_delay = 10 ms
  
CHIRP:
  f_min = 1000 Hz
  f_max = 10000 Hz
  T_chirp = 100 ms (10 Hz resolution)
  A_chirp = 0.5
  
SAMPLING:
  f_s = 48000 Hz (standard audio rate)
  
BUFFER:
  N_sync = 2400 samples
  N_delay = 480 samples
  N_chirp = 4800 samples
  N_total = 7680 samples (160 ms)
  
CIR EXTRACTION:
  N_CIR = 480 samples (10 ms window)
  Regularization: ε = 0.01
```

### Example Parameter Set (RF Channel, 1-100 MHz)

```
SYNC TONE:
  f_sync = 500 kHz (below band)
  T_sync = 10 ms
  A_sync = 1.0
  
DELAY:
  T_delay = 5 ms
  
CHIRP:
  f_min = 1 MHz
  f_max = 100 MHz
  T_chirp = 10 ms (100 kHz resolution)
  A_chirp = 1.0
  
SAMPLING:
  f_s = 250 MHz (Nyquist with margin)
  
BUFFER:
  N_sync = 2.5M samples
  N_delay = 1.25M samples
  N_chirp = 2.5M samples
  N_total = 6.25M samples (25 ms)
  
CIR EXTRACTION:
  N_CIR = 2500 samples (10 μs window)
  Regularization: ε = 0.001
```

---

## PART 12: VALIDATION AND CALIBRATION

### Reference Channel Test

**Loopback test** (transmitter → receiver directly):
```
Expected results:
  |H(f)| ≈ 1 (0 dB) for all f
  SNR > 40 dB (system noise floor)
  THD < 0.1% (no distortion)
  τ_rms ≈ 0 (no multipath)

Use to:
  - Calibrate system
  - Measure noise floor
  - Verify signal generation
```

### Known Channel Test

**Add known impairment**:
```
Test 1: Attenuation
  Insert -10 dB attenuator
  Expect: |H(f)| = -10 dB, SNR reduced by 10 dB

Test 2: Bandpass filter
  Insert filter [f1, f2]
  Expect: |H(f)| = 0 dB in [f1,f2], low outside

Test 3: Delay line
  Insert cable of length L
  Expect: τ = L/v, group delay = τ
```

### Confidence Metrics

**Measurement quality**:
```
Coherence function:
  γ²(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))

where:
  S_xy = cross-spectrum of input/output
  S_xx = input power spectrum
  S_yy = output power spectrum

Interpretation:
  γ² ≈ 1: High confidence, linear channel
  γ² < 0.8: Low confidence, nonlinear or noisy
```

**SNR confidence interval**:
```
Standard error:
  SE = SNR / √(2N)

where N = number of independent samples

95% confidence:
  SNR ± 1.96 × SE
```

---

## SUMMARY CHECKLIST

### Transmitter Implementation
- [ ] Generate sync tone: `A_sync × cos(2πf_sync × t)`
- [ ] Insert delay: silence for T_delay
- [ ] Generate log chirp: phase `φ(t) = 2πf_min T/L × [e^(Lt/T) - 1]`
- [ ] Concatenate: sync + delay + chirp + post-silence
- [ ] Normalize amplitude to prevent clipping
- [ ] Store reference chirp for receiver

### Receiver Implementation
- [ ] Detect sync tone: cross-correlation or Goertzel
- [ ] Timestamp detection: t_sync = peak location
- [ ] Extract chirp: y[n_start:n_end] where n_start = (t_sync+T_sync+T_delay)×f_s
- [ ] Compute CIR: H(f) = Y(f)/S(f), h[n] = IFFT(H)
- [ ] Calculate SNR: from CIR energy distribution
- [ ] Calculate THD: from harmonic power ratios
- [ ] Calculate delay spread: τ_rms from CIR
- [ ] Calculate coherence BW: B_c ≈ 1/(5τ_rms)

### Decision Logic
- [ ] Map (SNR, B_c, THD) → modulation scheme
- [ ] Validate measurement quality (coherence, confidence)
- [ ] Store channel state for tracking
- [ ] Schedule re-measurement based on triggers
- [ ] Adapt modulation when channel changes

---

This specification provides complete mathematical foundation for implementing channel measurement without prescribing code structure.