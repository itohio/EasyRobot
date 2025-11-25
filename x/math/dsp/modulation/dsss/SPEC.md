# Mathematical Specification: Modular DSSS Architecture

## Overview

This document describes the mathematical operations each component must perform, without implementation details. Use this to map to your existing codebase and identify missing functionality.

---

## TRANSMITTER CHAIN

### Component 1: Symbol Mapper

**Purpose**: Convert byte stream to symbol values

**Input**: 
- B = [b₀, b₁, b₂, ..., bₙ₋₁] where bᵢ ∈ {0, 1, ..., 255}

**Output**: 
- S = [s₀, s₁, s₂, ..., sₘ₋₁] where sᵢ ∈ {0, 1, ..., M-1}

**Parameters**:
- k = bits per symbol = log₂(M)
- M = modulation order (2 for binary, 4 for 4-ary, etc.)

**Operation**:

1. **Bit Extraction**: Convert bytes to bit stream
   ```
   For each byte bⱼ:
     Extract 8 bits: [b₇, b₆, b₅, b₄, b₃, b₂, b₁, b₀]
     where bᵢ = (bⱼ >> i) & 1
   
   Bit stream: B_bits = [b₀⁽⁰⁾, b₁⁽⁰⁾, ..., b₇⁽⁰⁾, b₀⁽¹⁾, ..., b₇⁽ⁿ⁻¹⁾]
   ```

2. **Symbol Packing**: Group k bits into symbols
   ```
   For i = 0 to m-1:
     sᵢ = Σ(j=0 to k-1) B_bits[i*k + j] × 2^(k-1-j)
   
   Example (binary, k=1):
     s₀ = B_bits[0]
     s₁ = B_bits[1]
   
   Example (4-ary, k=2):
     s₀ = B_bits[0]×2 + B_bits[1]×1
     s₁ = B_bits[2]×2 + B_bits[3]×1
   ```

**Number of Symbols**:
```
m = ⌈(n × 8) / k⌉
```

**Mathematical Properties**:
- Reversible mapping (bijection)
- No information loss
- Handles partial symbols (zero padding)

---

### Component 2: Chip Generator

**Purpose**: Convert symbols to chip sequences using spreading codes

**Input**: 
- sᵢ = symbol value, sᵢ ∈ {0, 1, ..., M-1}

**Output**: 
- C⁽ⁱ⁾ = [c₀⁽ⁱ⁾, c₁⁽ⁱ⁾, ..., c_{L-1}⁽ⁱ⁾] where cⱼ⁽ⁱ⁾ ∈ {-1, +1}

**Parameters**:
- L = spreading factor (chips per symbol)
- P = {P₀, P₁, ..., P_{M-1}} = set of M spreading codes
- Pₘ = [p₀⁽ᵐ⁾, p₁⁽ᵐ⁾, ..., p_{L-1}⁽ᵐ⁾] where pⱼ⁽ᵐ⁾ ∈ {-1, +1}

**Operation Types**:

#### A. Direct Mapping (Binary, M=2)
```
If sᵢ = 0:
  C⁽ⁱ⁾ = -P₀ = [-p₀, -p₁, ..., -p_{L-1}]

If sᵢ = 1:
  C⁽ⁱ⁾ = +P₀ = [+p₀, +p₁, ..., +p_{L-1}]
```

#### B. Code Selection (M-ary, M>2)
```
C⁽ⁱ⁾ = P_{sᵢ}

Example (4-ary):
  s=0 → C = P₀
  s=1 → C = P₁
  s=2 → C = P₂
  s=3 → C = P₃
```

#### C. Differential Encoding (Optional)
```
Maintain state: s_prev (initialized to 0)

Transmitted symbol:
  s_tx = s_prev ⊕ sᵢ  (XOR for binary)
  s_tx = (s_prev + sᵢ) mod M  (for M-ary)

Then apply mapping A or B to s_tx

Update state:
  s_prev ← s_tx
```

**Code Requirements**:
- All codes must have length L
- For M-ary: need M distinct codes
- Codes should be orthogonal or near-orthogonal:
  ```
  ⟨Pᵢ, Pⱼ⟩ = (1/L)Σ(k=0 to L-1) pₖ⁽ⁱ⁾ × pₖ⁽ʲ⁾ ≈ δᵢⱼ
  
  where δᵢⱼ = 1 if i=j, 0 otherwise
  ```

**Processing Gain**:
```
G_p = L (in linear scale)
G_p[dB] = 10 log₁₀(L)

Example: L=7 → G_p = 8.45 dB
```

---

### Component 3: Pulse Shaper

**Purpose**: Convert discrete chips to continuous baseband waveform

**Input**: 
- cⱼ = chip value, cⱼ ∈ {-1, +1}

**Output**: 
- x(t) = baseband waveform for chip duration T_c

**Parameters**:
- T_c = chip duration (seconds)
- N_sc = samples per chip
- f_s = sample rate (Hz)
- T_s = 1/f_s = sample period
- g(t) = pulse shape function

**Sampling Constraint**:
```
N_sc = ⌊T_c × f_s⌋ = ⌊f_s / (R_s × L)⌋

where:
  R_s = symbol rate (symbols/sec)
  T_c = 1/(R_s × L) = chip duration
```

**Operation**:

#### A. Rectangular Pulse (Simplest)
```
For chip cⱼ, generate N_sc samples:

x[n] = A × cⱼ,  n = 0, 1, ..., N_sc-1

where A = amplitude
```

**Properties**:
- No inter-chip interference
- Maximum bandwidth (sinc spectrum)
- Simple implementation

#### B. Raised Cosine Pulse (Bandwidth Efficient)
```
Pulse shape in time domain:

g(t) = sinc(πt/T_c) × cos(παt/T_c) / (1 - (2αt/T_c)²)

where:
  sinc(x) = sin(x)/x
  α = roll-off factor ∈ [0, 1]
  t = time relative to pulse center

For chip cⱼ, generate samples:

x[n] = A × cⱼ × g(n×T_s - T_c/2)

where n = 0, 1, ..., N_sc-1
```

**Frequency Response**:
```
G(f) = {
  T_c,                                           |f| ≤ (1-α)/(2T_c)
  (T_c/2)[1 + cos(π(T_c|f| - (1-α))/(2α))],    (1-α)/(2T_c) < |f| ≤ (1+α)/(2T_c)
  0,                                            |f| > (1+α)/(2T_c)
}
```

**Bandwidth**:
```
B_null = (1 + α)/(2T_c) Hz

Example:
  α = 0: B = 1/(2T_c)  (minimum bandwidth)
  α = 0.5: B = 0.75/T_c
  α = 1: B = 1/T_c
```

**Properties**:
- Zero ISI at optimal sampling points
- Controlled bandwidth
- Requires filter memory (spans multiple chips)

#### C. Root Raised Cosine (Matched Filter Optimum)
```
Pulse shape:

g_rrc(t) = {
  (1-α+4α/π)/T_c,                                           t = 0
  (α/√2T_c)[(1+2/π)sin(π/4α) + (1-2/π)cos(π/4α)],        |t| = T_c/4α
  (sin(π(1-α)t/T_c) + 4α(t/T_c)cos(π(1+α)t/T_c)) / (πt(1-(4αt/T_c)²))  otherwise
}

Used when matched filter is implemented in receiver.
```

**Total Sample Count per Symbol**:
```
N_symbol = L × N_sc
```

---

### Component 4: Upconverter (Carrier Modulation)

**Purpose**: Translate baseband signal to passband

**Input**: 
- x[n] = baseband samples

**Output**: 
- y[n] = passband samples

**Parameters**:
- f_c = carrier frequency (Hz)
- f_s = sample rate (Hz)
- φ(n) = carrier phase at sample n

**Operation Types**:

#### A. Passthrough (Baseband, No Carrier)
```
y[n] = x[n]

Use when testing or for baseband transmission.
```

#### B. Coherent BPSK/DSSS
```
y[n] = x[n] × cos(2πf_c n/f_s + φ₀)

where:
  φ₀ = initial phase (constant)
  φ(n) = 2πf_c n/f_s + φ₀  (linear phase)
```

**Phase Continuity**:
```
Must maintain phase across multiple Encode() calls:

φ(n) = 2πf_c (n_global)/f_s + φ₀

where n_global = total samples processed since initialization

To prevent overflow, wrap phase:
  φ(n) mod 2π

Or maintain as:
  n_counter (mod f_s) and compute φ = 2πf_c × n_counter/f_s
```

#### C. Quadrature Modulation (Future Extension)
```
I(n) = x_I[n] × cos(2πf_c n/f_s)
Q(n) = x_Q[n] × sin(2πf_c n/f_s)

y[n] = I(n) + Q(n)
```

**Nyquist Constraint**:
```
f_s ≥ 2(f_c + B/2)

where B = signal bandwidth

For safety, typically:
  f_s ≥ 4f_c
```

**Output Signal**:
```
Total samples for N bytes:
  N_total = ⌈(N × 8)/k⌉ × L × N_sc
  
where:
  k = bits per symbol
  L = spreading factor
  N_sc = samples per chip
```

---

## RECEIVER CHAIN

### Component 5: Chip Correlator

**Purpose**: Compute correlation between received signal and spreading codes

**Input**: 
- r[n] = received samples for one symbol period
- Length: N_symbol = L × N_sc samples

**Output**: 
- ρ = [ρ₀, ρ₁, ..., ρ_{M-1}] = correlation values
- ρᵢ ∈ ℝ (correlation with code Pᵢ)

**Parameters**:
- P = {P₀, P₁, ..., P_{M-1}} = spreading codes
- L = spreading factor
- N_sc = samples per chip
- f_c = carrier frequency (if coherent)
- f_s = sample rate
- φ̂ = estimated carrier phase

**Operation Types**:

#### A. Baseband Correlation
```
For each code Pᵢ:

ρᵢ = (1/(L×N_sc)) Σ(l=0 to L-1) Σ(m=0 to N_sc-1) r[l×N_sc + m] × pₗ⁽ⁱ⁾

where pₗ⁽ⁱ⁾ = l-th chip of code Pᵢ

Simplified:
  For each chip position l:
    Integrate N_sc samples
    Multiply by corresponding code chip
    Accumulate
```

**Step-by-step**:
```
1. Divide r[n] into L chip-length segments
   Segment l: r[l×N_sc] to r[(l+1)×N_sc - 1]

2. For each segment l:
   chip_sum = Σ(m=0 to N_sc-1) r[l×N_sc + m]
   
3. Multiply by code chip:
   chip_corr = chip_sum × pₗ⁽ⁱ⁾
   
4. Accumulate:
   ρᵢ = Σ(l=0 to L-1) chip_corr_l
```

#### B. Coherent Correlation (With Carrier and Pulse Shaping)
```
For each code Pᵢ:

ρᵢ = (1/(L×N_sc)) Σ(l=0 to L-1) Σ(m=0 to N_sc-1) r[l×N_sc + m] × cos(θ[l×N_sc + m]) × h[m] × pₗ⁽ⁱ⁾

where:
  θ[n] = 2πf_c n/f_s + φ̂
  φ̂ = estimated carrier phase
  h[m] = pulse shape (matched filter) at sample m within chip

This performs:
  1. Carrier demodulation: r[n] × cos(θ[n])
  2. Matched filtering: multiply by pulse shape h[m]
  3. Chip correlation: multiply by code, integrate
  4. Normalization: divide by total samples (L×N_sc)
```

**Important Note on Normalization:**
The normalization by `(L×N_sc)` means the correlation value is scaled down from the raw sum.
For a perfect match with amplitude A=1.0:
- Raw correlation sum ≈ A × L × N_sc (if all samples align perfectly)
- After normalization: ρᵢ ≈ A × L × N_sc / (L×N_sc) = A
- However, with pulse shaping (raised cosine), the effective amplitude is reduced
- Typical normalized correlation for matched code: ρᵢ ≈ 0.3-0.7 × A (depending on rolloff factor)

#### C. Non-Coherent Correlation (Envelope Detection)
```
For each code Pᵢ:

I component:
  Iᵢ = Σ(l=0 to L-1) Σ(m=0 to N_sc-1) r[l×N_sc + m] × cos(θ[l×N_sc + m]) × pₗ⁽ⁱ⁾

Q component:
  Qᵢ = Σ(l=0 to L-1) Σ(m=0 to N_sc-1) r[l×N_sc + m] × sin(θ[l×N_sc + m]) × pₗ⁽ⁱ⁾

Envelope:
  ρᵢ = √(Iᵢ² + Qᵢ²)
```

**Energy Normalization** (Optional but Recommended):
```
Compute signal energy:
  E = Σ(n=0 to N_symbol-1) r[n]²

Normalized correlation:
  ρ̃ᵢ = ρᵢ / √E

Benefits:
  - Amplitude independent
  - Improves threshold decisions
  - Better AGC tolerance
```

**Theoretical Values** (Perfect Channel, After Normalization):
```
Transmitted symbol sⱼ using code Pⱼ, with amplitude A=1.0:

For normalized correlation (divided by L×N_sc):
  ρᵢ = {
    ≈ A × α,     i = j (matched code)
    ≈ 0,         i ≠ j (orthogonal codes)
  }
  
where:
  α = pulse shape energy factor (typically 0.3-0.7 for raised cosine)
  
With noise:
  ρᵢ ~ N(μᵢ, σ²)
  
where:
  μⱼ ≈ A × α,  μᵢ = 0 for i≠j
  σ² = noise variance / (L×N_sc)
```

**Signal Detection Threshold:**
For reliable signal detection, the threshold should be:
- Minimum: 3×σ (noise standard deviation)
- Recommended: 0.05-0.1 for normalized correlation with A=1.0
- This accounts for pulse shaping reducing peak correlation values

---

### Component 6: Symbol Decoder

**Purpose**: Decide which symbol was transmitted based on correlations

**Input**: 
- ρ = [ρ₀, ρ₁, ..., ρ_{M-1}] = correlation values

**Output**: 
- ŝ = detected symbol, ŝ ∈ {0, 1, ..., M-1}
- confidence metric (optional)

**Operation Types**:

#### A. Maximum Likelihood (Binary)
```
Given: single correlation ρ₀ with code P₀

Decision:
  ŝ = {
    0,  if ρ₀ < 0
    1,  if ρ₀ ≥ 0
  }

Confidence:
  C = |ρ₀|  (distance from threshold)
```

**Rationale**:
```
Symbol 0 transmitted: -P₀ → ρ₀ < 0
Symbol 1 transmitted: +P₀ → ρ₀ > 0
```

#### B. Maximum Likelihood (M-ary)
```
Decision:
  ŝ = arg max_{i} |ρᵢ|

i.e., choose symbol corresponding to code with highest absolute correlation.

Confidence:
  C = |ρ_ŝ|
  
or relative confidence:
  C = (|ρ_ŝ| - max_{i≠ŝ}|ρᵢ|) / |ρ_ŝ|
```

#### C. Soft Decision (For FEC)
```
Instead of hard symbol ŝ, output:
  - All correlation values ρ
  - Or log-likelihood ratios:
    
    LLR_i = log(P(symbol=i | ρ) / P(symbol≠i | ρ))
```

#### D. Differential Decoding
```
Maintain state: ŝ_prev (initialized to 0)

After detecting ŝ_current:
  
  ŝ_data = ŝ_prev ⊕ ŝ_current  (XOR for binary)
  ŝ_data = (ŝ_current - ŝ_prev) mod M  (for M-ary)

Update state:
  ŝ_prev ← ŝ_current

Output:
  ŝ_data (the actual data symbol)
```

**Error Propagation**:
```
Differential decoding causes:
  - 1 channel error → 2 decoded errors (error propagation)
  - But immune to phase ambiguity (0° vs 180°)
```

**Decision Regions** (Binary):
```
Symbol space: ρ ∈ ℝ
Threshold: τ = 0

Decide 0: ρ < τ
Decide 1: ρ ≥ τ

Probability of error:
  P_e = Q(√(2E_b/N_0))
  
where:
  E_b/N_0 = energy per bit to noise ratio
  Q(x) = (1/√(2π)) ∫(x to ∞) e^(-t²/2) dt
```

---

### Component 7: Symbol Demapper

**Purpose**: Convert symbol stream back to bytes

**Input**: 
- Ŝ = [ŝ₀, ŝ₁, ..., ŝₘ₋₁] where ŝᵢ ∈ {0, 1, ..., M-1}

**Output**: 
- B̂ = [b̂₀, b̂₁, ..., b̂ₙ₋₁] where b̂ᵢ ∈ {0, 1, ..., 255}

**Parameters**:
- k = bits per symbol = log₂(M)

**Operation**:

1. **Bit Extraction**: Convert symbols to bit stream
   ```
   For each symbol ŝᵢ:
     Extract k bits:
       bit_{i×k+j} = (ŝᵢ >> (k-1-j)) & 1,  j = 0, 1, ..., k-1
   
   Example (binary, k=1):
     ŝ₀ = 1 → bit₀ = 1
     ŝ₁ = 0 → bit₁ = 0
   
   Example (4-ary, k=2):
     ŝ₀ = 3 = 0b11 → [bit₀=1, bit₁=1]
     ŝ₁ = 2 = 0b10 → [bit₂=1, bit₃=0]
   ```

2. **Byte Packing**: Group 8 bits into bytes
   ```
   For j = 0 to n-1:
     b̂ⱼ = Σ(i=0 to 7) bit_{j×8+i} × 2^(7-i)
   
   MSB-first packing.
   ```

**Handling Incomplete Bytes**:
```
If total_bits mod 8 ≠ 0:
  - Last byte has fewer than 8 bits
  - Zero-pad remaining bits
  - Track actual bit count for trimming
```

**Number of Output Bytes**:
```
n = ⌊(m × k) / 8⌋

where m = number of symbols
```

---

## SYNCHRONIZATION REQUIREMENTS

### Timing Synchronization

**Symbol Timing**: Determine where symbols start/end

**Coarse Acquisition** (Using Preamble):
```
Preamble: Known sequence of N_pre symbols

Detection:
  1. Compute correlation with preamble at different offsets
  
  2. For each offset τ:
     R(τ) = Σ(i=0 to N_pre-1) r[τ + i×N_symbol] × p_pre[i]
  
  3. Find peak:
     τ̂ = arg max_τ R(τ)
  
  4. Symbol boundary at sample τ̂
```

**Fine Tracking** (Early-Late Gate):
```
Compute correlations at three timing offsets:
  - Early: τ - Δ
  - Prompt: τ
  - Late: τ + Δ

Timing error estimate:
  ε = |R(τ - Δ)| - |R(τ + Δ)|

Update:
  τ ← τ + μ × ε
  
where μ = loop gain
```

### Carrier Synchronization

**Phase Estimation** (Data-Aided):
```
Using known symbols ŝ_known:

Compute phase error for each symbol:
  ε_i = ∠(ρ_i × conj(ŝ_known[i]))

Average:
  φ̂ = (1/N) Σ(i=0 to N-1) ε_i
```

**Phase Tracking** (Decision-Directed):
```
After each symbol decision:

Phase error:
  ε = ∠(ρ × conj(ŝ_decided))

Update phase estimate:
  φ̂ ← φ̂ + μ × ε
  
where μ = loop gain
```

**Costas Loop** (For BPSK):
```
I = ρ_I = Re{r × e^(-jφ̂)}
Q = ρ_Q = Im{r × e^(-jφ̂)}

Phase error:
  ε = sign(I) × Q

Update:
  φ̂ ← φ̂ + μ × ε
```

---

## PERFORMANCE METRICS

### Signal-to-Noise Ratio

**Input SNR**:
```
SNR_in = P_signal / P_noise

where:
  P_signal = average signal power
  P_noise = noise power in chip bandwidth
```

**Output SNR** (After Despreading):
```
SNR_out = SNR_in × L

where L = spreading factor (processing gain)

In dB:
  SNR_out[dB] = SNR_in[dB] + 10 log₁₀(L)
```

### Bit Error Rate

**Theoretical BER** (BPSK, AWGN):
```
P_b = Q(√(2E_b/N_0))

where:
  E_b/N_0 = bit energy to noise ratio
  E_b = E_s / k  (E_s = symbol energy, k = bits/symbol)
```

**With Spreading**:
```
E_b/N_0 = (E_s/N_0) / k = (SNR_out × T_s) / k

Processing gain improves:
  (E_b/N_0)_out = (E_b/N_0)_in × L
```

### Bandwidth Efficiency

**Chip Rate**:
```
R_c = R_s × L

where:
  R_s = symbol rate
  L = spreading factor
```

**Null Bandwidth**:
```
B_null = (1 + α) × R_c

where α = pulse shaping roll-off

Example:
  R_s = 1000 symbols/s
  L = 7
  α = 0.5
  
  R_c = 7000 chips/s
  B = 1.5 × 7000 = 10.5 kHz
```

**Spectral Efficiency**:
```
η = (R_s × k) / B_null  bits/s/Hz

Example (binary, L=7, α=0.5):
  η = (1000 × 1) / 10500 = 0.095 bits/s/Hz
```

---

## IMPLEMENTATION CHECKLIST

### Transmitter Components

- [ ] **Symbol Mapper**
  - [ ] Byte to bit stream conversion
  - [ ] Bit grouping (k bits → symbol)
  - [ ] Partial symbol handling (padding)
  - [ ] State management for streaming

- [ ] **Chip Generator**
  - [ ] Code storage (bipolar format ±1)
  - [ ] Symbol to chip mapping
  - [ ] Differential encoding (optional)
  - [ ] Support for M codes (M-ary)

- [ ] **Pulse Shaper**
  - [ ] Rectangular pulse (basic)
  - [ ] Raised cosine (bandwidth efficient)
  - [ ] Root raised cosine (matched filter)
  - [ ] Filter state management

- [ ] **Upconverter**
  - [ ] Passthrough (baseband)
  - [ ] Coherent modulation (cos carrier)
  - [ ] Phase continuity across calls
  - [ ] Phase wrapping (prevent overflow)

### Receiver Components

- [ ] **Chip Correlator**
  - [ ] Baseband correlation
  - [ ] Coherent demodulation + correlation
  - [ ] Non-coherent (I/Q) correlation
  - [ ] Energy normalization
  - [ ] Support for M codes

- [ ] **Symbol Decoder**
  - [ ] Maximum likelihood decision
  - [ ] Threshold detection (binary)
  - [ ] Differential decoding
  - [ ] Confidence metric calculation

- [ ] **Symbol Demapper**
  - [ ] Symbol to bit stream conversion
  - [ ] Bit packing into bytes
  - [ ] Partial byte handling
  - [ ] State management for streaming

### Synchronization (Optional but Recommended)

- [ ] **Preamble Generator**
  - [ ] Known symbol sequence
  - [ ] Good autocorrelation properties

- [ ] **Preamble Detector**
  - [ ] Correlation-based search
  - [ ] Threshold detection
  - [ ] Timing offset estimation

- [ ] **Phase Estimator**
  - [ ] Data-aided estimation
  - [ ] Decision-directed tracking
  - [ ] Costas loop (continuous tracking)

---

## PARAMETER RELATIONSHIPS

### Key Constraints

```
Sample Rate:
  f_s ≥ max(4f_c, R_c × N_sc)
  
where:
  R_c = chip rate = R_s × L
  N_sc = samples per chip (≥ 4 recommended)

Example:
  R_s = 1000 symbols/s
  L = 7 (spreading factor)
  f_c = 5000 Hz (carrier)
  
  R_c = 7000 chips/s
  
  Constraint 1: f_s ≥ 4 × 5000 = 20000 Hz
  Constraint 2: f_s ≥ 7000 × 4 = 28000 Hz
  
  Choose: f_s ≥ 28000 Hz (use 48000 Hz for standard)
```

### Throughput Calculation

```
Data Rate = R_s × k bits/s

where:
  R_s = symbol rate (symbols/s)
  k = log₂(M) bits per symbol

Example comparisons (R_s = 1000):
  Binary (M=2):  1000 bits/s
  4-ary (M=4):   2000 bits/s
  8-ary (M=8):   3000 bits/s
  16-ary (M=16): 4000 bits/s

All use same chip rate: R_c = R_s × L
```

### Processing Gain vs Data Rate

```
Bandwidth fixed: B = R_c × (1 + α)

Trade-off:
  Higher L → More processing gain, lower data rate
  Higher M → Higher data rate, lower code orthogonality

Optimal choice depends on:
  - Required data rate
  - Channel SNR
  - Interference environment
```

---

## Encoder/Decoder Composition Patterns

### EncoderChain and DecoderChain

**Purpose**: Chain multiple encoders/decoders together in sequence.

**EncoderChain Flow**:
```
src -> encoder1 -> buffer1 -> encoder2 -> buffer2 -> ... -> encoderN -> dst
```

**DecoderChain Flow**:
```
src -> decoder1 -> buffer1 -> decoder2 -> buffer2 -> ... -> decoderN -> dst
```

**Type Parameters**:
- `DST`: Destination sample type (output type of all encoders/decoders)
- `SRC`: Source sample type (input type, must equal DST for chains)
- `BUF`: Buffer type for intermediate storage (must implement `WriterReader[DST]`)

**Type Constraints**:
- All encoders/decoders in chain must be `Encoder[DST, SRC]` / `Decoder[DST, SRC]`
- For chains to work, **SRC == DST** (all stages work on same type)
- `BUF` must implement `WriterReader[DST]` interface:
  - `buffer.Writer[DST]` (Write method)
  - `buffer.Reader[DST]` (Read method)
  - `Reset()` method
  - `Cap() int` method
  - `Len() int` method
- Type adapter handles conversion between `buffer.Reader[DST]` and `buffer.Reader[SRC]`

**Usage with Ring Buffer** (for byte-based encoders):
```go
factory := func(cap int) *buffer.Ring[float32] {
    return buffer.NewRing(make([]float32, 0, cap))
}
chain := NewEncoderChain[float32, float32, *buffer.Ring[float32]](factory, encoder1, encoder2, encoder3)
samplesWritten := chain.Encode(dst, src)
```

**Usage with BitsRing** (for bit-based encoders like DSSS, FSK):
```go
// When BitsRing is implemented
factory := func(cap int) *buffer.BitsRing[byte] {
    return buffer.NewBitsRing(make([]byte, 0, cap))
}
chain := NewEncoderChain[byte, byte, *buffer.BitsRing[byte]](factory, encoder1, encoder2, encoder3)
samplesWritten := chain.Encode(dst, src)
```

**Buffer Management**:
- Intermediate buffers are allocated on first call with non-nil dst
- Buffer factory function creates buffers with specified capacity
- Buffer sizes are calculated by propagating size estimates through all encoders
- Buffers are reset between calls but retain capacity

### ComposeEncoder and ComposeDecoder

**Purpose**: Compose two encoders/decoders with different intermediate types.

**ComposeEncoder Flow**:
```
src (SRC) -> src.Encode() -> intermediate buffer (MID) -> dst.Encode() -> dst (DST)
```

**ComposeDecoder Flow**:
```
src (SRC) -> src.Decode() -> intermediate buffer (MID) -> dst.Decode() -> dst (DST)
```

**Type Parameters**:
- `DST`: Destination sample type (output type of second encoder/decoder)
- `MID`: Intermediate sample type (output of first, input to second)
- `SRC`: Source sample type (input type of first encoder/decoder)
- `BUF`: Buffer type for intermediate storage (must implement `WriterReader[MID]`)

**Usage with Ring Buffer**:
```go
factory := func(cap int) *buffer.Ring[[]byte] {
    return buffer.NewRing(make([][]byte, 0, cap))
}
compose := NewComposeEncoder[float32, []byte, byte, *buffer.Ring[[]byte]](factory, dstEncoder, srcEncoder)
samplesWritten := compose.Encode(dst, src)
```

**Usage with BitsRing** (for bit-based intermediate stages):
```go
// When BitsRing is implemented
factory := func(cap int) *buffer.BitsRing[byte] {
    return buffer.NewBitsRing(make([]byte, 0, cap))
}
compose := NewComposeEncoder[float32, byte, byte, *buffer.BitsRing[byte]](factory, dstEncoder, srcEncoder)
samplesWritten := compose.Encode(dst, src)
```

**Buffer Management**:
- Intermediate buffer is allocated on first call with non-nil dst
- Buffer factory function creates buffer with specified capacity
- Buffer size is calculated from src encoder output size estimate
- Buffer is reset between calls but retains capacity

**Benefits**:
- Allows type transformations: SRC -> MID -> DST
- More flexible than chains (supports different intermediate types)
- User can choose appropriate buffer type (Ring for bytes, BitsRing for bits)
- Useful for multi-stage encoding/decoding pipelines

---

This mathematical specification provides everything needed to implement or integrate DSSS components without prescribing specific code structures.