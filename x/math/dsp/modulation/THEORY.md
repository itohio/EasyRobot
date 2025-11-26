# Modulation Scheme Comparison

## Comparison Table

### Nomenclature
- **BDSSS**: Binary Direct Sequence Spread Spectrum
- **BFSK**: Binary Frequency Shift Keying
- **BPSK**: Binary Phase Shift Keying
- **4DSSS**: 4-ary DSSS (2 bits/symbol)
- **4FSK**: 4-ary FSK (2 bits/symbol)
- **4PSK** (QPSK): Quadrature Phase Shift Keying (2 bits/symbol)
- **4ASK**: 4-ary Amplitude Shift Keying (2 bits/symbol)
- **4QAM**: 4-ary Quadrature Amplitude Modulation (2 bits/symbol, same as QPSK)
- **8DSSS**: 8-ary DSSS (3 bits/symbol)
- **8FSK**: 8-ary FSK (3 bits/symbol)
- **8PSK**: 8-ary Phase Shift Keying (3 bits/symbol)
- **8ASK**: 8-ary Amplitude Shift Keying (3 bits/symbol)
- **8QAM**: 8-ary Quadrature Amplitude Modulation (3 bits/symbol)
- **16QAM**: 16-ary QAM (4 bits/symbol)
- **64QAM**: 64-ary QAM (6 bits/symbol)
- **256QAM**: 256-ary QAM (8 bits/symbol)

### Assumptions for Comparison
- Symbol rate: R_s = 1000 symbols/second
- Spreading factor (DSSS): L = 7
- Pulse shaping roll-off: α = 0.5 (where applicable)
- Carrier frequency: f_c = 10 kHz (for coherent schemes)
- Target BER: 10^-5

---

## Main Comparison Table

| Scheme | Bandwidth (Hz) | Symbol Spacing | Peak/Avg Power | Required SNR (dB) | Linearity Req. | Bits/Symbol |
|--------|----------------|----------------|----------------|-------------------|----------------|-------------|
| **BDSSS** | 10,500 | ±L | 1.0 / 1.0 | 9.6 | Low | 1 |
| **BFSK** | 3,000 | f_0 ± 500 Hz | 1.0 / 1.0 | 13.5 | Very Low | 1 |
| **BPSK** | 1,500 | 180° | 1.0 / 1.0 | 9.6 | Medium | 1 |
| **4DSSS** | 10,500 | ±L (4 codes) | 1.0 / 1.0 | 12.6 | Low | 2 |
| **4FSK** | 6,000 | f_0 ± {750, 250} Hz | 1.0 / 1.0 | 16.5 | Very Low | 2 |
| **4PSK** | 1,500 | 90° | 1.0 / 1.0 | 9.6 | Medium | 2 |
| **4ASK** | 1,500 | Amp: {A, 3A, 5A, 7A} | 2.33 / 1.0 | 13.9 | Very High | 2 |
| **4QAM** | 1,500 | Grid 2×2 | 1.0 / 1.0 | 9.6 | Medium-High | 2 |
| **8DSSS** | 10,500 | ±L (8 codes) | 1.0 / 1.0 | 15.6 | Low | 3 |
| **8FSK** | 12,000 | f_0 ± {1750, 1250, ...} Hz | 1.0 / 1.0 | 19.5 | Very Low | 3 |
| **8PSK** | 1,500 | 45° | 1.0 / 1.0 | 13.9 | High | 3 |
| **8ASK** | 1,500 | 8 amplitudes | 3.57 / 1.0 | 18.5 | Very High | 3 |
| **8QAM** | 1,500 | Mixed grid | 1.65 / 1.0 | 13.0 | High | 3 |
| **16QAM** | 1,500 | Grid 4×4 | 1.32 / 1.0 | 13.4 | Very High | 4 |
| **64QAM** | 1,500 | Grid 8×8 | 1.18 / 1.0 | 19.4 | Extreme | 6 |
| **256QAM** | 1,500 | Grid 16×16 | 1.11 / 1.0 | 25.4 | Extreme | 8 |

---

## Detailed Parameter Calculations

### Bandwidth

**DSSS (All Orders)**:
```
B_DSSS = (1 + α) × R_s × L
       = 1.5 × 1000 × 7
       = 10,500 Hz

Independent of modulation order (same spreading factor)
```

**FSK**:
```
B_FSK = M × Δf + B_symbol
      ≈ M × R_s/2

Binary FSK:
  B = 2 × 500 + 1000 = 3,000 Hz

4-ary FSK:
  Δf = 500 Hz, M=4
  B = 4 × 500 + 1000 ≈ 6,000 Hz

8-ary FSK:
  Δf = 500 Hz, M=8
  B = 8 × 500 + 1000 ≈ 12,000 Hz

Bandwidth grows linearly with M
```

**PSK**:
```
B_PSK = (1 + α) × R_s
      = 1.5 × 1000
      = 1,500 Hz

Independent of M (same symbol rate, different phases)
Most bandwidth efficient
```

### Symbol Spacing

**DSSS**:
```
Spreading codes provide separation

Binary: 2 codes (code, -code)
  Distance: 2L (orthogonal in chip space)

M-ary: M orthogonal codes
  Distance: √(2L) (Walsh codes)
  
Separation in correlation space, not frequency/phase
```

**FSK**:
```
Frequency spacing for orthogonality:
  Δf_min = R_s / 2

Binary: f_0, f_1
  Spacing: Δf = 500 Hz

4-ary: f_0, f_1, f_2, f_3
  f_i = f_c + (2i - M - 1) × Δf/2
  Spacing: 500 Hz between adjacent

8-ary: 8 frequencies
  Spacing: 500 Hz between adjacent
  Span: 3500 Hz total
```

**PSK**:
```
Phase spacing:
  Δφ = 2π / M

Binary: 0°, 180°
  Spacing: 180°

4-PSK: 0°, 90°, 180°, 270°
  Spacing: 90°

8-PSK: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
  Spacing: 45°

Closer spacing → higher error sensitivity
```

### Power (Peak-to-Average Ratio)

**Constant Envelope (PAPR = 1.0)**:
- DSSS (all orders): constant amplitude
- FSK (all orders): constant amplitude per frequency
- BPSK: antipodal (±1)
- 4PSK/QPSK: constant magnitude (unit circle)
- 8PSK: constant magnitude

```
PAPR = Peak Power / Average Power = 1.0 for all

Advantage: Efficient power amplifier operation
           No backoff required
           Maximum battery efficiency
```

**Variable Envelope - ASK Family**:

**4ASK** (4 amplitude levels):
```
Symbols: {A, 3A, 5A, 7A} (normalized to average = 1)

Peak power: (7A)² = 49A²
Average power: (A² + 9A² + 25A² + 49A²)/4 = 21A²

PAPR = 49A² / 21A² = 2.33

Requires 3.7 dB amplifier backoff
```

**8ASK** (8 amplitude levels):
```
Symbols: {A, 3A, 5A, 7A, 9A, 11A, 13A, 15A}

Peak power: (15A)² = 225A²
Average power: (A² + 9A² + ... + 225A²)/8 = 63A²

PAPR = 225A² / 63A² = 3.57

Requires 5.5 dB amplifier backoff
```

**QAM Family**:

**4QAM** (same as QPSK):
```
Constellation: 4 points on unit circle
PAPR = 1.0 (constant envelope)
```

**8QAM** (rectangular):
```
Constellation: 8 points, non-uniform spacing
Inner points: r = 1
Outer points: r = √3

Peak power: 3
Average power: ~1.82

PAPR = 1.65

Requires 2.2 dB amplifier backoff
```

**16QAM** (square grid):
```
Constellation: 4×4 grid
Points at: ±1, ±3 (I and Q)

Peak amplitude: √(3² + 3²) = √18
Average power: E[|s|²] = 10 (normalized)

Peak power: 18
Average power: 13.6

PAPR = 1.32

Requires 1.2 dB amplifier backoff
```

**64QAM** (square grid):
```
Constellation: 8×8 grid
Points at: ±1, ±3, ±5, ±7

Peak amplitude: √(7² + 7²) = √98
Average power: 42

PAPR = 98/42 = 1.18

Requires 0.7 dB amplifier backoff
```

**256QAM** (square grid):
```
Constellation: 16×16 grid
Points at: ±1, ±3, ..., ±15

Peak amplitude: √(15² + 15²) = √450
Average power: 170

PAPR = 450/170 = 1.11

Requires 0.5 dB amplifier backoff
```

**Key Insight**:
```
ASK: Very high PAPR (2.33-3.57), power inefficient
QAM: Moderate PAPR (1.1-1.65), much better than ASK
PSK: Perfect PAPR = 1.0, most power efficient

As QAM order increases, PAPR approaches 1.0
because constellation fills circle more uniformly
```

### Required SNR for BER = 10^-5

**Theoretical E_b/N_0** (AWGN channel):

**Binary Schemes** (1 bit/symbol):
```
BPSK:
  E_b/N_0 = 9.6 dB (theoretical optimum for binary)

BFSK (non-coherent):
  E_b/N_0 = 13.5 dB
  
BFSK (coherent):
  E_b/N_0 = 9.6 dB (same as BPSK)

BDSSS:
  E_b/N_0 = 9.6 dB (same as BPSK after despreading)
  
  BUT: Processing gain G_p = L = 7 (8.45 dB)
  
  Input SNR requirement:
    SNR_in = E_b/N_0 - G_p = 9.6 - 8.45 = 1.15 dB
  
  Can operate at negative input SNR!
```

**4-ary Schemes** (2 bits/symbol):
```
E_s/N_0 = E_b/N_0 + 3 dB (energy per symbol = 2 × energy per bit)

4PSK/QPSK:
  E_s/N_0 = 9.6 dB (same as BPSK)
  Twice the throughput for same SNR!

4FSK (non-coherent):
  E_s/N_0 = 16.5 dB

4DSSS:
  E_s/N_0 = 12.6 dB (degrades 3 dB vs binary due to code correlation)
  With processing gain:
    SNR_in = 12.6 - 8.45 = 4.15 dB
```

**8-ary Schemes** (3 bits/symbol):
```
E_s/N_0 = E_b/N_0 + 4.77 dB (energy per symbol = 3 × energy per bit)

8PSK:
  E_s/N_0 = 13.9 dB
  Higher than 4PSK due to closer phase spacing

8FSK (non-coherent):
  E_s/N_0 = 19.5 dB
  
8DSSS:
  E_s/N_0 = 15.6 dB
  With processing gain:
    SNR_in = 15.6 - 8.45 = 7.15 dB
```

**Summary Pattern**:
```
PSK: Most power efficient, grows slowly with M
FSK: Least power efficient, grows ~3 dB per doubling of M
DSSS: Middle ground, benefits from processing gain
```

### Channel Linearity Requirements

**Definition**: How sensitive is the modulation to nonlinear distortion (amplitude compression, intermodulation)?

**Very Low (FSK)**:
```
FSK uses frequency separation
Information in frequency, not amplitude or phase

Tolerances:
  - Amplitude compression: No effect
  - AM-AM distortion: No effect
  - AM-PM distortion: Minimal effect
  - Harmonic distortion: Filtered out (orthogonal frequencies)

Ideal for:
  - Saturated amplifiers
  - Class C, Class E amplifiers (high efficiency)
  - Nonlinear channels (magnetic recording, etc.)
```

**Low (DSSS)**:
```
Constant envelope (like FSK)
Information in chip sequence correlation

Tolerances:
  - Amplitude compression: No effect (constant envelope)
  - Phase distortion: Some tolerance (correlation-based)
  - Clipping: Moderate tolerance due to spreading

Better than:
  - PSK (no amplitude sensitivity)

Worse than:
  - FSK (phase still matters somewhat)
```

**Medium (BPSK, 4PSK/QPSK)**:
```
Information in phase only (constant envelope)

Tolerances:
  - Amplitude compression: No effect
  - Phase distortion: Sensitive
  - AM-PM conversion: Degrading (phase errors)

Requirements:
  - Linear phase response
  - Moderate AM-PM (<5° at saturation)
  - Can use near-saturation amplifiers (1-2 dB backoff)

Class AB amplifiers suitable
```

**High (8PSK)**:
```
Dense phase constellation (45° spacing)

Sensitivities:
  - Phase noise: Very sensitive
  - AM-PM distortion: Critical
  - Phase nonlinearity: Causes constellation rotation

Requirements:
  - Highly linear phase
  - AM-PM < 2° at operating point
  - Higher amplifier backoff (3-5 dB)
  - Better oscillator phase noise

Requires Class A or backed-off Class AB
```

**Quantitative Linearity Metric** (3rd-order intercept point):
```
Relative OIP3 requirements (approximate):

FSK: OIP3 > P_out + 10 dB (relaxed)
DSSS: OIP3 > P_out + 15 dB
BPSK/QPSK: OIP3 > P_out + 20 dB
8PSK: OIP3 > P_out + 30 dB (stringent)

where P_out = output power
```

---

## Detailed Scheme Analysis

### Binary DSSS (BDSSS)

**How It Works**:
```
Bit → Spread with code → Modulate carrier

Example:
  Bit 1 → code sequence [+1,-1,+1,+1,-1,-1,+1]
  Bit 0 → inverted code [-1,+1,-1,-1,+1,+1,-1]
  
Each bit expanded to L chips
Occupies L times bandwidth
```

**Pros**:
- **Processing gain**: Can operate below noise floor (anti-jam, LPI)
- **Interference rejection**: Narrowband interference suppressed by 10 log₁₀(L) dB
- **Multiple access**: Different codes = different users (CDMA)
- **Multipath resistance**: Code autocorrelation helps time resolution
- **Covert**: Looks like noise, hard to detect
- **Constant envelope**: Power efficient amplifiers

**Cons**:
- **Wide bandwidth**: L times wider than BPSK
- **Code synchronization**: Requires accurate chip timing
- **Near-far problem**: Strong signals swamp weak ones
- **Self-interference**: Non-perfect code orthogonality
- **Complexity**: Correlation receiver more complex than PSK

**Best For**:
- Low SNR environments (< 0 dB)
- Interference-heavy spectrum
- Secure communications (military)
- Multiple simultaneous users
- Multipath channels

**Avoid When**:
- Bandwidth limited
- Simple receiver desired
- High data rate priority

---

### Binary FSK (BFSK)

**How It Works**:
```
Bit → Select frequency

Bit 0 → f_0 = f_c - Δf/2
Bit 1 → f_1 = f_c + Δf/2

Non-coherent detection: energy detectors per frequency
Coherent detection: correlate with both frequencies
```

**Pros**:
- **Non-coherent detection**: No phase tracking needed
- **Simple implementation**: Oscillator + frequency detector
- **Nonlinearity immune**: Constant envelope, amplitude independent
- **Phase noise tolerant**: Information in frequency, not phase
- **Cheap hardware**: Can use simple discriminators
- **Doppler tolerant**: Frequency shift doesn't destroy information

**Cons**:
- **Power inefficient**: 3.5-4 dB worse than BPSK (non-coherent)
- **Bandwidth inefficient**: 2× wider than BPSK minimum
- **Frequency stability required**: Drift causes errors
- **Slower than PSK**: Requires observation over symbol period
- **Synchronous detection hard**: Orthogonality requires precise Δf

**Best For**:
- Simple, low-cost receivers
- Nonlinear channels (saturation, limiting)
- Non-coherent detection (no phase lock)
- Frequency-division multiple access
- Low data rate, robust links

**Avoid When**:
- Power limited (battery)
- Bandwidth limited
- High data rate needed
- Channel has frequency-selective fading

---

### Binary PSK (BPSK)

**How It Works**:
```
Bit → Phase

Bit 0 → φ = 0° (cos(ωt))
Bit 1 → φ = 180° (-cos(ωt))

Coherent detection: multiply by carrier, integrate
Maximum separation in phase space (antipodal)
```

**Pros**:
- **Power efficient**: Theoretical optimum for binary (9.6 dB @ 10^-5 BER)
- **Bandwidth efficient**: Minimum bandwidth for binary
- **Simple constellation**: Two points, maximum separation
- **Differential version exists**: DBPSK avoids phase ambiguity
- **Mature technology**: Well understood, widely implemented
- **Constant envelope**: Efficient amplification

**Cons**:
- **Phase synchronization required**: Needs carrier recovery
- **Phase ambiguity**: 180° ambiguity (use differential or sync sequence)
- **Sensitive to phase noise**: Phase jitter directly causes errors
- **Linearity needed**: AM-PM distortion degrades performance
- **Doppler sensitive**: Frequency offset causes phase rotation

**Best For**:
- Power-limited channels (satellite, deep space)
- Narrowband channels
- Coherent detection available
- Standard communications (Wi-Fi, cellular baseline)

**Avoid When**:
- Phase tracking unavailable
- Severe phase noise
- Non-coherent detection required
- Highly nonlinear channel

---

### 4-ary DSSS (4DSSS)

**How It Works**:
```
2 bits → Select one of 4 orthogonal codes

00 → Code_0
01 → Code_1
10 → Code_2
11 → Code_3

Typically Walsh codes for orthogonality
Correlate received with all 4 codes, pick maximum
```

**Pros**:
- **2× throughput vs BDSSS**: Same bandwidth, double data rate
- **Processing gain**: Still benefits from spreading
- **Interference rejection**: Like BDSSS
- **Multiple access**: 4 orthogonal channels
- **Graceful degradation**: Can fall back to binary

**Cons**:
- **3 dB SNR penalty**: Reduced code separation vs binary
- **4 correlators needed**: More complex receiver
- **Code orthogonality critical**: Multipath destroys orthogonality
- **Same wide bandwidth**: No bandwidth savings vs BDSSS
- **Synchronization harder**: Need precise code alignment

**Best For**:
- Higher throughput in spread spectrum
- CDMA with moderate data rates
- When already using DSSS, want more capacity
- Interference-heavy with decent SNR (>5 dB)

**Avoid When**:
- SNR limited (< 5 dB)
- Simple receiver required
- Bandwidth at premium

---

### 4-ary FSK (4FSK)

**How It Works**:
```
2 bits → Select one of 4 frequencies

00 → f_0
01 → f_1
10 → f_2
11 → f_3

Spacing: Δf = R_s/2 for orthogonality
```

**Pros**:
- **2× throughput vs BFSK**: Same symbol rate, double bits
- **Non-coherent detection**: Still possible
- **Nonlinear tolerant**: Like BFSK
- **Frequency diversity**: Each tone independent

**Cons**:
- **Wide bandwidth**: 2× wider than BFSK
- **Power penalty**: ~3 dB worse than 4PSK
- **4 frequency sources**: Or frequency synthesizer needed
- **Frequency stability**: More critical with 4 tones
- **Detector complexity**: 4 matched filters or discriminators

**Best For**:
- Non-coherent detection with higher data rate
- Frequency-hopping systems (FH-MFSK)
- Nonlinear channels, higher throughput needed

**Avoid When**:
- Bandwidth very limited
- Power limited
- Frequency stability poor

---

### 4-PSK / QPSK

**How It Works**:
```
2 bits → One of 4 phases

00 → 45° (or 0°, depending on Gray coding)
01 → 135°
10 → 225°
11 → 315°

Two bits transmitted per symbol
Same symbol rate as BPSK, double throughput
```

**Pros**:
- **Bandwidth efficient**: Same as BPSK, but 2 bits/symbol
- **Power efficient**: Same SNR as BPSK for BER
- **Widely used**: Standard in satellite, cellular (LTE, 5G)
- **Mature technology**: Hardware readily available
- **Gray coding**: Minimizes bit errors (adjacent symbols differ by 1 bit)
- **Differential version**: DQPSK avoids carrier recovery

**Cons**:
- **Coherent detection**: Requires I/Q demodulation
- **Carrier recovery**: More complex than BPSK
- **Phase ambiguity**: 4-fold (90° steps), needs resolution
- **Phase noise sensitive**: Tighter constellation than BPSK

**Best For**:
- **Optimal choice** for most applications needing 2 bits/symbol
- Satellite communications
- Digital TV (DVB-S2)
- Wireless standards (Wi-Fi, LTE)
- Power and bandwidth constrained

**Avoid When**:
- Very simple receiver needed
- Extreme phase noise
- Non-coherent detection required

---

### 8-ary DSSS (8DSSS)

**How It Works**:
```
3 bits → Select one of 8 orthogonal codes

000 → Code_0
001 → Code_1
...
111 → Code_7

8 correlators, pick maximum
```

**Pros**:
- **3× throughput vs BDSSS**: Triple data rate, same bandwidth
- **Processing gain maintained**: Still spread spectrum benefits
- **Can multiplex 8 users**: CDMA capacity

**Cons**:
- **6 dB SNR penalty vs BDSSS**: Much tighter code spacing
- **8 correlators**: Complex receiver
- **Code orthogonality very critical**: Small errors cause confusion
- **Not commonly used**: Binary and 4-ary more practical
- **Multipath severely degrades**: Loss of orthogonality

**Best For**:
- Research / specialized applications
- When DSSS needed and high data rate required
- High SNR environments (> 10 dB)

**Avoid When**:
- SNR limited
- Multipath channel
- Standard equipment preferred

---

### 8-ary FSK (8FSK)

**How It Works**:
```
3 bits → Select one of 8 frequencies

000 → f_0
001 → f_1
...
111 → f_7

Span: 3.5 kHz (with 500 Hz spacing)
```

**Pros**:
- **High throughput**: 3 bits/symbol
- **Non-coherent possible**: Still simple detection
- **Frequency diversity**: 8 independent tones

**Cons**:
- **Very wide bandwidth**: 4× wider than BFSK
- **Poor power efficiency**: ~6 dB worse than 8PSK
- **8 frequency detectors**: Complex
- **Frequency stability critical**: Tight spacing with wide span
- **Rarely used**: Better alternatives exist (8PSK, OFDM)

**Best For**:
- Legacy systems (rarely used today)
- Non-coherent high data rate (niche)

**Avoid When**:
- Almost always avoid (8PSK or OFDM better)
- Bandwidth limited
- Power limited

---

### 8-PSK

**How It Works**:
```
3 bits → One of 8 phases

000 → 0°
001 → 45°
010 → 90°
...
111 → 315°

45° spacing (tight)
```

**Pros**:
- **Bandwidth efficient**: Same as BPSK/QPSK, 3 bits/symbol
- **Higher throughput**: 1.5× QPSK data rate
- **Used in practice**: Satellite (DVB-S2), some cellular

**Cons**:
- **Power penalty**: ~4 dB worse than QPSK
- **Tight constellation**: 45° spacing, sensitive to noise
- **High linearity required**: AM-PM distortion critical
- **Phase noise sensitive**: Small phase error causes wrong symbol
- **Amplifier backoff needed**: 3-5 dB, reduces efficiency

**Best For**:
- High data rate with bandwidth constraint
- Good SNR (> 15 dB)
- High-quality transceivers (low phase noise)
- Satellite downlinks (high power available)

**Avoid When**:
- SNR limited (< 12 dB)
- Phase noise high
- Low-cost, simple equipment
- Power efficiency critical

---

## Selection Guide by Application

### Power Limited (Satellite Uplink, Battery Devices)
```
Ranking (best to worst):
1. BPSK / 4PSK (tie) - 9.6 dB
2. BDSSS - 9.6 dB (with processing gain advantage)
3. 4DSSS - 12.6 dB
4. BFSK - 13.5 dB
5. 8PSK - 13.9 dB
6. 8DSSS - 15.6 dB
7. 4FSK - 16.5 dB
8. 8FSK - 19.5 dB

Recommendation: QPSK (4PSK) for best efficiency + throughput
```

### Bandwidth Limited (Spectrum Auction, Licensed Bands)
```
Ranking (best to worst):
1. 8PSK - 1,500 Hz for 3 bits/symbol
2. 4PSK - 1,500 Hz for 2 bits/symbol
3. BPSK - 1,500 Hz for 1 bit/symbol
4. BFSK - 3,000 Hz for 1 bit/symbol
5. 4FSK - 6,000 Hz for 2 bits/symbol
6. BDSSS - 10,500 Hz for 1 bit/symbol
7. 4DSSS - 10,500 Hz for 2 bits/symbol
8. 8DSSS - 10,500 Hz for 3 bits/symbol
9. 8FSK - 12,000 Hz for 3 bits/symbol

Recommendation: 4PSK or 8PSK (if SNR permits)
```

### Interference Heavy (ISM Bands, Urban RF)
```
Ranking (best to worst):
1. BDSSS - 8.45 dB processing gain
2. 4DSSS - 8.45 dB processing gain, higher throughput
3. 8DSSS - 8.45 dB processing gain, highest throughput
4. (All others have no spreading gain)

Recommendation: BDSSS or 4DSSS depending on SNR
```

### Nonlinear Channel (Saturating Amplifier, Magnetic Recording)
```
Ranking (best to worst):
1. BFSK / 4FSK / 8FSK - immune to amplitude distortion
2. BDSSS / 4DSSS / 8DSSS - constant envelope, some tolerance
3. BPSK / 4PSK - moderate tolerance (constant envelope)
4. 8PSK - sensitive to phase distortion

Recommendation: FSK family for extreme nonlinearity
```

### Low SNR (< 5 dB)
```
Ranking (best to worst):
1. BDSSS - Can operate at negative SNR
2. BPSK - 9.6 dB required
3. 4PSK - 9.6 dB required
4. 4DSSS - 12.6 dB required
5. BFSK - 13.5 dB required
6. Others require > 13 dB

Recommendation: BDSSS for sub-5 dB operation
```

### Simple Implementation (Low Cost, Low Power MCU)
```
Ranking (easiest to hardest):
1. BFSK - frequency discriminator, non-coherent
2. BPSK - simple correlator
3. 4FSK - 4 discriminators
4. 4PSK - I/Q demodulator, carrier recovery
5. BDSSS - code correlation, synchronization
6. 8PSK - tight constellation, high precision
7. 4DSSS - 4 correlators
8. 8DSSS / 8FSK - very complex

Recommendation: BFSK for simplest, BPSK if coherent OK
```

### High Throughput (Fiber Backhaul, Point-to-Point)
```
Ranking (bits/symbol / bandwidth):
1. 8PSK - 3 bits / 1,500 Hz = 2.0 bits/s/Hz
2. 4PSK - 2 bits / 1,500 Hz = 1.33 bits/s/Hz
3. 8DSSS - 3 bits / 10,500 Hz = 0.29 bits/s/Hz
4. 4DSSS - 2 bits / 10,500 Hz = 0.19 bits/s/Hz
5. 8FSK - 3 bits / 12,000 Hz = 0.25 bits/s/Hz
6. 4FSK - 2 bits / 6,000 Hz = 0.33 bits/s/Hz
7. BPSK - 1 bit / 1,500 Hz = 0.67 bits/s/Hz
8. BFSK - 1 bit / 3,000 Hz = 0.33 bits/s/Hz
9. BDSSS - 1 bit / 10,500 Hz = 0.095 bits/s/Hz

Recommendation: 8PSK or higher-order QAM
```

---

## Hybrid and Advanced Schemes

### Frequency-Hopped Spread Spectrum (FH-FSK)
```
Combine FSK with frequency hopping

Advantages:
  - FSK nonlinearity tolerance
  - Frequency diversity (hopping)
  - Anti-jam (hop pattern unknown to adversary)
  - Multiple access (different hop sequences)

Bandwidth: Wide (entire hop band)
Complexity: Medium (frequency synthesizer)
Use: Military, Bluetooth (79 channels × 1 MHz)
```

### OFDM (Orthogonal Frequency Division Multiplexing)
```
Divide bandwidth into N orthogonal subcarriers
Each subcarrier uses PSK or QAM

Advantages:
  - Frequency-selective fading → flat per subcarrier
  - High spectral efficiency
  - Adaptive per-subcarrier (water-filling)
  - Computationally efficient (FFT/IFFT)

Disadvantages:
  - High PAPR (multiple tones)
  - Sensitive to frequency offset
  - Complex (FFT, cyclic prefix, pilot tones)

Use: Wi-Fi, LTE, 5G, DSL
```

### Adaptive Modulation
```
Switch scheme based on channel:

Good channel (SNR > 20 dB):
  → 8PSK or 16-QAM

Medium channel (SNR 10-20 dB):
  → 4PSK

Poor channel (SNR < 10 dB):
  → BPSK or BDSSS

Combine with adaptive coding for link adaptation
```

---

## Summary: Quick Selection Matrix

| Requirement | First Choice | Second Choice | Avoid |
|-------------|--------------|---------------|-------|
| **Power efficiency** | 4PSK (QPSK) | BPSK | FSK family |
| **Bandwidth efficiency** | 8PSK | 4PSK | DSSS family |
| **Anti-jam / LPI** | BDSSS | 4DSSS | All others |
| **Simplicity** | BFSK | BPSK | 8DSSS |
| **Nonlinearity** | FSK family | DSSS family | 8PSK |
| **Low SNR (< 5 dB)** | BDSSS | BPSK | 8FSK |
| **High throughput** | 8PSK | OFDM-QAM | BDSSS |
| **Multiple access** | DSSS family | FH-FSK | PSK family |

---

## Conclusion

**Most Practical Schemes** (widely used):
1. **4PSK (QPSK)**: Best all-around for most applications
2. **BPSK**: When simplicity or power efficiency critical
3. **8PSK**: When bandwidth limited, good SNR available
4. **BDSSS**: Military, anti-jam, CDMA applications

**Niche/Specialized**:
- **FSK family**: Legacy systems, very simple receivers, extreme nonlinearity
- **M-ary DSSS (M>2)**: Research, specialized CDMA

**Rarely Used Today**:
- **8FSK**: Bandwidth inefficient, power inefficient
- **8DSSS**: Complex, better alternatives exist

**Future Trends**:
- Higher-order modulation (16-QAM, 64-QAM, 256-QAM) with OFDM
- Adaptive coding and modulation (ACM)
- MIMO (multiple antenna diversity)
- Cognitive radio (dynamic scheme selection)

References:
- MD in Telecommunications physics
- Claude hallucinations (did not verify the math, trust me bro)
- https://clem.dii.unisi.it/~vipp/website_resources/courses/information_theory/Lectures_Information_Theory_Coding_Barni_Tondi.pdf
