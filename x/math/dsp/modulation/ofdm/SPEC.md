# Spec: OFDM Modulator and Demodulator Implementation (Engineering / Mathematical)

## 0. Overview & Assumptions

- **OFDM (Orthogonal Frequency Division Multiplexing)**: Multi-carrier modulation scheme that divides available bandwidth into multiple orthogonal subcarriers
- **Transmitter**: Maps data symbols to subcarriers, performs IFFT to generate time-domain samples, adds cyclic prefix (CP) for ISI mitigation
- **Receiver**: Removes CP, performs FFT to recover frequency-domain symbols, equalizes channel, demaps to bits
- **Input/Output**: Complex baseband samples `r[n]` sampled at `F_s` (samples/sec). If input from ADC is real passband, assume external front-end downconverts to complex baseband (IQ)
- **Subcarrier Modulation**: Each subcarrier uses QAM or PSK modulation (configurable bits per subcarrier)
- **Orthogonality**: Subcarriers are spaced at `Δf = 1/T_sym` where `T_sym` is the OFDM symbol duration (excluding CP)
- **Cyclic Prefix**: CP length `N_CP` samples prepended to each OFDM symbol to handle multipath and timing errors

---

## 1. Signal Model (Baseband)

### 1.1 Transmitted Signal

For OFDM symbol index `k`, the transmitted complex baseband signal (excluding CP) is:

\[
s_k[n] = \frac{1}{N} \sum_{m=0}^{N-1} X_k[m] \cdot e^{j 2\pi m n / N}, \quad n = 0, 1, \ldots, N-1
\]

where:
- `N` = number of subcarriers (FFT size)
- `X_k[m]` = complex symbol on subcarrier `m` for symbol `k` (from QAM/PSK constellation)
- `m` = subcarrier index (0 = DC, 1..N/2-1 = positive frequencies, N/2 = Nyquist, N/2+1..N-1 = negative frequencies)

**With Cyclic Prefix**:
\[
s_k^{CP}[n] = \begin{cases}
s_k[n + N - N_{CP}], & n = 0, \ldots, N_{CP}-1 \quad \text{(CP: last N_CP samples)} \\
s_k[n - N_{CP}], & n = N_{CP}, \ldots, N_{CP}+N-1 \quad \text{(OFDM symbol)}
\end{cases}
\]

Total samples per OFDM symbol: `N_{total} = N + N_{CP}`

### 1.2 Received Signal

Receiver observes (with channel `h[n]`, CFO `f_Δ`, and noise `w[n]`):

\[
r[n] = \sum_l h[l] \cdot s_k^{CP}[n-l] \cdot e^{j 2\pi f_Δ n T_s} + w[n]
\]

After CP removal and FFT:

\[
Y_k[m] = H[m] \cdot X_k[m] \cdot e^{j \phi_k[m]} + W_k[m]
\]

where:
- `H[m]` = channel frequency response at subcarrier `m`
- `\phi_k[m]` = phase error from CFO and timing offset
- `W_k[m]` = noise in frequency domain

**Objective**: Estimate `X_k[m]` from `Y_k[m]` using channel estimation and equalization.

---

## 2. High-Level Architecture

### 2.1 Transmitter Chain

1. **Bit-to-Symbol Mapping**: Convert input bits to complex symbols (QAM/PSK constellation)
2. **Subcarrier Allocation**: Map symbols to active subcarriers (skip DC and guard bands)
3. **Pilot Insertion**: Insert known pilot symbols at designated subcarrier positions
4. **IFFT**: Transform frequency-domain symbols to time-domain samples
5. **Cyclic Prefix Addition**: Prepend last `N_CP` samples to symbol
6. **Windowing (Optional)**: Apply windowing function to reduce spectral leakage
7. **Preamble Generation**: Generate training symbols for synchronization

### 2.2 Receiver Chain

1. **Front-end Processing**: AGC, DC blocking, anti-aliasing
2. **Frame Detection**: Detect OFDM frame start using preamble correlation
3. **Coarse Synchronization**: Estimate symbol timing and CFO using preamble
4. **Fine Synchronization**: Track timing and CFO using pilot symbols
5. **CP Removal**: Strip cyclic prefix from each OFDM symbol
6. **FFT**: Transform time-domain samples to frequency-domain symbols
7. **Channel Estimation**: Estimate `H[m]` using pilot symbols
8. **Equalization**: Compensate channel: `X̂_k[m] = Y_k[m] / Ĥ[m]`
9. **CFO Compensation**: Rotate symbols to correct residual frequency offset
10. **Symbol Demapping**: Convert equalized symbols to bits
11. **Frame Assembly**: Extract payload, verify CRC

---

## 3. Key Mathematical Blocks

### 3.1 IFFT (Inverse Fast Fourier Transform)

For modulator, compute time-domain samples from frequency-domain symbols:

\[
s[n] = \frac{1}{N} \sum_{m=0}^{N-1} X[m] \cdot e^{j 2\pi m n / N}
\]

**Implementation**: Use radix-2 FFT library (e.g., `x/math/dsp/fft`) with size `N` (power of 2 recommended)

**Scaling**: IFFT typically includes `1/N` normalization factor

### 3.2 FFT (Fast Fourier Transform)

For demodulator, compute frequency-domain symbols from time-domain samples:

\[
Y[m] = \sum_{n=0}^{N-1} r[n] \cdot e^{-j 2\pi m n / N}
\]

**Implementation**: Use same FFT library with forward transform

### 3.3 Cyclic Prefix

**Purpose**: 
- Convert linear convolution to circular convolution (eliminates ISI if CP length ≥ channel delay spread)
- Provides timing error tolerance

**Operation**:
- Copy last `N_CP` samples of IFFT output
- Prepend to symbol: `[s[N-N_CP], ..., s[N-1], s[0], ..., s[N-1]]`

**CP Length Selection**:
- Minimum: `N_CP ≥ L_h - 1` where `L_h` is channel impulse response length
- Typical: `N_CP = N/4` to `N/8` (trade-off between overhead and robustness)

### 3.4 Subcarrier Allocation

**Active Subcarriers**: Not all `N` subcarriers carry data
- **DC subcarrier** (`m=0`): Usually null (DC offset issues)
- **Guard bands**: High-frequency subcarriers near Nyquist may be null
- **Pilot subcarriers**: Known symbols for channel estimation and synchronization
- **Data subcarriers**: Carry user data

**Example for N=64**:
- DC: `m=0` (null)
- Data: `m=1..26, 38..63` (52 subcarriers)
- Pilots: `m=7, 21, 43, 57` (4 pilots)
- Guard: `m=27..37` (null)

### 3.5 Symbol Mapping (QAM/PSK)

Each active subcarrier carries a complex symbol from constellation:

**BPSK** (1 bit/subcarrier):
\[
X \in \{-1, +1\}
\]

**QPSK** (2 bits/subcarrier):
\[
X \in \{\frac{1+j}{\sqrt{2}}, \frac{-1+j}{\sqrt{2}}, \frac{-1-j}{\sqrt{2}}, \frac{1-j}{\sqrt{2}}\}
\]

**16-QAM** (4 bits/subcarrier):
\[
X \in \{\pm 1 \pm j, \pm 1 \pm 3j, \pm 3 \pm j, \pm 3 \pm 3j\} / \sqrt{10}
\]

**64-QAM** (6 bits/subcarrier):
\[
X \in \{\text{64 points on 8×8 grid}\} / \sqrt{42}
\]

**Gray Mapping**: Adjacent constellation points differ by 1 bit (reduces BER)

### 3.6 Channel Estimation

**Pilot-Based Estimation**:

For pilot subcarrier `p` with known symbol `X_p`:

\[
\hat{H}[p] = \frac{Y[p]}{X_p}
\]

**Interpolation**: Estimate `H[m]` for data subcarriers using interpolation:
- Linear interpolation
- DFT-based interpolation (zero-padding in time domain)
- Spline interpolation

**Time-Averaging**: Average estimates across multiple OFDM symbols for noise reduction

### 3.7 Equalization

**Zero-Forcing (ZF)**:
\[
\hat{X}[m] = \frac{Y[m]}{\hat{H}[m]}
\]

**Minimum Mean Square Error (MMSE)**:
\[
\hat{X}[m] = \frac{Y[m] \cdot \hat{H}^*[m]}{|\hat{H}[m]|^2 + \sigma^2}
\]

where `σ²` is noise variance estimate

### 3.8 CFO Estimation and Correction

**Coarse CFO** (from preamble):
\[
\hat{f}_Δ = \frac{1}{2\pi T_{sym}} \angle\left( \sum_{m} Y_1[m] \cdot Y_2^*[m] \right)
\]

where `Y_1`, `Y_2` are two consecutive preamble symbols

**Fine CFO** (from pilots):
\[
\hat{f}_Δ = \frac{1}{2\pi T_{sym}} \angle\left( \sum_{p} Y_k[p] \cdot Y_{k-1}^*[p] \right)
\]

**Correction**: Rotate frequency-domain symbols:
\[
Y_{corrected}[m] = Y[m] \cdot e^{-j 2\pi \hat{f}_Δ m T_s / N}
\]

---

## 4. Synchronization

### 4.1 Frame Detection (Preamble)

**Preamble Structure**: Two identical OFDM symbols (or known pattern)

**Correlation-Based Detection**:
\[
R(\tau) = \left| \sum_{n=0}^{N-1} r[\tau+n] \cdot r^*[\tau+n+N] \right|
\]

Peak at `τ*` indicates frame start

**Threshold**: `R(τ*) > T_detect` where `T_detect` is based on noise estimate

### 4.2 Symbol Timing Recovery

**Coarse Timing**: From preamble correlation peak

**Fine Timing**: Use early/late correlation or Gardner algorithm on CP

**CP-Based Timing**:
\[
\varepsilon_t = \arg\left( \sum_{n=0}^{N_{CP}-1} r[n] \cdot r^*[n+N] \right)
\]

Update timing offset: `τ ← τ + μ · ε_t`

### 4.3 Carrier Frequency Offset (CFO)

**CFO Effects**:
- Subcarrier rotation: `Y[m]` rotated by `e^{j 2π f_Δ m T_s / N}`
- ICI (Inter-Carrier Interference) if CFO is large

**Estimation**: Use preamble or pilot phase differences

**Correction**: Rotate time-domain samples before FFT:
\[
r_{corrected}[n] = r[n] \cdot e^{-j 2\pi \hat{f}_Δ n T_s}
\]

---

## 5. Frame Structure

### 5.1 Preamble

**Short Training Sequence (STS)**: For coarse timing and AGC
- Typically 2-4 identical short symbols
- Length: `N_STS` samples

**Long Training Sequence (LTS)**: For fine timing, CFO, and channel estimation
- Typically 2 identical long symbols (OFDM symbols)
- Known pattern for correlation

### 5.2 Payload

**OFDM Symbols**: Each symbol contains:
- CP: `N_CP` samples
- Data: `N` samples (IFFT output)

**Pilot Pattern**: Pilots inserted every `P` symbols or at fixed subcarrier positions

**CRC**: Cyclic redundancy check for error detection

---

## 6. Structs / API (Abstract)

### 6.1 `OFDMEncoder` (Modulator)

**State Fields**:
- `Fs`, `N` (FFT size), `N_CP` (cyclic prefix length)
- `N_active` (active subcarriers), `N_data` (data subcarriers), `N_pilot` (pilot subcarriers)
- `BitsPerSubcarrier` (1, 2, 4, 6 for BPSK, QPSK, 16-QAM, 64-QAM)
- `Constellation` (QAM/PSK mapping table)
- `PilotPattern` (subcarrier indices for pilots)
- `PreamblePattern` (known preamble symbols)
- `IFFT` (IFFT instance)
- `SymbolBuffer` (complex buffer of size `N`)
- `CPBuffer` (complex buffer of size `N_CP`)
- `SampleBuffer` (float32 buffer of size `N + N_CP`)
- `PreambleGenerated` (bool), `PreambleIndex` (int)
- `CurrentSymbol` (subcarrier symbols), `SymbolIndex` (int)
- `BitBuffer` (partial bits), `BitPos` (int)

**Methods**:
- `Init(params)` — set FFT size, CP length, constellation, pilot pattern
- `Process(dst, src)` — read bytes, map to symbols, IFFT, add CP, write samples
- `GeneratePreamble()` — generate preamble symbols
- `Reset()` — reset state, regenerate preamble

### 6.2 `OFDMDecoder` (Demodulator)

**State Fields**:
- `Fs`, `N`, `N_CP`, `N_active`, `N_data`, `N_pilot`
- `BitsPerSubcarrier`, `Constellation`
- `PilotPattern`, `PreamblePattern`
- `FFT` (FFT instance)
- `SampleBuffer` (float32 buffer, size ≥ `N + N_CP`)
- `SymbolBuffer` (complex buffer of size `N`)
- `ChannelEstimate` (complex buffer of size `N`)
- `State` ∈ {SEARCH, ACQUIRE, TRACK}
- `FrameDetected` (bool), `SymbolOffset` (int)
- `CFOEstimate` (float32), `TimingOffset` (int)
- `PreambleCorrelation` (float32)
- `EqualizerType` ("ZF" or "MMSE")
- `NoiseVariance` (float32)

**Methods**:
- `Init(params)` — set FFT size, CP length, constellation, equalizer type
- `Process(dst, src)` — read samples, detect frame, remove CP, FFT, equalize, demap, write bytes
- `DetectFrame()` — correlate with preamble, estimate timing/CFO
- `EstimateChannel()` — use pilots to estimate `H[m]`
- `Equalize()` — compensate channel using ZF or MMSE
- `Reset()` — reset state, clear buffers

---

## 7. OFDM Parameters and Relationships

### 7.1 Key Parameters

- **FFT Size** `N`: Number of subcarriers (typically power of 2: 64, 128, 256, 512, 1024)
- **Cyclic Prefix Length** `N_CP`: Samples in CP (typically `N/4` to `N/8`)
- **Subcarrier Spacing** `Δf = F_s / N`: Frequency spacing between subcarriers
- **OFDM Symbol Duration** `T_sym = N / F_s`: Duration of one OFDM symbol (excluding CP)
- **Total Symbol Duration** `T_total = (N + N_CP) / F_s`: Including CP
- **Active Subcarriers** `N_active`: Subcarriers carrying data or pilots
- **Data Subcarriers** `N_data`: Subcarriers carrying user data
- **Pilot Subcarriers** `N_pilot`: Subcarriers with known symbols

### 7.2 Bandwidth Calculation

**Occupied Bandwidth**:
\[
B = N_{active} \cdot \Delta f = N_{active} \cdot \frac{F_s}{N}
\]

**Null-to-Null Bandwidth** (with windowing):
\[
B_{null} = (N + N_{CP}) \cdot \Delta f = (N + N_{CP}) \cdot \frac{F_s}{N}
\]

**Spectral Efficiency**:
\[
\eta = \frac{N_{data} \cdot k}{N + N_{CP}} \quad \text{bits/s/Hz}
\]

where `k` = bits per subcarrier

### 7.3 Sample Rate Requirements

**Minimum Sample Rate**:
\[
F_s \geq 2 \cdot B
\]

**Recommended**: `F_s ≥ 2.5 · B` for guard bands and filtering

**Nyquist Constraint**: `F_s ≥ 2 · f_{max}` where `f_{max}` is highest subcarrier frequency

---

## 8. SNR and Performance

### 8.1 SNR Requirements

**Per-Subcarrier SNR**:
\[
SNR_{sub} = \frac{|H[m]|^2 \cdot |X[m]|^2}{\sigma^2}
\]

**Average SNR** (over all subcarriers):
\[
SNR_{avg} = \frac{1}{N_{data}} \sum_{m \in \text{data}} SNR_{sub}[m]
\]

**Theoretical Eb/N0** (for AWGN channel):
- **BPSK**: ~10 dB for BER = 10^-5
- **QPSK**: ~10 dB (same as BPSK, 2x data rate)
- **16-QAM**: ~15 dB
- **64-QAM**: ~20 dB

**Implementation Margin**: Add 2-3 dB for implementation losses (timing errors, CFO, channel estimation errors)

### 8.2 Channel Effects

**Multipath**: CP length must exceed channel delay spread
- If `N_CP < L_h - 1`: ISI occurs, performance degrades

**Frequency Selectivity**: Channel varies across subcarriers
- Equalization compensates per-subcarrier

**CFO**: Causes ICI if not corrected
- Maximum tolerable CFO: `|f_Δ| < Δf / 2` (half subcarrier spacing)

**Phase Noise**: Causes ICI, degrades performance at high frequencies

---

## 9. Implementation Considerations

### 9.1 FFT/IFFT

**Library**: Use `x/math/dsp/fft` or similar radix-2 FFT
- Prefer power-of-2 sizes for efficiency
- Pre-allocate FFT buffers (no allocations in hot path)

**Scaling**: IFFT includes `1/N` normalization
- Verify library scaling behavior

### 9.2 Memory Budget

**Per Modulator Instance**:
- IFFT buffer: `N * 8 bytes` (complex64) = 512 bytes for N=64
- CP buffer: `N_CP * 8 bytes` = 128 bytes for N_CP=16
- Symbol buffer: `N * 8 bytes` = 512 bytes
- **Total**: ~1.2 KB for N=64

**Per Demodulator Instance**:
- FFT buffer: `N * 8 bytes` = 512 bytes
- Sample buffer: `(N + N_CP) * 4 bytes` = 320 bytes (float32)
- Channel estimate: `N * 8 bytes` = 512 bytes
- **Total**: ~1.3 KB for N=64

### 9.3 Processing Complexity

**Per OFDM Symbol**:
- IFFT/FFT: `O(N log N)` operations
- CP add/remove: `O(N_CP)` operations
- Channel estimation: `O(N_pilot)` operations
- Equalization: `O(N_data)` operations
- Symbol demapping: `O(N_data)` operations

**Total**: Dominated by FFT, typically `~5N log N` operations per symbol

---

## 10. Testing and Validation

### 10.1 Unit Tests

- IFFT/FFT correctness with known frequency-domain input
- CP addition/removal with known samples
- Symbol mapping (QAM/PSK) with known bits
- Channel estimation with known channel
- Equalization with known channel and noise

### 10.2 End-to-End Tests

**Roundtrip (Infinite SNR)**:
- 1 byte, 4 bytes, 33 bytes
- Verify perfect reconstruction

**AWGN Channel**:
- 7 bytes with 20 dB, 15 dB, 10 dB SNR
- Measure BER vs theoretical

**Multipath Channel**:
- Two-tap channel with delay < CP length
- Verify ISI mitigation

**CFO Test**:
- Inject CFO ±Δf/4, ±Δf/2
- Verify acquisition and tracking

### 10.3 Performance Metrics

- **BER vs Eb/N0**: Compare to theoretical curves
- **Frame Detection Probability**: P_d vs SNR
- **CFO Estimation Accuracy**: Mean squared error vs SNR
- **Channel Estimation MSE**: vs SNR and pilot density

---

## 11. Example Parameter Sets

### 11.1 Basic OFDM (N=64)

- **FFT Size**: `N = 64`
- **CP Length**: `N_CP = 16` (N/4)
- **Sample Rate**: `F_s = 96,000 Hz`
- **Subcarrier Spacing**: `Δf = 96,000 / 64 = 1,500 Hz`
- **Symbol Duration**: `T_sym = 64 / 96,000 = 0.667 ms`
- **Total Symbol Duration**: `T_total = 80 / 96,000 = 0.833 ms`
- **Active Subcarriers**: `N_active = 52` (skip DC and guards)
- **Data Subcarriers**: `N_data = 48`
- **Pilot Subcarriers**: `N_pilot = 4`
- **Bits Per Subcarrier**: `k = 2` (QPSK)
- **Data Rate**: `48 × 2 / 0.833 ms = 115.2 kbps` (raw, excluding preamble)

### 11.2 High-Rate OFDM (N=256)

- **FFT Size**: `N = 256`
- **CP Length**: `N_CP = 64` (N/4)
- **Sample Rate**: `F_s = 192,000 Hz`
- **Subcarrier Spacing**: `Δf = 192,000 / 256 = 750 Hz`
- **Symbol Duration**: `T_sym = 256 / 192,000 = 1.33 ms`
- **Total Symbol Duration**: `T_total = 320 / 192,000 = 1.67 ms`
- **Active Subcarriers**: `N_active = 200`
- **Data Subcarriers**: `N_data = 192`
- **Pilot Subcarriers**: `N_pilot = 8`
- **Bits Per Subcarrier**: `k = 4` (16-QAM)
- **Data Rate**: `192 × 4 / 1.67 ms = 460.8 kbps` (raw)

---

## 12. Summary of Processing Pipeline

### 12.1 Modulator

1. Read bytes from `BitsReader` (validate interface and `BitsLength`)
2. Map bits to complex symbols (QAM/PSK constellation)
3. Allocate symbols to active subcarriers (skip DC, insert pilots)
4. Perform IFFT to generate time-domain samples
5. Add cyclic prefix (prepend last `N_CP` samples)
6. Write samples to `Writer[float32]`
7. Generate preamble before payload (if not already emitted)

### 12.2 Demodulator

1. Read samples into buffer (maintain `≥ N + N_CP` samples)
2. **SEARCH**: Correlate with preamble, detect frame start
3. **ACQUIRE**: Estimate coarse timing and CFO from preamble
4. **TRACK**: For each OFDM symbol:
   - Remove CP
   - Perform FFT
   - Estimate/update channel using pilots
   - Equalize symbols (ZF or MMSE)
   - Correct residual CFO
   - Demap symbols to bits
   - Write bytes to `BitsWriter`
5. Detect frame end (energy drop or CRC failure)
6. Return to SEARCH state

---

## 13. Theoretical SNR Limits and Practical Margins

### 13.1 Per-Subcarrier SNR Requirements

For AWGN channel with perfect synchronization:

**Theoretical Eb/N0 (dB) for BER = 10^-5:**
- BPSK (1 bit/subcarrier): ~10 dB
- QPSK (2 bits/subcarrier): ~10 dB
- 16-QAM (4 bits/subcarrier): ~15 dB
- 64-QAM (6 bits/subcarrier): ~20 dB

**Implementation Margins:**
- **Standard margin**: 2-3 dB (for timing errors, CFO, channel estimation errors)
- **Multipath margin**: Additional 1-2 dB if CP length is marginal

**Practical Minimum SNR (dB) for BER = 10^-5:**
- BPSK: ~12-13 dB (theoretical 10 + 2-3 margin)
- QPSK: ~12-13 dB
- 16-QAM: ~17-18 dB
- 64-QAM: ~22-23 dB

### 13.2 System-Level SNR

**Input SNR** (at receiver input):
\[
SNR_{in} = \frac{P_{signal}}{P_{noise}}
\]

**Per-Subcarrier SNR** (after FFT):
\[
SNR_{sub} = SNR_{in} \cdot \frac{N_{active}}{N} \cdot \frac{1}{1 + N_{CP}/N}
\]

The factor `N_{active}/N` accounts for power distribution across active subcarriers only.

**Processing Gain**: From symbol integration (FFT):
\[
G_p = 10 \log_{10}(N) \quad \text{dB}
\]

Example: N=64 → G_p ≈ 18 dB

### 13.3 Test Scenarios

**Roundtrip Tests:**
- **Baseline (no noise)**: All OFDM configurations should pass
- **QPSK with noise**: Passes at 12+ dB SNR
- **16-QAM with noise**: Passes at 17+ dB SNR
- **64-QAM with noise**: Requires 22+ dB SNR

**Limitations:**
- CFO tolerance: `|f_Δ| < Δf/2` (half subcarrier spacing)
- CP length must exceed channel delay spread
- Channel estimation accuracy depends on pilot density

---

## 14. Current Implementation Status

### 14.1 What Needs to Be Implemented

**Encoder (`ofdm/encoder.go`):**
- ❌ Bit-to-symbol mapping (QAM/PSK constellation)
- ❌ Subcarrier allocation (active/data/pilot subcarriers)
- ❌ IFFT computation
- ❌ Cyclic prefix addition
- ❌ Preamble generation
- ❌ Frame structure (preamble + payload)

**Decoder (`ofdm/decoder.go`):**
- ❌ Frame detection (preamble correlation)
- ❌ Symbol timing recovery
- ❌ CFO estimation and correction
- ❌ CP removal
- ❌ FFT computation
- ❌ Channel estimation (pilot-based)
- ❌ Equalization (ZF or MMSE)
- ❌ Symbol demapping (QAM/PSK)
- ❌ Frame assembly and CRC verification

**Current Architecture** (to be implemented):
```
Modulator: Bits → Symbol Mapping → Subcarrier Allocation → IFFT → CP Addition → Samples
Demodulator: Samples → Frame Detection → CP Removal → FFT → Channel Estimation → Equalization → Symbol Demapping → Bits
```

---

## 15. Embedded-Friendly Implementation

### 15.1 Design Principles

- **No allocations**: Pre-allocate all buffers at initialization
- **Fixed-point optional**: Use float32/complex64 for simplicity, optimize later if needed
- **Ring buffers**: Use circular indexing for sample buffers
- **State machines**: Clear state transitions (SEARCH → ACQUIRE → TRACK)
- **FFT library**: Use existing `x/math/dsp/fft` or similar

### 15.2 Memory Budget

**Per modulator instance:**
- IFFT buffer: `N * 8 bytes` (complex64)
- CP buffer: `N_CP * 8 bytes`
- Symbol buffer: `N * 8 bytes`
- Sample buffer: `(N + N_CP) * 4 bytes` (float32)
- **Total**: ~2-3 KB for N=64, ~8-12 KB for N=256

**Per demodulator instance:**
- FFT buffer: `N * 8 bytes`
- Sample buffer: `(N + N_CP) * 4 bytes`
- Channel estimate: `N * 8 bytes`
- Correlation buffer: `N * 8 bytes` (for preamble)
- **Total**: ~3-4 KB for N=64, ~12-16 KB for N=256

### 15.3 Processing Pipeline

```
1. SEARCH state:
   - Correlate samples with preamble pattern
   - If correlation > threshold, switch to ACQUIRE

2. ACQUIRE state:
   - Estimate symbol timing from correlation peak
   - Estimate coarse CFO from preamble phase difference
   - Initialize channel estimate (from preamble)
   - Switch to TRACK

3. TRACK state:
   - For each OFDM symbol:
     a. Remove CP
     b. Perform FFT
     c. Update channel estimate using pilots
     d. Equalize symbols (ZF or MMSE)
     e. Correct residual CFO
     f. Demap symbols to bits
     g. Write bytes to output
   - If frame end detected, switch to SEARCH
```

---

### End of Specification

