# Spec: DSSS Encoder / Decoder (High-level, mathematical)

> Purpose: provide a clear, implementation-ready specification for a Direct-Sequence Spread Spectrum (DSSS) encoder and decoder.  
> Audience: DSP / radio engineer implementing encoder/decoder structs that maintain state.  
> Output format: encoder consumes bytes and outputs DAC sample values; decoder consumes ADC samples and outputs recovered bytes (when signal detected). The implementation is samplerate-aware and accepts externally provided spreading codes.

---

## 0. Assumptions & conventions

- Continuous-time signals are represented discretely at sample rate `Fs` (samples/sec).  
- Carrier frequency is `f_c` (Hz). Baseband processing (complex baseband) is assumed preferred: samples are complex (`I+jQ`). If implementing real-passband, map to real cosine/sine carriers as noted.  
- Spreading code sequences are sequences of chips `c[n] ∈ {+1, -1}` (bipolar). Codes are provided to the encoder/decoder as arrays of `±1` (length `L`). For M-ary orthogonal signaling, a symbol selects one of `M` orthogonal codes `C_m[n]`.  
- Symbol rate (data) is `R_sym` (symbols/sec). Each symbol is spread by `L` chips. Chip rate `R_chip = R_sym * L`.  
- Chip period `T_chip = 1 / R_chip`. Sample period `T_s = 1 / Fs`. Samples per chip `N_sc = Fs / R_chip` (must be integer for simplicity, else handle fractional resampling).  
- Use bipolar mapping for modulation: bit `0 -> +1`, bit `1 -> -1` (or vice versa), consistently.

---

## 1. Mathematical model

### 1.1 Encoded continuous-time transmitted signal (baseband view)
Let data bits (or symbols) be `b[k] ∈ {+1, -1}` for k-th symbol (after bit-to-symbol mapping). Let spreading code (for symbol k) be `c_k[n] ∈ {+1, -1}`, n = 0..L-1.

Continuous-time baseband transmitted waveform for symbol k:
\[
s_k(t) = A \cdot b[k] \cdot \sum_{n=0}^{L-1} c_k[n] \, p(t - (kL+n)T_{chip})
\]
where:
- `A` is amplitude scaling,
- `p(t)` is the chip pulse waveform (unit energy or normalized; e.g., rectangular of duration `T_chip` or RRC-shaped pulse),
- chips occupy consecutive `T_chip` intervals.

Overall:
\[
s(t) = \sum_k s_k(t)
\]

If modulating onto real carrier `f_c` (passband):
\[
x(t) = \Re\{ s(t) e^{j 2\pi f_c t} \}
\]

### 1.2 Discrete-time sampled representation
Samples indexed by `m`, `t_m = m T_s`.

Chip index for sample `m`:
\[
n_{chip}(m) = \left\lfloor \frac{m T_s}{T_{chip}} \right\rfloor
\]
Symbol index:
\[
k(m) = \left\lfloor \frac{n_{chip}(m)}{L} \right\rfloor
\]

Discrete sample `s[m]`:
\[
s[m] = A \cdot b[k(m)] \cdot c_{k(m)}[ n_{chip}(m) \bmod L ] \cdot p_s[ m \bmod N_{sc} ]
\]
where `p_s[t]` is the discrete chip pulse sampled at `Fs` (length `N_sc` per chip).

---

## 2. Code generation and requirements

### 2.1 Code types (supported)
- **m-sequences (maximal LFSR)**: generate using LFSR polynomials; length `L = 2^n - 1`. Good autocorrelation, used for long PN.  
- **Gold codes**: combine two preferred m-sequences (XOR) to get large family with controlled cross-correlation. Candidate for many users.  
- **Kasami codes**: smaller cross-correlation sets, often used where low cross-correlation required.  
- **Walsh/Hadamard codes**: strictly orthogonal (length power-of-two) — good for synchronous orthogonal signaling (M-ary orthogonal).  
- **Custom orthogonal sets**: e.g., using Gram-Schmidt on random sequences, or discrete Fourier/Zadoff-Chu sequences for constant amplitude.

### 2.2 Code representation
- Each code is stored as a bit-packed structure with 8 chips per byte (bits `c[0..L-1]` where each element is `0` or `1`).
- For signal processing (modulation/demodulation), codes are converted to bipolar (`+1`/`-1`) on-the-fly using `GetBipolar(index)` method without allocation.
- This representation saves memory (8x reduction: 1 bit per chip vs 1 byte per chip in previous implementation).
- For orthogonal M-ary mapping, provide `M` codes each length `L`: `C_m[n]`. Codes should be pairwise (near-)orthogonal: inner product `Σ_n C_i[n] C_j[n] ≈ 0` for i≠j (computed in bipolar form).

### 2.3 Code requirements for M-ary DSSS

The number of codes required depends on the modulation order (symbol length):

| Modulation Order | Bits per Symbol | Number of Codes Required | Symbol Values |
|-----------------|-----------------|-------------------------|---------------|
| Binary (2-ary)  | 1 bit           | **1 code**              | 0, 1          |
| 4-ary           | 2 bits          | **4 codes**             | 00, 01, 10, 11 |
| 8-ary           | 3 bits          | **8 codes**             | 000..111      |
| 16-ary          | 4 bits          | **16 codes**            | 0000..1111    |
| 32-ary          | 5 bits          | **32 codes**            | 00000..11111  |
| M-ary           | log₂(M) bits    | **M codes**             | 0..M-1        |

**Important constraints:**
- M must be a power of 2 (2, 4, 8, 16, 32, 64, ...)
- All codes must have the same length (spreading factor)
- For binary DSSS: 1 code is reused with ±1 based on bit value
- For M-ary DSSS: Each symbol value (0 to M-1) maps to a different code
- Codes should be orthogonal or near-orthogonal for best performance

### 2.4 Generating codes for M-ary DSSS

#### Binary DSSS (1 code)
```go
// Generate a single m-sequence code
code := GenerateMSequence(3, []int{2, 1, 0}, 1) // Length = 7 chips
modulator := NewModulator(48000, 1000, 1000, code)
```

#### 4-ary DSSS (4 codes)
```go
// Option 1: Walsh codes (orthogonal, deterministic)
codeTable := GenerateWalshCodeTable(4) // 4 orthogonal codes of length 4
modulator := NewModulator(48000, 1000, 1000, codeTable[0], codeTable[1], codeTable[2], codeTable[3])
// Or using spread operator:
modulator := NewModulator(48000, 1000, 1000, codeTable...)

// Option 2: M-sequences with unique seeds (deterministic)
codeTable := GenerateMSequenceCodeTable(3, []int{2, 1, 0}, 4, false) // 4 codes of length 7
modulator := NewModulator(48000, 1000, 1000, codeTable...)

// Option 3: M-sequences with random seeds (unique per session)
codeTable := GenerateMSequenceCodeTable(3, []int{2, 1, 0}, 4, true) // 4 unique codes
modulator := NewModulator(48000, 1000, 1000, codeTable...)
```

#### 8-ary DSSS (8 codes)
```go
// Using GetUniqueSymbolCodeTable helper
codeTable := GetUniqueSymbolCodeTable("walsh", 8, 8, false) // 8 Walsh codes
modulator := NewModulator(48000, 1000, 1000, codeTable...)

// Or with unique m-sequences
codeTable := GetUniqueSymbolCodeTable("m-sequence", 8, 15, true) // 8 unique codes of length 15
modulator := NewModulator(48000, 1000, 1000, codeTable...)

// Or with Gold codes
codeTable := GetUniqueSymbolCodeTable("gold", 8, 15, true) // 8 unique Gold codes
modulator := NewModulator(48000, 1000, 1000, codeTable...)
```

#### 16-ary DSSS (16 codes)
```go
// Walsh codes (orthogonal, length = 16)
codeTable := GenerateWalshCodeTable(16) // 16 codes of length 16
modulator := NewModulator(48000, 1000, 1000, codeTable...)

// M-sequences (longer codes, better autocorrelation)
codeTable := GenerateMSequenceCodeTable(4, []int{3, 2, 0}, 16, false) // 16 codes of length 15
modulator := NewModulator(48000, 1000, 1000, codeTable...)
```

**Code selection guidelines:**
- **Walsh codes**: Best for synchronous systems, strictly orthogonal, length must equal order (power of 2)
- **M-sequences**: Good autocorrelation, can generate many codes with different seeds, length = 2^n - 1
- **Gold codes**: Good for CDMA, controlled cross-correlation, large code families
- **For unique codes**: Use `useRandom=true` in code generation functions to get different codes per session

### 2.5 Generation APIs (mathematical spec)
- **LFSR polynomial**: given register length `n` and tap positions `T = {t1, t2,...}`, produce sequence `s_i` where `s_{i+n} = XOR_{t∈T}(s_{i+t})`. Output is bits (0/1), stored in bit-packed `Code` struct.
- **Gold code**: given two m-sequences `a[n]`, `b[n]` of same length, produce `g[n] = a[n] XOR b[n⊕τ]` (τ = shift). Output is bits (0/1).
- **Walsh matrix**: generate Hadamard matrix of order `L=2^m`; rows are Walsh codes. Output is bits (0/1).

**Code generation functions:**
- `GenerateMSequence(n, taps, seed)`: Generate single m-sequence code
- `GenerateMSequenceCodeTable(n, taps, numCodes, useRandomSeeds)`: Generate table of m-sequence codes
- `GenerateGoldCode(a, b, shift)`: Generate single Gold code
- `GenerateGoldCodeTable(n, tapsA, tapsB, seedA, seedB, numCodes, useRandomShifts)`: Generate table of Gold codes
- `GenerateWalshCode(order, row)`: Generate single Walsh code
- `GenerateWalshCodeTable(order)`: Generate complete Walsh code table
- `GetUniqueSymbolCodeTable(method, order, codeLength, useRandom)`: High-level helper to generate code tables

**Notes:**
- Ensure seed/state choice avoids trivial all-zero LFSR state.
- Provide code normalization if needed: divide by `sqrt(L)` for unit energy when used in correlation (normalization operates on bipolar representation).
- **Unique code generation**: To generate unique codes across sessions/instances:
  - For m-sequences: use `GenerateMSequenceCodeTable(n, taps, numCodes, useRandomSeeds=true)` to generate random seeds via `crypto/rand`.
  - For Gold codes: use `GenerateGoldCodeTable(n, tapsA, tapsB, seedA, seedB, numCodes, useRandomShifts=true)` to generate random shifts, or use random seeds for the m-sequences.
  - For Walsh codes: codes are deterministic by design (orthogonal set). To get different sets, use different orders or permute the code table after generation.
- Inner product computation works directly with bits, converting to bipolar on-the-fly for efficiency.

---

## 3. Modulation (encoder)

### 3.1 Parameters
- `Fs`: sample rate (Hz)  
- `R_sym`: symbol (bit) rate (symbols/sec)  
- `L`: chips per symbol (spreading factor)  
- `R_chip = R_sym * L` (chip rate)  
- `N_sc = Fs / R_chip` (samples per chip, integer)  
- `p_s[i]` (i=0..N_sc-1): chip pulse samples (unit-energy normalized) — recommended: rectangular or RRC with rolloff α and span in chips.

### 3.2 Encoder struct (abstract spec)
- **State fields**
  - `Fs`, `R_sym`, `L`, `N_sc` (ints/floats)
  - `codes`: reference to codebook (for binary: single code; for M-ary: array)
  - `chip_index` (current chip within symbol) — integer
  - `sample_index_within_chip` — integer
  - `symbol_queue` — FIFO of pending input symbols/bits
  - `carrier_phase` (if passband; float)
  - `amplitude_scale` (float)
  - `pulse_shape` (array of N_sc samples) or pointer to pulse generator
- **Primary methods**
  - `Init(params)` — set rates, codes, pulse shape
  - `FeedBytes(bytes[])` — queues bits/bytes into `symbol_queue` (define endianness and bit order)
  - `NextSamples(n)` → returns `n` complex samples (or real samples if passband) ready to feed DAC. This method advances internal state (chip_index, sample_index).
  - `Flush()` — pad remaining symbols with zeros or known termination code
  - `Reset()` — clear state

### 3.3 Encoder operation (algorithm)
- On `FeedBytes` convert bytes into symbol sequence `b[k] ∈ {+1,-1}` or symbol indices for M-ary. Append to `symbol_queue`.
- On each output sample:
  1. If `symbol_queue` empty, output `0` (or idle code) until more data.
  2. Determine current symbol `b = current symbol`.
  3. Determine chip `c = code[chip_index]` (or `C_m[chip_index]` for M-ary, where m is symbol index).
  4. Compute chip pulse sample value `p = pulse_shape[sample_index_within_chip]`.
  5. Baseband sample contribution: `y = amplitude_scale * b * c * p`. (If M-ary, `b` folded into code selection — no separate `b`).
  6. If implementing carrier: multiply by `e^{j 2π f_c t + φ}` or upconvert to stereo real samples.
  7. Increment `sample_index_within_chip`. If it equals `N_sc`, set to 0 and increment `chip_index`. If `chip_index == L`, reset to 0 and pop next symbol.
- Return samples as float32/float64 (I/Q complex) scaled to DAC range.

### 3.4 Pulse shaping recommendations
- Rectangular pulse (simple) low complexity. For spectral control, use root-raised-cosine (RRC) with roll-off α (e.g., 0.2–0.35). Ensure `pulse_shape` length = `N_sc` and normalized energy per chip.

---

## 4. Demodulation (decoder)

### 4.1 High-level tasks
1. **Acquisition**: detect the presence of spreading code, estimate code phase (chip offset) and coarse carrier frequency offset (if passband).  
2. **Tracking**: maintain fine synchronization in time (chip/symbol) and frequency/phase (carrier), typically using Delay-Locked Loop (DLL) for timing and Phase-Locked Loop (PLL) or Costas loop for carrier.  
3. **Despreader & symbol decision**: correlate received samples with code to recover symbol energies, decide bits, and output bytes.  
4. **De-framing / FEC**: handle framing, CRC, and optional error-correction.

### 4.2 Decoder struct (abstract spec)
- **State fields**
  - `Fs`, `R_sym`, `L`, `N_sc`
  - `codes`: codebook (same format used by encoder)
  - `acquired` (bool)
  - `code_phase` (chip offset in samples or fractional)
  - `carrier_freq_offset` (Hz)
  - `carrier_phase` (radians)
  - `buffer` (ring buffer of incoming samples, length at least `L * N_sc + margin`)
  - `DLL_state` parameters for timing lock
  - `PLL_state` parameters for carrier lock
  - `integration_window` length (in samples)
  - `detection_threshold` (float)
  - `symbol_buffer` (recovered symbol decisions)
  - `confidence/metrics` (SNR estimate, correlation peak value)
- **Primary methods**
  - `Init(params)` — set rates, codes, thresholds
  - `FeedSamples(samples[])` — push ADC samples into internal buffer and process (non-blocking)
  - `Process()` — attempt acquisition/tracking/decoding using buffered samples; returns any newly recovered bytes
  - `GetMetrics()` — returns SNR estimate, lock status, correlation peaks
  - `Reset()` — reset all state

### 4.3 Acquisition algorithm (mathematical)
- Goal: find `τ` (code phase) and coarse `f_est` (carrier offset) maximizing correlation.
- For baseband complex samples `r[m]`:
  - For each candidate code phase offset (in sample units) `τ` over 0..(`L*N_sc -1`):
    - Compute correlation `R(τ) = Σ_{n=0}^{L-1} Σ_{i=0}^{N_sc-1} r[ m0 + τ + n*N_sc + i ] * p_s[i] * c[n]` (complex inner product)
    - Optionally, compute magnitude squared `|R(τ)|^2`.
  - To discover carrier offset, either:
    - perform this correlation across a grid of frequency bins (multiply `r` by `e^{-j2π f t}` candidates), or
    - use FFT-based method: take block, multiply by code, and FFT to search frequency.
- Detection criterion: if `|R(τ)|` (peak) exceeds `detection_threshold * rms_noise`, declare acquisition, set `code_phase = τ`, `carrier_phase = angle(R(τ))`, and compute coarse `carrier_freq_offset` from phase drift if multiple blocks used.
- Complexity reduction: use FFT convolution: correlation across many τ can be computed rapidly by FFT-based matched-filtering.

### 4.4 Tracking loops
- **DLL (Delay-Locked Loop) for timing**:
  - Use early/late correlators:
    - `E = correlation at (code_phase - δ)`, `L = correlation at (code_phase + δ)`, `P = correlation at code_phase`.
    - Timing error estimate: `ε_t = (|E| - |L|) / (|E| + |L|)` (or atan version).
    - Update `code_phase` by `code_phase += K_t * ε_t`.
- **PLL / Carrier tracking**:
  - After coarse freq estimate, use Costas loop: multiply incoming sample by local carrier `e^{-j θ}`, compute phase error from I/Q, integrate to update `carrier_phase` and `carrier_freq_offset`.
  - Phase error estimator: `ε_φ = atan2(Imag(P), Real(P))` for BPSK-like modulation.

### 4.5 Despreading & symbol decision
- With synchronized `code_phase` and carrier phase/freq corrected:
  - Integrate over symbol window:
    \[
    z[k] = \sum_{n=0}^{L-1} \sum_{i=0}^{N_{sc}-1} r[ t_{start} + n N_{sc} + i ] \cdot p_s[i] \cdot c[n] \cdot e^{-j (2\pi f_{c_{est}} t + \phi_{est})}
    \]
  - `z[k]` is complex symbol energy; for BPSK data mapping, decision `b_hat = +1` if Re(z[k])>0 else -1.
  - For M-ary orthogonal: compute inner product with each code `C_m`: `Z_m = Σ r * C_m` and choose `m` maximizing `|Z_m|`.
- Output bit/byte packing: define symbol-to-bit mapping and handle endianness.

### 4.6 Detection & false alarm control
- Maintain noise power estimate `σ^2` by measuring off-peak correlation energy or using background samples.  
- Use detection threshold `T = k * σ` where `k` determined from desired P_FA. Empirically tune `k` (e.g., 6–8 dB margin for reliable detection at low SNR).  
- Provide confidence metric: `SNR_est ≈ (|R_peak|^2 / (L * noise_power)) - 1`.

### 4.7 Frame alignment and byte extraction
- Require known frame preamble (short unique spreading-coded symbol or known PN segment). Use correlator to detect preamble, then start demodulation for fixed number of symbols or until CRC.  
- Implement bit/byte buffering with sequence numbers / CRC check. On CRC failure, re-acquisition.

---

## 5. Implementation considerations & best practices

### 5.1 Sample rate and integer ratios
- Prefer `Fs` such that `N_sc` is integer. If not, use resampling (rational resampler) or fractional-N timing with interpolation filters.
- Keep internal buffers at least `L * N_sc * 3` samples to allow acquisition windows and loop latency.

### 5.2 Complexity / FFT optimization
- For long codes, use FFT-based correlation:
  - Precompute FFT(code_padded) and multiply with FFT(received_block) to compute cross-correlation fast.
  - Use overlap-save or overlap-add for streaming correlation.
- For multi-frequency search, compute correlation in frequency domain across bins, or use coarse FFT followed by fine time-domain refinement.

### 5.3 Fixed-point vs floating
- For embedded targets, consider fixed-point arithmetic:
  - Normalize all gains and use headroom to avoid overflow.
  - Use Q-format with sufficient bits for correlation accumulators (`bits >= log2(L * N_sc) + dynamic headroom`).

### 5.4 Pulse shaping and spectral mask
- RRC shaping reduces out-of-band emissions and inter-chip interference. If using RRC, reconstruct pulse response `p_s` carefully and ensure matched filtering at the decoder uses the same pulse.

### 5.5 Handling Doppler / frequency offset
- For passband with mobile platforms, carrier Doppler may be significant. Use wide acquisition frequency grids or frequency-locked loops to track drift.

### 5.6 Multi-user separation (multiple codes)
- If receiver must handle multiple simultaneous users/codes, implement successive interference cancellation (SIC) or perform multi-code correlators in parallel. Ensure near-orthogonality of codebook.

### 5.7 Security (LPI & code secrecy)
- For LPI / encrypted codes, treat codebook as secret; implement secure key storage and code hopping if needed. Implement code generation from key material (e.g., seed for LFSR) to allow synchronized pseudo-random code sequences.

---

## 6. API / Data contracts

### Encoder API
```go
// Unified constructor with varargs
NewModulator(sampleRate, carrierFreq, symbolRate float32, codes ...Code) *Modulator

// Binary DSSS: 1 code
code := GenerateMSequence(3, []int{2, 1, 0}, 1)
modulator := NewModulator(48000, 1000, 1000, code)

// M-ary DSSS: M codes (M must be power of 2)
codeTable := GenerateWalshCodeTable(4) // 4 codes
modulator := NewModulator(48000, 1000, 1000, codeTable...)

// Methods
modulator.Encode(amplitudes vecTypes.Vector, data []byte) int
modulator.Reset()
modulator.SetAmplitude(amplitude float32)
modulator.ModulationOrder() int
modulator.SpreadingFactor() int
```

### Decoder API
```go
// Unified constructor with varargs
NewDemodulator(sampleRate, carrierFreq, symbolRate float32, codes ...Code) *Demodulator

// Binary DSSS: 1 code (must match encoder)
code := GenerateMSequence(3, []int{2, 1, 0}, 1)
demodulator := NewDemodulator(48000, 1000, 1000, code)

// M-ary DSSS: M codes in same order as encoder
codeTable := GenerateWalshCodeTable(4)
demodulator := NewDemodulator(48000, 1000, 1000, codeTable...)

// Methods
demodulator.Decode(data []byte, amplitudes vecTypes.Vector) int
demodulator.Reset()
demodulator.IsLocked() bool
demodulator.OnIsEndOfPacket(callback func([]byte) bool)
```

**Code requirements:**
- **Binary DSSS**: 1 code (reused with ±1 based on bit value)
- **M-ary DSSS**: M codes where M = 2^k (k bits per symbol)
  - Codes must be in order: code[0] for symbol 0, code[1] for symbol 1, ..., code[M-1] for symbol M-1
  - All codes must have the same length (spreading factor)
  - Codes should be orthogonal or near-orthogonal for best performance
  - Encoder and decoder must use the same codes in the same order

**Notes:**
- All timestamps are sample-index-relative; provide API hooks to map sample index to wall-clock if needed.
- The varargs API automatically detects binary vs M-ary based on number of codes provided.
- M must be a power of 2 (2, 4, 8, 16, 32, 64, ...) for M-ary DSSS.

---

## 7. Test vectors & validation

- Provide deterministic test vectors:
  - small code `L=31` m-sequence, `R_sym=1000` sym/s, `Fs=32000` => `N_sc=32` samples/chip. Generate a short byte sequence, pass through encoder with AWGN channel and verify decoder recovers bytes at given SNR.
- Validate:
  - autocorrelation sidelobes close to theoretical values,
  - correlation-based acquisition peak-to-sidelobe ratio,
  - timing jitter and carrier phase estimation accuracy,
  - bit error rate vs SNR curve matches theoretical CDMA performance.

---

## 8. Example parameter sets (for guidance)
- **Low-rate LPI**: `R_sym = 10` sps, `L = 1023`, `Fs = 1023*8*10 = 81840` Hz (i.e. `N_sc=8`) — extreme spreading, very low PSD.  
- **GNSS-like**: `R_sym = 50` bps, `L = 1023`, `Fs ≈ 1.023e6` for classic C/A-like sampling; but for embedded testbeds lower-rate scaled versions OK.  
- **High-rate data**: `R_sym = 1000` sps, `L = 16` (chip rate 16k), `Fs = 160k` (N_sc=10).

---

## 9. Deliverables for engineer
- Implement `Encoder` and `Decoder` structs per the APIs above. Maintain internal state for timing and carrier loops.  
- Provide selectable code generators (LFSR/Golomb/Gold/Walsh) with seed control.  
- Implement acquisition using FFT correlation and fallback direct correlation in small windows.  
- Implement DLL & PLL tracking loops with configurable gains and time constants.  
- Provide unit tests with synthetic AWGN channel for BER vs SNR validation and deterministic vectors.  
- Provide CLI/utility to export code sequences and sample dumps (for offline inspection).

---

## 10. References (conceptual)
- DSSS theory: correlation properties, CDMA multiuser detection.  
- LFSR / Gold code construction mathematics.  
- Standard tracking loops: DLL (early/late), Costas loop for BPSK.  
- FFT-based convolution/correlation (Overlap-save method).

---

### End of spec
