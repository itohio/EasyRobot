# M-ASK Modulator/Demodulator Implementation Plan

## Overview

This document outlines the implementation plan for M-ASK (M-level Amplitude Shift Keying) modulator and demodulator following the patterns established in `dsss`, `mfsk`, and `mpsk` packages. The implementation must support streaming processing via `chain.Processor` interface with `BitsReader`/`BitsWriter` semantics.

## Key Requirements

### Modulator Requirements
1. Validate `src` implements `buffer.BitsReader` and `BitsLength()` matches modulator settings
2. Read bytes one symbol at a time (assume `buffer.Bits` or `buffer.BitRing` type)
3. Read into array of size 1, mask into appropriate number of bits before using to fetch symbol code
4. Maintain state: phase, current symbol portion written, preamble progress
5. Write to `dst` samples as `float32`
6. Write the number of samples it takes to encode each symbol
7. If no more samples left in `src` - return number of samples written, continue modulating with subsequent call
8. If `dst` is `nil` - calculate maximum number of samples needed to encode the source
9. If `dst` has not enough samples - continue when more space is provided in subsequent call
10. Pattern: call `Process` in a loop giving portions of a stream of data

### Demodulator Requirements
1. Validate `dst` implements `buffer.BitsWriter`
2. Read as many samples as needed to maintain internal state (transmission detection, symbol tracking, signal decoding)
3. Keep buffer small enough to be useful (not thousands of symbols)
4. When symbol correlation is detected, write to `dst`
5. If no more samples available - return number of decoded bytes written, continue demodulating with subsequent call
6. If no more space left in `dst` - return number of decoded bytes written, continue with subsequent `Process` call
7. Demodulator should not write bytes if there is no transmission detected
8. Write bytes when transmission is detected and being decoded
9. Stop decoding once transmission is no longer detected (wait for next transmission in stream)

### Common Requirements
- Reading should continue as long as either `src` is exhausted or `dst` is exhausted
- Return number of elements written to `dst`
- Modulator/Demodulator need preamble (alternating symbols) to achieve sync
- Demodulator ideally should not write preamble bytes to `dst`
- Use generators from `dsp` package
- Preamble is absolutely essential - measure reference amplitude from preamble and use it to normalize symbol amplitudes

## Test Cases

### Basic Roundtrip Tests
1. **Modulate one byte infSNR**
   - Surround transmission samples with 0.5s of silence
   - Demodulate and receive one byte that matches transmitted byte
   - Split the input buffer into three parts (call `demodulator.Process` three times with advancing the buffer - instantiate new buffer passed as `src` with different slice of samples buffer)

2. **Modulate four bytes infSNR**

3. **Modulate 33 bytes infSNR**

### Noise Tests
4. **7 bytes with noise added - SNR 30dB**

5. **7 bytes with noise added - SNR 35dB**

6. **7 bytes with 35dB SNR** (if theoretically possible given modulation scheme and symbol size)

### Test Parameters
- Sample rate: 22kHz
- Carrier frequency: 10kHz
- Bits per symbol: 2, 4, 8 (for 7 bytes with 30dB SNR)

## Implementation Structure

### Files to Create

1. **`config.go`** - Configuration structure and validation
2. **`modulator.go`** - M-ASK modulator implementation
3. **`demodulator.go`** - M-ASK demodulator implementation
4. **`waveform.go`** - Symbol envelope/pulse shaping functions
5. **`math.go`** - Mathematical utilities (amplitude levels, quantization, etc.)
6. **`roundtrip_test.go`** - Roundtrip tests with noise scenarios

### Configuration (`config.go`)

```go
type Config struct {
    SampleRate       float32
    CarrierHz        float32
    BitsPerSymbol    int
    SamplesPerSymbol int
    Amplitude        float32
    
    // Amplitude level configuration
    AmplitudeLevels  []float32  // Pre-computed amplitude levels A_0, A_1, ..., A_{M-1}
    LevelSpacing     string     // "even" or "power_efficient"
    MinAmplitude     float32    // For power-efficient (non-zero minimum)
    
    // Pulse shaping
    Envelope         string     // "rectangular", "raised_cosine", "gaussian"
    RaisedCosineBeta float32
    
    // Preamble
    PreambleLength   int
    PreamblePattern  []int      // Alternating pattern for sync
    
    // Demodulator parameters
    EnergyThreshold  float32    // For transmission detection
    TransmissionGap  int        // Symbols without signal before declaring transmission end
    AGCWindow       int        // Samples for AGC amplitude estimation
    CoherentMode     bool       // true for coherent, false for envelope detection
    
    // Carrier recovery (for coherent mode)
    LoopBandwidth    float32
    LoopDamping      float32
    
    // Timing recovery
    TimingLoopBandwidth float32
}
```

**Key Methods:**
- `Validate() error` - Validate and apply defaults
- `SymbolRate() float32` - Calculate symbol rate
- `SamplesForBits(bits int) int` - Calculate samples needed for bits
- `RequiredBandwidth() float32` - Calculate theoretical bandwidth
- `SymbolSeparationMetric() float32` - Calculate amplitude separation between adjacent symbols
- `MinimumEbN0(targetBER float32) float32` - Theoretical minimum Eb/N0
- `MinimumSNR(targetBER, marginDb float32) float32` - Theoretical minimum SNR
- `IsSNRSufficient(snrDb, targetBER, marginDb float32) bool` - Check if SNR is sufficient

**Amplitude Level Calculation:**
- For evenly spaced: `A_m = m * A_max / (M-1)` for m = 0, 1, ..., M-1
- For power-efficient: `A_m = A_min + m * (A_max - A_min) / (M-1)`
- M=2 (OOK): `A_0 = 0`, `A_1 = A_max`
- M=4: `A_0 = 0`, `A_1 = A_max/3`, `A_2 = 2*A_max/3`, `A_3 = A_max`

### Modulator (`modulator.go`)

**State:**
```go
type Modulator struct {
    cfg Config
    
    carrier    dsp.CosineGenerator  // For passband modulation
    envelope   []float32             // Pre-computed symbol envelope
    scratch    []float32             // Scratch buffer for symbol generation
    
    hasPending      bool
    pendingSymbol   int
    pendingSample   int
    
    preambleEmitted int
    tmp [1]byte
}
```

**Key Methods:**
- `NewModulator(cfg Config) (*Modulator, error)` - Constructor
- `Process(dst buffer.Writer[float32], src buffer.Reader[byte]) int` - Main processing
- `nextSymbol(br buffer.BitsReader) (int, bool)` - Read next symbol (preamble or data)
- `renderSymbol(symbol int)` - Generate samples for symbol
- `estimateSamples(br buffer.BitsReader) int` - Estimate samples needed
- `Reset()` - Reset state

**Processing Logic:**
1. Validate `src` implements `BitsReader` and `BitsLength() == cfg.BitsPerSymbol`
2. If `dst == nil`, return `estimateSamples(src)`
3. Loop:
   - If no pending symbol, read next symbol (preamble or data)
   - Render symbol into scratch buffer
   - Write samples to `dst` incrementally
   - Track pending state for partial writes
4. Return total samples written

**Symbol Rendering:**
- For baseband: `s(t) = A_m` (constant amplitude)
- For passband: `s(t) = A_m * cos(2π f_c t + φ)`
- Apply envelope shaping (rectangular, raised cosine, etc.)
- Use `dsp.CosineGenerator` for carrier modulation

### Demodulator (`demodulator.go`)

**State:**
```go
type Demodulator struct {
    cfg Config
    
    // Buffering
    fifo      []float32
    buffered  int
    readBuf   []float32
    
    // Transmission detection
    active           bool
    preambleDetected bool
    preambleProgress int
    silenceCount     int
    
    // AGC (Amplitude normalization)
    agcGain          float32
    agcReference    float32  // Measured from preamble
    agcBuffer       []float32
    
    // Carrier recovery (for coherent mode)
    pllLoop          *pll.Loop
    carrierPhase     float32
    
    // Timing recovery
    timingPhase      float32
    
    tmpSymbol [1]byte
}
```

**Key Methods:**
- `NewDemodulator(cfg Config) (*Demodulator, error)` - Constructor
- `Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int` - Main processing
- `decodeSymbol(samples []float32) int` - Decode symbol from samples
- `detectSymbol(samples []float32) int` - Detect symbol using amplitude quantization
- `updateAGC(samples []float32)` - Update AGC gain from preamble
- `estimateBytes(src buffer.Reader[float32]) int` - Estimate bytes from samples
- `Reset()` - Reset state

**Processing Logic:**
1. Validate `dst` implements `BitsWriter` and `BitsLength() == cfg.BitsPerSymbol`
2. If `dst == nil`, return `estimateBytes(src)`
3. Loop:
   - Read samples into FIFO until we have `SamplesPerSymbol`
   - If not enough samples and source exhausted, break
   - Process symbol from FIFO
   - If transmission not detected, skip writing
   - If preamble not complete, update AGC and skip writing
   - If symbol decoded, write to `dst`
   - Update state (AGC, PLL, timing)
4. Return bytes written

**Symbol Detection:**
- **Coherent mode**: Downconvert, matched filter, quantize amplitude
- **Envelope mode**: Compute envelope `|Y| = sqrt(I² + Q²)`, quantize amplitude
- Quantization: `Y' = Y / A_max` (normalized), then `m = round(Y' * (M-1))` clamped to [0, M-1]
- Use AGC-normalized amplitude for quantization

**Transmission Detection:**
- Compute symbol energy: `E = (1/N) * sum(samples²)`
- If `E > EnergyThreshold`, transmission active
- If `E < EnergyThreshold` for `TransmissionGap` symbols, transmission ended

**AGC (Amplitude Normalization):**
- During preamble, measure amplitude: `A_ref = (1/L_p) * sum(|Y_preamble[k]|)`
- Normalize received signal: `r'[n] = r[n] / A_ref`
- Use normalized amplitude for symbol quantization
- Critical for M-ASK as it's amplitude-sensitive

**Preamble Processing:**
- Preamble pattern: alternating symbols (e.g., [M-1, 0, M-1, 0, ...])
- During preamble:
  - Measure reference amplitude for AGC
  - Track carrier phase (coherent mode)
  - Align symbol timing
  - Do NOT write preamble bytes to `dst`

### Waveform (`waveform.go`)

**Functions:**
- `generateEnvelope(cfg Config) []float32` - Generate symbol envelope
  - Rectangular: all ones
  - Raised cosine: use `dsp` generators or manual calculation
  - Gaussian: use `dsp.GaussianGenerator`
- `applyEnvelope(samples []float32, envelope []float32, amplitude float32)` - Apply envelope to samples

### Math Utilities (`math.go`)

**Functions:**
- `calculateAmplitudeLevels(cfg Config) []float32` - Calculate amplitude levels
- `quantizeAmplitude(value float32, levels []float32) int` - Quantize amplitude to nearest level
- `symbolEnergy(samples []float32) float32` - Calculate symbol energy
- `envelopeDetection(samples []float32, carrierFreq, sampleRate float32) float32` - Envelope detection for non-coherent mode

## Theoretical Calculations

### Bandwidth Calculation
- **Baseband M-ASK**: `B = R_sym * (1 + β)` where β is rolloff factor
- **Passband M-ASK**: Same as baseband (bandwidth doesn't change with carrier)
- For rectangular pulse: `B = R_sym / 2` (null bandwidth)
- For raised cosine: `B = R_sym * (1 + β) / 2`

### SNR Limits
- **Coherent M-ASK**: 
  - Eb/N0 ≈ 9.5 + 10*log10(2*log2(M)) dB for BER=1e-5
  - SNR ≈ Eb/N0 - 10*log10(BitsPerSymbol) + margin
- **Non-coherent (envelope)**: ~3dB worse than coherent
- Different M require different minimum SNR:
  - M=2 (OOK): ~10-12 dB
  - M=4: ~14-16 dB
  - M=8: ~18-20 dB
  - M=16: ~22-24 dB

### Symbol Separation
- For evenly spaced levels: `ΔA = A_max / (M-1)`
- Minimum separation for reliable detection: `ΔA_min = 2 * σ_noise * margin`
- Where `σ_noise = A_max / sqrt(SNR_linear)`
- Symbol separation metric: `ΔA / A_max = 1 / (M-1)`

## Implementation Steps

### Phase 1: Configuration and Math
1. Implement `config.go` with amplitude level calculation
2. Implement `math.go` with quantization and energy calculation
3. Implement `waveform.go` with envelope generation
4. Add theoretical calculation methods (bandwidth, SNR limits, symbol separation)

### Phase 2: Modulator
1. Implement `modulator.go` structure
2. Implement symbol reading (preamble + data)
3. Implement symbol rendering (baseband and passband)
4. Implement streaming processing with state management
5. Add `Reset()` method

### Phase 3: Demodulator (Basic)
1. Implement `demodulator.go` structure
2. Implement sample buffering
3. Implement transmission detection
4. Implement basic symbol detection (envelope mode first)
5. Implement AGC from preamble
6. Implement streaming processing

### Phase 4: Demodulator (Advanced)
1. Add coherent mode with PLL (reuse from `filter/pll`)
2. Add timing recovery
3. Add preamble processing (skip writing preamble bytes)
4. Add transmission gap detection

### Phase 5: Testing
1. Implement basic roundtrip tests (1 byte, 4 bytes, 33 bytes, infSNR)
2. Implement noise tests (30dB, 35dB SNR)
3. Implement split buffer test (three Process calls)
4. Test with different M (2, 4, 8 bits per symbol)
5. Test with 0.5s silence padding

### Phase 6: Optimization
1. Optimize hot paths (avoid allocations in loops)
2. Use SIMD where applicable
3. Optimize amplitude quantization (lookup tables)
4. Profile and optimize critical sections

## Dependencies

- `github.com/itohio/EasyRobot/x/buffer` - BitsReader, BitsWriter, Writer, Reader interfaces
- `github.com/itohio/EasyRobot/x/chain` - Processor interface
- `github.com/itohio/EasyRobot/x/math/dsp` - Generators (CosineGenerator, etc.)
- `github.com/itohio/EasyRobot/x/math/filter/pll` - PLL for carrier recovery (coherent mode)
- `github.com/itohio/EasyRobot/x/math/primitive/fp32` - Optimized float32 operations (if needed)

## Notes

1. **Preamble is Critical**: Must measure reference amplitude from preamble for AGC normalization. M-ASK is very sensitive to amplitude errors.

2. **Coherent vs Non-coherent**: 
   - Coherent mode: Better performance (~3dB gain) but requires PLL
   - Non-coherent (envelope): Simpler, more robust to phase errors, ~3dB worse performance
   - Start with envelope detection, add coherent mode later

3. **Amplitude Sensitivity**: M-ASK is amplitude-based, so AGC is absolutely essential. Without proper normalization, symbol detection will fail.

4. **Different M Require Different SNR**: Higher M (more bits per symbol) requires higher SNR. Test cases should verify theoretical limits.

5. **Symbol Separation**: For M-ASK, symbol separation is the amplitude difference between adjacent levels. This decreases as M increases, making higher M more sensitive to noise.

6. **Streaming Pattern**: Both modulator and demodulator must support incremental processing - reading/writing partial symbols across multiple `Process` calls.

7. **State Management**: Maintain state for:
   - Preamble progress
   - Partial symbol writes (modulator)
   - Sample buffering (demodulator)
   - AGC gain
   - Carrier phase (coherent mode)
   - Timing phase

8. **Error Handling**: Return 0 on validation errors, continue processing on partial reads/writes.

