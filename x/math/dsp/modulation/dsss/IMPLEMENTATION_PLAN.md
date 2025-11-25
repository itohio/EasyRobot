# DSSS Modulator/Demodulator Implementation Plan

## Overview

This document outlines the implementation plan for DSSS (Direct Sequence Spread Spectrum) modulator and demodulator components that implement the `Processor` interface for integration into the EasyRobot signal processing chain.

## Core Components

### 1. Modulator (`Modulator`)

**Type**: `Processor[float32, byte]`

**Responsibilities**:
- Read bytes from `BitsReader` source (validates `BitsReader` interface and `BitsLength` compatibility)
- Convert bytes to symbols based on modulation order (k bits per symbol)
- Map symbols to chip sequences using spreading codes
- Apply pulse shaping (raised cosine for spectral efficiency)
- Upconvert to carrier frequency
- Write float32 samples to destination
- Maintain state for streaming across multiple `Process` calls

**Key Features**:
- **Input Validation**: 
  - Verify `src` implements `BitsReader` interface
  - Verify `src.BitsLength()` matches modulator's expected bits per symbol (k)
  - Return 0 if validation fails
  
- **Symbol-by-Symbol Reading**:
  - Read one symbol at a time from source
  - Assume `buffer.Bits` or `buffer.BitRing` type
  - Read into array of size 1 (single byte)
  - Mask byte to appropriate number of bits (k) before symbol mapping
  - Handle partial symbols (zero padding if needed)
  
- **State Management**:
  - Carrier phase: `phase float32` (maintains continuity across Process calls)
  - Current symbol state: tracks partial symbol bits across calls
  - Chip position: tracks position within current chip sequence
  - Sample position: tracks position within current chip samples
  - Sample counter: global sample count for phase calculation
  - Pulse shaper state: filter memory for raised cosine (if used)
  
- **Streaming Pattern**:
  - If no more samples left in src: return number of samples written, continue with next call
  - If dst has not enough space: return number written, continue with next call
  - Process in loop giving portions of stream data
  
- **Output**:
  - Write float32 samples to `dst`
  - If `dst` is nil: calculate maximum samples needed
  - Formula: `N_total = ⌈(bytes × 8)/k⌉ × L × N_sc`
    - `k` = bits per symbol
    - `L` = spreading factor (code length = 8 for tests)
    - `N_sc` = samples per chip = `⌊f_s / (R_s × L)⌋`
    - `R_s` = symbol rate (derived from parameters)

**Processing Flow**:
```
Byte → Extract k bits → Symbol value → Chip sequence → Pulse shape → Carrier modulation → Samples
```

### 2. Demodulator (`Demodulator`)

**Type**: `Processor[byte, float32]`

**Responsibilities**:
- Read float32 samples from source
- Maintain small internal buffer for symbol correlation (not excessive - keep manageable)
- Detect transmission presence
- Downconvert from carrier frequency
- Correlate with spreading codes to detect symbols
- Track symbol boundaries and timing
- Decode symbols back to bits
- Pack bits into bytes
- Write bytes to `BitsWriter` destination
- Stop decoding when transmission ends

**Key Features**:
- **Input Processing**:
  - Read samples as needed to maintain internal state
  - Keep buffer small enough for efficient processing (e.g., 2-3 symbol periods)
  - Support transmission detection (preamble correlation or energy-based)
  - Track symbol boundaries (coarse and fine timing)
  
- **State Management**:
  - Carrier phase estimate: `phaseEstimate float32` (for coherent demodulation)
  - Symbol timing: `symbolOffset int` (sample offset within symbol period)
  - Correlation buffer: accumulate samples for symbol correlation (small, ~2-3 symbols)
  - Preamble detection state: track if preamble detected
  - Partial symbol state: track incomplete symbols
  - Transmission state: `transmissionDetected bool`, `transmissionActive bool`
  - Sample counter: global sample count
  
- **Symbol Detection**:
  - For each symbol period (L × N_sc samples):
    - Correlate with all M spreading codes
    - Select code with maximum correlation (M-ary) or sign-based decision (binary)
    - Extract symbol value
  - Handle preamble: detect but ignore (don't write to output)
  
- **Transmission Detection**:
  - Detect transmission start: energy threshold or preamble correlation
  - Set `transmissionActive = true` when transmission detected
  - Only write bytes when `transmissionActive == true`
  - Detect transmission end: energy drops below threshold for N consecutive symbols
  - Set `transmissionActive = false` when transmission ends
  - Wait for next transmission in stream
  
- **Streaming Pattern**:
  - If no more samples available: return number of decoded bytes written, continue with next call
  - If no more space in dst: return number written, continue with next call
  - Process in loop giving portions of stream data
  
- **Output**:
  - Write decoded bytes to `dst` (BitsWriter)
  - If `dst` is nil: calculate maximum bytes that could be decoded
  - Formula: `N_bytes = ⌊(samples / (L × N_sc)) × k / 8⌋`
  - Only write when transmission is detected and active

## Implementation Details

### Configuration Structure

```go
type Config struct {
    // Modulation parameters
    BitsPerSymbol int     // k: 1, 2, 3, etc. (log2(M))
    ModulationOrder int   // M: 2, 4, 8, etc. (M = 2^k)
    
    // Spreading codes
    Codes [][]int8        // M codes, each of length L (bipolar: -1, +1)
    SpreadingFactor int   // L: code length (8 for tests)
    
    // Timing parameters
    SampleRate float32    // f_s: 96000 Hz
    SymbolRate float32    // R_s: symbols per second
    CarrierFreq float32   // f_c: 20000 Hz
    SamplesPerChip int    // N_sc = ⌊f_s / (R_s × L)⌋
    
    // Pulse shaping
    PulseShape string     // "rectangular", "raised_cosine", "root_raised_cosine"
    Rolloff float32       // For raised cosine (0.0-1.0), typically 0.35-0.5
    PulseFilterSpan int   // Number of chip periods for pulse shape (typically 4-8)
    
    // Preamble
    PreambleLength int    // Number of preamble symbols
    PreamblePattern []int // Symbol sequence for preamble (alternating for sync)
    
    // Transmission detection
    EnergyThreshold float32    // Energy threshold for transmission detection
    DetectionWindow int        // Samples for energy calculation
    TransmissionTimeout int    // Symbols without signal before declaring transmission end
    
    // Amplitude
    Amplitude float32         // Signal amplitude (typically 1.0)
}
```

### Modulator State

```go
type Modulator struct {
    config Config
    
    // State
    carrierGen *dsp.CosineGenerator  // Carrier generator (maintains phase state)
    sampleCount int      // Global sample counter
    currentByte byte     // Current byte being processed
    bytePos int          // Bit position within current byte (0-7)
    symbolPos int        // Symbol position within current byte
    chipPos int          // Chip position within current symbol
    samplePos int        // Sample position within current chip
    
    // Pulse shaper state (for raised cosine)
    pulseFilter []float32  // Filter coefficients (pre-computed)
    pulseState []float32   // Filter state (delay line)
    pulseStateIdx int      // Current position in delay line
    
    // Pre-computed values
    chipSamples []float32 // Pre-computed pulse shape samples per chip
}
```

**Note**: Using `CosineGenerator` maintains phase continuity automatically and is more efficient than manual phase tracking.

### Demodulator State

```go
type Demodulator struct {
    config Config
    
    // State
    phaseEstimate float32    // Estimated carrier phase
    symbolOffset int         // Sample offset within symbol period
    correlationBuffer []float32 // Buffer for accumulating symbol samples (small, ~2-3 symbols)
    preambleDetected bool     // Whether preamble has been detected
    preambleCount int        // Number of preamble symbols processed
    currentSymbol int        // Current symbol being decoded
    bitBuffer byte           // Accumulated bits for current byte
    bitPos int               // Bit position within current byte (0-7)
    sampleCount int          // Global sample counter
    
    // Transmission detection
    transmissionDetected bool  // Transmission has been detected
    transmissionActive bool    // Currently decoding transmission
    energyAccumulator float32  // Energy accumulator for detection
    silenceCount int          // Consecutive symbols without signal
    
    // Pre-computed values
    codes [][]int8           // Spreading codes (bipolar)
    correlationThreshold float32 // Threshold for symbol detection
}
```

### Key Algorithms

#### Modulator: Symbol to Chip Mapping

```go
// For binary (M=2, k=1):
if symbol == 0 {
    chips = negateCode(codes[0])  // -P₀
} else {
    chips = codes[0]              // +P₀
}

// For M-ary (M>2, k>1):
chips = codes[symbol]  // Direct code selection
```

#### Modulator: Raised Cosine Pulse Shaping

```go
// Pre-compute pulse shape samples for one chip period
// Use raised cosine formula from SPEC.md
// Formula: g(t) = sinc(πt/T_c) × cos(παt/T_c) / (1 - (2αt/T_c)²)

func generateRaisedCosine(rolloff float32, samplesPerChip int, span int) []float32 {
    // Generate pulse shape spanning multiple chip periods
    // Use math32 for float32 operations
    // Use sinc function from filter/fir package or implement inline
}
```

#### Modulator: Carrier Modulation

**Option 1: Use CosineGenerator.GenerateWithOp (Recommended)**
```go
// Create cosine generator for carrier
carrierGen := dsp.NewCosineGenerator(sampleRate, carrierFreq, 1.0, 0.0)

// For each chip's samples, multiply baseband by carrier directly:
carrierGen.GenerateWithOp(basebandSamples[:N_sc], false, func(d *float32, carrierValue float32) {
    *d *= carrierValue
})
```

**Option 2: Use CosineGenerator.Generate with small buffer**
```go
// Create cosine generator for carrier
carrierGen := dsp.NewCosineGenerator(sampleRate, carrierFreq, 1.0, 0.0)

// For each chip's samples:
var carrierSamples [N_sc]float32  // Small stack array
carrierGen.Generate(carrierSamples[:])

// Multiply baseband by carrier
for i := range N_sc {
    samples[i] = basebandSamples[i] * carrierSamples[i]
}
```

**Option 3: Phase Accumulator (Alternative)**
```go
// For each sample n:
phase = 2π × carrierFreq × sampleCount / sampleRate + phaseOffset
sample = basebandSample × cos(phase)

// Phase wrapping:
phase = phase - 2π × floor(phase / 2π)
```

**Note**: `GenerateWithOp` allows direct application of the generated value to existing samples without requiring an intermediate buffer, which is more efficient for hot paths. The closure receives a pointer to the destination element, allowing direct modification.

#### Demodulator: Chip Correlation

```go
// For each code Pᵢ:
correlation = 0
for chipIdx := range spreadingFactor {
    chipStart := symbolOffset + chipIdx × samplesPerChip
    chipEnd := chipStart + samplesPerChip
    
    // Integrate samples over chip duration
    chipSum := integrate(samples[chipStart:chipEnd])
    
    // Multiply by code chip and accumulate
    correlation += chipSum × codes[i][chipIdx]
}

// Normalize
correlation /= (spreadingFactor × samplesPerChip)
```

#### Demodulator: Symbol Decision

```go
// Binary (M=2):
if correlations[0] < 0 {
    symbol = 0
} else {
    symbol = 1
}

// M-ary (M>2):
symbol = argMax(abs(correlations))
```

#### Demodulator: Transmission Detection

```go
// Energy-based detection
energy := calculateEnergy(samples[window])
if energy > energyThreshold && !transmissionDetected {
    transmissionDetected = true
    transmissionActive = true
    silenceCount = 0
}

// Transmission end detection
if transmissionActive {
    if energy < energyThreshold {
        silenceCount++
        if silenceCount >= transmissionTimeout {
            transmissionActive = false
            transmissionDetected = false
        }
    } else {
        silenceCount = 0
    }
}
```

## Bandwidth Calculation Methods

### Required Bandwidth

```go
// CalculateRequiredBandwidth calculates the minimum bandwidth required for transmission
func (c *Config) CalculateRequiredBandwidth() float32 {
    chipRate := c.SymbolRate * float32(c.SpreadingFactor)
    
    switch c.PulseShape {
    case "rectangular":
        // Null bandwidth: B = R_c / 2 (minimum)
        return chipRate / 2.0
    case "raised_cosine", "root_raised_cosine":
        // Null bandwidth: B = (1 + α) × R_c / 2
        return (1.0 + c.Rolloff) * chipRate / 2.0
    default:
        return chipRate / 2.0 // Conservative estimate
    }
}

// CalculateMinimumSampleRate calculates minimum sample rate for given carrier
func (c *Config) CalculateMinimumSampleRate() float32 {
    bandwidth := c.CalculateRequiredBandwidth()
    // Nyquist: f_s ≥ 2(f_c + B/2)
    // For safety: f_s ≥ 4f_c (if carrier >> bandwidth)
    // Or: f_s ≥ 2(f_c + B/2)
    return max(4.0*c.CarrierFreq, 2.0*(c.CarrierFreq + bandwidth/2.0))
}

// CalculateSamplesPerChip calculates samples per chip for given sample rate
func (c *Config) CalculateSamplesPerChip(sampleRate float32) int {
    chipRate := c.SymbolRate * float32(c.SpreadingFactor)
    return int(sampleRate / chipRate)
}

// ValidateSampleRate validates that sample rate meets requirements
func (c *Config) ValidateSampleRate(sampleRate float32) error {
    minRate := c.CalculateMinimumSampleRate()
    if sampleRate < minRate {
        return fmt.Errorf("sample rate %f Hz is below minimum %f Hz", sampleRate, minRate)
    }
    return nil
}
```

### SNR and Processing Gain Calculations

```go
// CalculateProcessingGain calculates the processing gain in linear and dB scale
func (c *Config) CalculateProcessingGain() (linear float32, dB float32) {
    linear = float32(c.SpreadingFactor)
    dB = 10.0 * math32.Log10(linear)
    return linear, dB
}

// CalculateOutputSNR calculates output SNR after despreading from input SNR
func (c *Config) CalculateOutputSNR(inputSNRdB float32) float32 {
    // SNR_out[dB] = SNR_in[dB] + 10*log10(L)
    _, processingGainDB := c.CalculateProcessingGain()
    return inputSNRdB + processingGainDB
}

// CalculateInputSNR calculates required input SNR to achieve target output SNR
func (c *Config) CalculateInputSNR(targetOutputSNRdB float32) float32 {
    // SNR_in[dB] = SNR_out[dB] - 10*log10(L)
    _, processingGainDB := c.CalculateProcessingGain()
    return targetOutputSNRdB - processingGainDB
}

// CalculateMinimumSNR calculates minimum input SNR required for reliable demodulation
// Returns minimum SNR in dB for given target BER
func (c *Config) CalculateMinimumSNR(targetBER float32) float32 {
    // For BPSK: P_b = Q(√(2*E_b/N_0))
    // E_b/N_0 = (SNR_out * T_s) / k
    // SNR_out[dB] = E_b/N_0[dB] + 10*log10(k) - 10*log10(T_s)
    // For reliable demodulation, typically need E_b/N_0 > 10dB (BER < 10^-5)
    
    // Approximate: E_b/N_0 ≈ 10dB for BER ≈ 10^-5
    // For higher order modulations, need more margin
    bitsPerSymbol := float32(c.BitsPerSymbol)
    
    // Base E_b/N_0 requirement (in dB)
    // Binary (k=1): ~10dB for BER 10^-5
    // Higher order: add margin based on modulation order
    baseEbN0 := 10.0 + 3.0*(bitsPerSymbol-1.0) // Add 3dB per extra bit
    
    // Convert to output SNR
    // SNR_out = E_b/N_0 + 10*log10(k) - 10*log10(T_s)
    // For simplicity, assume T_s = 1/R_s, and R_s is normalized
    // SNR_out ≈ E_b/N_0 + 10*log10(k)
    outputSNR := baseEbN0 + 10.0*math32.Log10(bitsPerSymbol)
    
    // Convert to input SNR
    return c.CalculateInputSNR(outputSNR)
}

// CalculateMaximumAchievableSNR calculates maximum achievable output SNR
// given input SNR and processing gain
func (c *Config) CalculateMaximumAchievableSNR(inputSNRdB float32) float32 {
    return c.CalculateOutputSNR(inputSNRdB)
}

// CalculateEbN0 calculates bit energy to noise ratio from output SNR
func (c *Config) CalculateEbN0(outputSNRdB float32, symbolRate float32) float32 {
    // E_b/N_0 = (SNR_out * T_s) / k
    // E_b/N_0[dB] = SNR_out[dB] + 10*log10(T_s) - 10*log10(k)
    // T_s = 1/R_s
    bitsPerSymbol := float32(c.BitsPerSymbol)
    symbolPeriod := 1.0 / symbolRate
    
    ebN0Linear := math32.Pow(10.0, outputSNRdB/10.0) * symbolPeriod / bitsPerSymbol
    return 10.0 * math32.Log10(ebN0Linear)
}

// EstimateBER estimates bit error rate from output SNR
// Returns approximate BER for BPSK-like modulation
func (c *Config) EstimateBER(outputSNRdB float32, symbolRate float32) float32 {
    // For BPSK: P_b = Q(√(2*E_b/N_0))
    // Q(x) ≈ 0.5 * erfc(x/√2)
    // For higher order modulations, BER is more complex
    
    ebN0dB := c.CalculateEbN0(outputSNRdB, symbolRate)
    ebN0Linear := math32.Pow(10.0, ebN0dB/10.0)
    
    // Approximate Q function for BPSK
    // Q(√(2*E_b/N_0)) ≈ 0.5 * erfc(√(E_b/N_0))
    sqrtEbN0 := math32.Sqrt(2.0 * ebN0Linear)
    
    // Use erfc approximation: erfc(x) ≈ exp(-x²) / (x*√π) for large x
    // For small x, use series expansion
    if sqrtEbN0 > 3.0 {
        // Large argument approximation
        return 0.5 * math32.Exp(-sqrtEbN0*sqrtEbN0/2.0) / (sqrtEbN0 * 1.772) // 1.772 ≈ √π
    } else {
        // Small argument: use Q function approximation
        // Q(x) ≈ 0.5 * (1 - erf(x/√2))
        // For simplicity, use exponential approximation
        return 0.5 * math32.Exp(-ebN0Linear)
    }
}

// CalculateSNRLimit calculates the SNR limit for reliable communication
// Returns the minimum input SNR required to achieve target BER
func (c *Config) CalculateSNRLimit(targetBER float32) float32 {
    return c.CalculateMinimumSNR(targetBER)
}

// IsSNRSufficient checks if given input SNR is sufficient for reliable demodulation
func (c *Config) IsSNRSufficient(inputSNRdB float32, targetBER float32) bool {
    minSNR := c.CalculateSNRLimit(targetBER)
    return inputSNRdB >= minSNR
}
```

### Symbol Separation and Code Correlation Calculations

```go
// CalculateCodeCorrelation calculates normalized correlation between two codes
// Returns correlation value in range [-1, 1]
// For orthogonal codes: correlation ≈ 0 for different codes, ≈ 1 for same code
func (c *Config) CalculateCodeCorrelation(codeI, codeJ []int8) float32 {
    if len(codeI) != len(codeJ) || len(codeI) != c.SpreadingFactor {
        return 0.0 // Invalid codes
    }
    
    // Normalized correlation: ⟨Pᵢ, Pⱼ⟩ = (1/L)Σ(k=0 to L-1) pₖ⁽ⁱ⁾ × pₖ⁽ʲ⁾
    var sum float32
    L := float32(c.SpreadingFactor)
    
    for k := range c.SpreadingFactor {
        sum += float32(codeI[k] * codeJ[k])
    }
    
    return sum / L
}

// CalculateMaximumCrossCorrelation calculates maximum cross-correlation between any two different codes
// Lower values indicate better code separation
func (c *Config) CalculateMaximumCrossCorrelation() float32 {
    if len(c.Codes) < 2 {
        return 1.0 // Single code or no codes
    }
    
    maxCorr := float32(0.0)
    
    // Check all pairs of different codes
    for i := range len(c.Codes) {
        for j := i + 1; j < len(c.Codes); j++ {
            corr := math32.Abs(c.CalculateCodeCorrelation(c.Codes[i], c.Codes[j]))
            if corr > maxCorr {
                maxCorr = corr
            }
        }
    }
    
    return maxCorr
}

// CalculateMinimumCodeSeparation calculates minimum separation between codes
// Returns the minimum distance between any two different codes
// Higher values indicate better symbol separation
func (c *Config) CalculateMinimumCodeSeparation() float32 {
    if len(c.Codes) < 2 {
        return 2.0 // Binary case: separation is 2 (between -1 and +1)
    }
    
    minSep := float32(2.0) // Maximum possible separation
    
    // For each pair of codes, calculate separation
    for i := range len(c.Codes) {
        for j := i + 1; j < len(c.Codes); j++ {
            // Separation = 1 - |correlation| (normalized)
            // For orthogonal codes: correlation ≈ 0, separation ≈ 1
            corr := math32.Abs(c.CalculateCodeCorrelation(c.Codes[i], c.Codes[j]))
            sep := 1.0 - corr
            if sep < minSep {
                minSep = sep
            }
        }
    }
    
    return minSep
}

// CalculateCodeOrthogonality calculates orthogonality metric for code set
// Returns value in range [0, 1] where 1 = perfectly orthogonal, 0 = identical codes
func (c *Config) CalculateCodeOrthogonality() float32 {
    if len(c.Codes) < 2 {
        return 1.0 // Single code is trivially orthogonal
    }
    
    // Orthogonality = 1 - max_cross_correlation
    maxCrossCorr := c.CalculateMaximumCrossCorrelation()
    return 1.0 - maxCrossCorr
}

// CalculateSymbolSeparationDistance calculates Euclidean distance between symbols in signal space
// For DSSS: separation depends on code correlation and modulation scheme
func (c *Config) CalculateSymbolSeparationDistance(symbolI, symbolJ int) float32 {
    if symbolI == symbolJ {
        return 0.0 // Same symbol
    }
    
    if symbolI < 0 || symbolI >= len(c.Codes) || symbolJ < 0 || symbolJ >= len(c.Codes) {
        return 0.0 // Invalid symbols
    }
    
    // For binary (M=2): separation is 2 (between -code and +code)
    if c.ModulationOrder == 2 {
        return 2.0 * c.Amplitude
    }
    
    // For M-ary: separation = √(2L × (1 - correlation))
    // This is the Euclidean distance in L-dimensional signal space
    corr := c.CalculateCodeCorrelation(c.Codes[symbolI], c.Codes[symbolJ])
    L := float32(c.SpreadingFactor)
    separation := math32.Sqrt(2.0 * L * (1.0 - corr)) * c.Amplitude
    
    return separation
}

// CalculateMinimumSymbolSeparation calculates minimum separation between any two different symbols
// This is the worst-case separation and determines demodulation reliability
func (c *Config) CalculateMinimumSymbolSeparation() float32 {
    if c.ModulationOrder == 2 {
        // Binary: separation is always 2 (between -code and +code)
        return 2.0 * c.Amplitude
    }
    
    if len(c.Codes) < 2 {
        return 0.0 // Need at least 2 codes
    }
    
    minSep := float32(math32.MaxFloat32)
    
    // Find minimum separation between any two different symbols
    for i := range len(c.Codes) {
        for j := i + 1; j < len(c.Codes); j++ {
            sep := c.CalculateSymbolSeparationDistance(i, j)
            if sep < minSep {
                minSep = sep
            }
        }
    }
    
    return minSep
}

// CalculateMinimumDetectableSeparation calculates minimum separation required for reliable detection
// Based on noise floor and SNR requirements
func (c *Config) CalculateMinimumDetectableSeparation(inputSNRdB float32) float32 {
    // Minimum detectable separation ≈ 2 × noise_amplitude × margin
    // noise_amplitude = signal_amplitude / √(SNR_linear)
    // margin accounts for correlation threshold (typically 2-3 sigma)
    
    inputSNRLinear := math32.Pow(10.0, inputSNRdB/10.0)
    noiseAmplitude := c.Amplitude / math32.Sqrt(inputSNRLinear)
    
    // Detection margin: typically 2-3 standard deviations for reliable detection
    // Use 2.5 as compromise between sensitivity and false alarm rate
    detectionMargin := float32(2.5)
    
    minDetectable := 2.0 * noiseAmplitude * detectionMargin
    
    return minDetectable
}

// IsSymbolSeparationSufficient checks if symbol separation is sufficient for reliable demodulation
func (c *Config) IsSymbolSeparationSufficient(inputSNRdB float32) bool {
    minSep := c.CalculateMinimumSymbolSeparation()
    minDetectable := c.CalculateMinimumDetectableSeparation(inputSNRdB)
    
    // Separation should be at least 2× the minimum detectable for reliable operation
    // This provides margin for noise and interference
    return minSep >= 2.0*minDetectable
}

// CalculateSymbolSeparationMargin calculates margin between actual and required separation
// Returns positive value if sufficient, negative if insufficient
func (c *Config) CalculateSymbolSeparationMargin(inputSNRdB float32) float32 {
    minSep := c.CalculateMinimumSymbolSeparation()
    minDetectable := c.CalculateMinimumDetectableSeparation(inputSNRdB)
    
    // Margin = actual_separation - 2×required_separation
    return minSep - 2.0*minDetectable
}
```

## Test Cases

### Test 1: Basic Roundtrip (1 byte, infinite SNR)
```
1. Create modulator with config (96kHz, 20kHz carrier, k=1 bit/symbol, L=8 codes)
2. Create test byte: 0x42
3. Create Bits buffer with test byte (BitsLength = k)
4. Modulate: write samples to buffer
5. Add 0.5s silence (48000 samples at 96kHz) before and after
6. Create demodulator with same config
7. Split samples into 3 parts (call Process 3 times with advancing buffer)
8. Demodulate: read bytes from Bits buffer
9. Verify: received byte == 0x42
```

### Test 2: Multiple Bytes (4 bytes, infinite SNR)
```
- Same as Test 1 but with 4 bytes: [0x01, 0x23, 0x45, 0x67]
- Verify all 4 bytes match
```

### Test 3: Large Buffer (33 bytes, infinite SNR)
```
- Same as Test 1 but with 33 bytes
- Verify all 33 bytes match
```

### Test 4: Noise Resilience (7 bytes, 20dB SNR)
```
- Modulate 7 bytes
- Add white Gaussian noise at 20dB SNR
- Demodulate and verify bytes match
```

### Test 5: Noise Resilience (7 bytes, 15dB SNR)
```
- Same as Test 4 but with 15dB SNR
- Test with k=2, k=4, and k=8 bits per symbol
- Codes are 8 bits each (L=8)
```

### Test 6: Noise Resilience (7 bytes, 10dB SNR)
```
- Same as Test 4 but with 10dB SNR
- Test if theoretically possible given modulation scheme
- May have some bit errors (acceptable for low SNR)
```

### Test 7: Noise Resilience (7 bytes, 5dB SNR)
```
- Same as Test 4 but with 5dB SNR
- Test if theoretically possible given modulation scheme
- May have significant bit errors (acceptable for very low SNR)
```

## Implementation Steps

### Phase 1: Core Modulator
1. ✅ Create `Modulator` struct with config
2. ✅ Implement `Process` method:
   - Validate `src` implements `BitsReader`
   - Validate `BitsLength` compatibility
   - Read bytes one at a time (symbol-by-symbol)
   - Extract k bits per symbol
   - Map to chip sequences
   - Apply pulse shaping (raised cosine)
   - Modulate carrier
   - Write float32 samples
   - Handle partial reads/writes (streaming)
3. ✅ Implement `Reset` method
4. ✅ Handle nil `dst` for size calculation
5. ✅ Maintain carrier phase continuity
6. ✅ Implement raised cosine pulse shaping

### Phase 2: Core Demodulator
1. ✅ Create `Demodulator` struct with config
2. ✅ Implement `Process` method:
   - Read samples as needed (small buffer)
   - Detect transmission
   - Downconvert carrier
   - Correlate with codes
   - Detect symbols
   - Extract bits
   - Pack into bytes
   - Write to `BitsWriter`
   - Handle transmission start/end
   - Handle partial reads/writes (streaming)
3. ✅ Implement `Reset` method
4. ✅ Handle nil `dst` for size calculation
5. ✅ Maintain symbol timing and phase tracking
6. ✅ Implement transmission detection logic

### Phase 3: Preamble Support
1. ✅ Add preamble generation to modulator
2. ✅ Add preamble detection to demodulator
3. ✅ Ignore preamble in demodulator output (don't write preamble bytes)

### Phase 4: Bandwidth, SNR, and Symbol Separation Calculations
1. ✅ Implement `CalculateRequiredBandwidth()` method
2. ✅ Implement `CalculateMinimumSampleRate()` method
3. ✅ Implement `CalculateSamplesPerChip()` method
4. ✅ Implement `ValidateSampleRate()` method
5. ✅ Add validation to ensure sample rate meets requirements
6. ✅ Implement `CalculateProcessingGain()` method (linear and dB)
7. ✅ Implement `CalculateOutputSNR()` method
8. ✅ Implement `CalculateInputSNR()` method
9. ✅ Implement `CalculateMinimumSNR()` method
10. ✅ Implement `CalculateMaximumAchievableSNR()` method
11. ✅ Implement `CalculateEbN0()` method
12. ✅ Implement `EstimateBER()` method
13. ✅ Implement `CalculateSNRLimit()` method
14. ✅ Implement `IsSNRSufficient()` method
15. ✅ Implement `CalculateCodeCorrelation()` method
16. ✅ Implement `CalculateMaximumCrossCorrelation()` method
17. ✅ Implement `CalculateMinimumCodeSeparation()` method
18. ✅ Implement `CalculateCodeOrthogonality()` method
19. ✅ Implement `CalculateSymbolSeparationDistance()` method
20. ✅ Implement `CalculateMinimumSymbolSeparation()` method
21. ✅ Implement `CalculateMinimumDetectableSeparation()` method
22. ✅ Implement `IsSymbolSeparationSufficient()` method
23. ✅ Implement `CalculateSymbolSeparationMargin()` method

### Phase 5: Testing
1. ✅ Implement Test 1: Basic roundtrip (1 byte)
2. ✅ Implement Test 2: 4 bytes
3. ✅ Implement Test 3: 33 bytes
4. ✅ Implement noise generation utility
5. ✅ Implement Test 4: 20dB SNR
6. ✅ Implement Test 5: 15dB SNR (k=2, 4, 8 bits/symbol)
7. ✅ Implement Test 6: 10dB SNR (if possible)
8. ✅ Implement Test 7: 5dB SNR (if possible)

### Phase 6: Optimization
1. ✅ Optimize hot paths (no type casts in loops)
2. ✅ Use `math32` for float32 operations
3. ✅ Minimize allocations (pre-allocate buffers)
4. ✅ Use `for range n` instead of `for i := 0; i < n; i++`
5. ✅ Boundary check elimination where possible
6. ✅ Pre-compute pulse shape samples
7. ✅ Use `fp32` package utilities where applicable

## File Structure

```
x/math/dsp/modulation/dsss/
├── SPEC.md                    (existing)
├── IMPLEMENTATION_PLAN.md     (this file)
├── modulator.go               (Modulator implementation)
├── demodulator.go             (Demodulator implementation)
├── config.go                  (Configuration structures)
├── pulse.go                   (Pulse shaping functions)
├── bandwidth.go               (Bandwidth calculation methods)
├── modulator_test.go          (Modulator unit tests)
├── demodulator_test.go        (Demodulator unit tests)
├── roundtrip_test.go          (End-to-end tests)
└── noise_test.go              (Noise resilience tests)
```

## Key Design Decisions

1. **Symbol-by-Symbol Processing**: Reading one symbol at a time simplifies state management and aligns with `BitsReader` interface expectations.

2. **Stateful Streaming**: Both modulator and demodulator maintain state across `Process` calls to support streaming scenarios with partial reads/writes.

3. **Preamble Handling**: Preamble is generated by modulator but ignored by demodulator (not written to output buffer).

4. **Nil Destination Handling**: When `dst` is nil, calculate theoretical maximum output size for buffer sizing.

5. **Validation**: Modulator validates `BitsReader` interface and `BitsLength` compatibility at start of `Process`.

6. **Carrier Phase Continuity**: Phase is maintained globally using sample counter to ensure continuous carrier across multiple `Process` calls.

7. **Symbol Timing**: Demodulator tracks symbol boundaries using sample offset and correlation peaks.

8. **Transmission Detection**: Demodulator only writes bytes when transmission is detected and active. Stops when transmission ends.

9. **Small Buffer**: Demodulator maintains small correlation buffer (2-3 symbol periods) for efficient processing.

10. **Spectrally Optimal Pulse Shaping**: Use raised cosine (or root raised cosine) for bandwidth efficiency instead of rectangular pulses.

11. **Bits Per Symbol**: Support k=1, 2, 4, 8 bits per symbol (M=2, 4, 16, 256)

12. **8-bit Codes**: All tests use codes of length L=8

## Dependencies

- `github.com/itohio/EasyRobot/x/buffer` - BitsReader, BitsWriter, Writer, Reader interfaces
- `github.com/itohio/EasyRobot/x/chain` - Processor interface
- `github.com/chewxy/math32` - Float32 math operations
- `github.com/itohio/EasyRobot/x/types` - Number types
- `github.com/itohio/EasyRobot/x/math/primitive/fp32` - Float32 primitives
- `github.com/itohio/EasyRobot/x/math/dsp` - CosineGenerator, SineGenerator, SquareGenerator, TriangleGenerator, and noise generators (all support `GenerateWithOp` for efficient buffer-free operations)
- `github.com/itohio/EasyRobot/x/math/filter/fir` - sinc function (or implement inline)

**Note on Generators**: The `dsp` package has `CosineGenerator`, `SineGenerator`, `SquareGenerator`, `TriangleGenerator`, and noise generators that all support `GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp)` where `GeneratorAssignOp` is `func(d *float32, value float32)`. This allows direct application of generated values to existing samples without requiring an intermediate buffer allocation. 

For DSSS carrier modulation, we can use `CosineGenerator.GenerateWithOp()` to multiply baseband samples directly:
```go
carrierGen.GenerateWithOp(basebandSamples, false, func(d *float32, carrierValue float32) {
    *d *= carrierValue
})
```

This is more efficient than using `Generate()` with a small stack array because it avoids the intermediate buffer allocation and copy operation.

## Test Parameters

### Standard Test Configuration
- **Sample Rate**: 96,000 Hz
- **Carrier Frequency**: 20,000 Hz
- **Symbol Rate**: Configurable (e.g., 1,000 symbols/sec)
- **Spreading Factor**: 8 (8-bit codes, L=8)
- **Bits Per Symbol**: 1, 2, 4, 8 (k = 1, 2, 4, 8)
- **Modulation Orders**: 2 (k=1), 4 (k=2), 16 (k=4), 256 (k=8)
- **Pulse Shape**: Raised cosine with rolloff 0.35-0.5
- **Amplitude**: 1.0
- **Preamble**: Alternating symbols (0, 1, 0, 1, ...) for sync

### Noise Test Parameters
- **SNR Levels**: 20dB, 15dB, 10dB (if possible), 5dB (if possible)
- **Noise Type**: White Gaussian noise (AWGN)
- **Test Data**: 7 bytes
- **Bits Per Symbol**: 2, 4, 8 (for 15dB SNR test)

## Notes

- Initial implementation uses raised cosine pulse shaping (spectrally optimal)
- Carrier modulation uses coherent BPSK/DSSS (cosine modulation)
- Symbol detection uses maximum likelihood (argmax of correlations)
- Preamble uses alternating symbols for easy detection
- Noise tests use white Gaussian noise added to signal
- SNR calculation: `SNR = 10×log₁₀(signal_power / noise_power)`
- Bandwidth calculation methods help users determine required sample rate and carrier frequency
- Demodulator buffer kept small (2-3 symbols) for efficient processing
- Transmission detection prevents writing noise as data
- Processing gain: G_p = L (8 for 8-bit codes = 9dB gain)
- Lower SNR tests (10dB, 5dB) may not be feasible depending on processing gain and noise characteristics
- Use `CalculateSNRLimit()` and `IsSNRSufficient()` to determine if test SNR levels are theoretically achievable
- Use `CalculateOutputSNR()` to verify processing gain improvement (e.g., 5dB input → 14dB output with L=8)
- Use `EstimateBER()` to predict bit error rates for given SNR conditions
- Use `CalculateCodeOrthogonality()` to verify code set quality (should be > 0.9 for good separation)
- Use `CalculateMinimumSymbolSeparation()` and `IsSymbolSeparationSufficient()` to verify codes are distinguishable at given SNR
- Use `CalculateSymbolSeparationMargin()` to determine safety margin for reliable demodulation
- Symbol separation depends on modulation scheme: binary uses sign separation (2×amplitude), M-ary uses code cross-correlation

