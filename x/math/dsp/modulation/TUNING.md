# Modulation Tuning Guide

This document summarizes critical lessons learned during the implementation and tuning of MPSK and QAM modulators/demodulators. It covers buffer management, streaming simulation, PLL tuning, signal detection, and testing strategies.

## Table of Contents

1. [Buffer Management in Tests](#buffer-management-in-tests)
2. [Streaming Simulation](#streaming-simulation)
3. [SNR-Based Thresholds](#snr-based-thresholds)
4. [PLL Tuning](#pll-tuning)
5. [Preamble Design](#preamble-design)
6. [AGC for Coherent Signals](#agc-for-coherent-signals)
7. [Phase Error Handling](#phase-error-handling)
8. [Signal Detection](#signal-detection)
9. [Signal Tracking](#signal-tracking)
10. [Reset Behavior](#reset-behavior)
11. [SOLID Principles](#solid-principles)
12. [Testing Strategies](#testing-strategies)

---

## Buffer Management in Tests

### Zero-Copy Buffer Strategy

**Critical**: Use a single allocated buffer and operate on slice views to avoid unnecessary copying and memory allocations.

```go
// Calculate total samples needed: padding + signal + padding
paddingSamples := int(0.1 * cfg.SampleRate) * 2 // IQ interleaved
totalSamples := paddingSamples*2 + signalLength
sampleBuffer := make([]float32, totalSamples)

// Modulate into a slice view (zero-copy)
signalStart := paddingSamples
signalEnd := paddingSamples + signalLength
signalBuf := buffer.New(sampleBuffer[signalStart:signalStart:signalEnd])
written := modulator.Process(signalBuf, inputBits)

// Add noise to entire buffer (including padding for realism)
addNoiseToBuffer(sampleBuffer, sampleBuffer[signalStart:signalEnd], snrDb)

// Demodulate from slice views (simulates streaming)
chunks := splitIntoChunksFromBuffer(sampleBuffer, 10)
for _, chunk := range chunks {
    reader := buffer.New(chunk) // Slice view, no copy
    demodulator.Process(outputBits, reader)
}
```

**Key Points**:
- Allocate one large buffer for the entire test scenario
- Use slice views (`buffer[start:start:end]`) to create zero-copy views
- Noise should be applied to the entire buffer (including padding) for realistic testing
- Padding should contain noise, not pure silence, to test signal detection robustness

### Buffer Structure

```
[Padding (noise)] [Signal] [Padding (noise)]
     ↑                ↑            ↑
  noise only    signal+noise   noise only
```

This structure tests:
- Signal detection in noise
- PLL acquisition from noise
- Silence detection between transmissions
- PLL reset and re-acquisition

---

## Streaming Simulation

### Chunked Processing

Simulate streaming by processing the buffer in multiple chunks:

```go
func splitIntoChunksFromBuffer(samples []float32, parts int) [][]float32 {
    if parts <= 1 {
        return [][]float32{samples}
    }
    chunks := make([][]float32, 0, parts)
    chunkSize := len(samples) / parts
    for i := range parts {
        start := i * chunkSize
        end := start + chunkSize
        if i == parts-1 {
            end = len(samples)
        }
        // Use slice view (no copying) - all chunks point to same underlying buffer
        chunks = append(chunks, samples[start:end])
    }
    return chunks
}
```

**Why This Matters**:
- Tests real-world streaming behavior
- Validates buffer management in demodulator
- Ensures PLL can maintain lock across buffer boundaries
- Catches buffer advancement bugs (e.g., incorrect `buffered` updates)

### Buffer Advancement Bug

**Common Bug**: Incorrect buffer advancement in demodulator

```go
// WRONG - loses samples
d.buffered -= symbolSamples

// CORRECT - preserves remaining samples
remaining := d.buffered - symbolSamples
if remaining > 0 {
    copy(d.fifo, d.fifo[symbolSamples:d.buffered])
}
d.buffered = remaining
```

Streaming tests catch this by processing in chunks - if buffer advancement is wrong, symbols will be lost or duplicated.

---

## SNR-Based Thresholds

### Energy Threshold Calculation

**Critical**: Thresholds must be calculated based on expected SNR, not fixed values.

```go
func (c Config) EnergyThresholdForSNR(snrDb float32) float32 {
    // At SNR dB, signal power is 10^(SNR/10) times noise power
    // Signal amplitude is sqrt(power), so signal amplitude is 10^(SNR/20) times noise amplitude
    signalAmp := c.SignalAmplitude
    noiseAmp := signalAmp / math32.Pow(10, snrDb/20)
    
    // Threshold should be between noise-only power and signal+noise power
    // Use geometric mean: sqrt(noisePower * signalPower)
    noisePower := noiseAmp * noiseAmp
    signalPower := signalAmp * signalAmp
    thresholdPower := math32.Sqrt(noisePower * signalPower)
    
    return thresholdPower
}
```

**Why This Matters**:
- Fixed thresholds fail at different SNR levels
- Too low: false detections from noise-only padding
- Too high: miss weak signals at low SNR
- SNR-based thresholds adapt to test conditions

### Adaptive Thresholds

Use different thresholds for different states:

```go
baseThreshold := cfg.EnergyThresholdForSNR(targetSNR)

var threshold float32
if active {
    // During active transmission: be lenient (noise might reduce energy)
    threshold = baseThreshold * 0.3
} else if preambleCount > 0 {
    // During preamble: moderate threshold
    threshold = baseThreshold * 0.5
} else {
    // Initial detection: use full SNR-based threshold
    threshold = baseThreshold
}
```

---

## PLL Tuning

### AGC Before PLL

**Critical Change**: AGC must normalize signals **before** PLL processing.

**Why**:
- PLL sees normalized signal amplitude, making phase tracking more stable
- Reduces amplitude-dependent phase errors
- Allows PLL to lock faster on consistent signal levels

**Implementation**:
```go
// 1. Integrate symbol
iAcc, qAcc := integrateSymbol(samples)

// 2. Apply AGC FIRST (normalize amplitude)
mag := math32.Hypot(iAcc, qAcc)
if mag > 0 {
    agc.Process(mag) // Update AGC
    iNorm, qNorm := applyAGC(iAcc, qAcc) // Normalize
    iAcc, qAcc = iNorm, qNorm
}

// 3. Apply PLL to normalized samples
iRot, qRot := pll.Rotate(iAcc, qAcc)
```

### PLL Bandwidth Tuning

**With AGC normalization**, PLL parameters need adjustment:

| SNR Range | Base Bandwidth (old) | Base Bandwidth (new) | Reason |
|-----------|---------------------|---------------------|--------|
| 25dB+     | 2.0%                | 2.5%                | AGC reduces amplitude variations |
| 20-25dB   | 5.0%                | 6.0%                | More stable with normalized signals |
| 15-20dB   | 2.0%                | 2.5%                | Consistent amplitude helps tracking |
| 10-15dB   | 1.0%                | 1.5%                | Wider bandwidth possible with AGC |

**Acquisition vs Tracking**:
- Acquisition: 1.8x base bandwidth (was 1.5x)
- Tracking: 0.6x base bandwidth
- Max acquisition: 12% of symbol rate (was 10%)

### PLL Damping

Damping should be tuned based on SNR:

| SNR Range | Damping | Reason |
|-----------|---------|--------|
| 25dB+     | 1.00    | Optimal for high SNR |
| 20-25dB   | 1.00    | Stable tracking |
| 15-20dB   | 0.707   | Standard critical damping |
| 10-15dB   | 1.50    | Higher damping for noise rejection |
| 8-10dB    | 0.50    | Lower damping for faster response |

### PLL Gotchas

1. **Phase Error Calculation**: Always use `wrapPi()` to handle phase wrap-around
   ```go
   phaseError := wrapPi(phase - expectedPhase)
   ```

2. **Adaptive Updates**: Use full update for small errors, reduced for large errors
   ```go
   maxPhaseError := twoPi / constellationSize * 0.6
   if math32.Abs(phaseError) < maxPhaseError {
       pll.Update(phaseError) // Full update
   } else {
       pll.Update(phaseError * 0.3) // Reduced update for noise spikes
   }
   ```

3. **Lock Detection**: Don't force lock - validate phase error is reasonable
   ```go
   if avgPhaseError < maxAllowedError {
       pllLocked = true
       setTrackingBandwidth()
   }
   ```

4. **Bandwidth Switching**: Switch from acquisition to tracking bandwidth after lock
   - Acquisition: wider bandwidth for fast convergence
   - Tracking: narrower bandwidth for noise filtering

---

## Preamble Design

### Purpose

Preambles serve multiple purposes:
1. **Phase Ambiguity Resolution**: Cover all constellation phases
2. **PLL Acquisition**: Provide known symbols for fast PLL lock
3. **AGC Training**: Allow AGC to converge before payload

### Optimal Preamble Pattern

**For QPSK (M=4)**:
```
First M symbols: [0, 1, 2, 3] - covers all phases
Next symbols: [0, 2, 1, 3, 0, 1, 2, 3, ...] - maximizes phase transitions
```

**Pattern Design Principles**:
- First M symbols: cover all phases (resolves ambiguity)
- Subsequent symbols: maximize phase transitions (helps PLL track)
- Repeat pattern every 2*M symbols for consistency

### Preamble Length

**Default**: 20 symbols
- Allows PLL to converge (typically locks after 12-16 symbols)
- Provides margin for noise-induced errors
- Long enough for AGC to stabilize

**Configurable**: Should depend on target SNR
- Higher SNR: shorter preamble possible
- Lower SNR: longer preamble needed for reliable lock

### Preamble Processing

**Key Points**:
- Use expected phase for PLL error calculation
- Be lenient with symbol matching (allow adjacent symbols)
- Track phase errors for lock validation
- Don't emit preamble bits to output

---

## AGC for Coherent Signals

### Why AGC is Critical

**Coherent demodulation** (MPSK, QAM) requires:
1. **Consistent amplitude**: Constellation points must be at expected levels
2. **Normalized signals**: PLL works better on normalized signals
3. **Noise rejection**: AGC should not amplify noise-only samples

### AGC Training Strategy

**Before PLL Lock**:
- Update AGC gain based on signal magnitude
- **Do not normalize** - might be noise
- Train AGC to converge to target level

**After PLL Lock**:
- AGC has converged
- **Apply normalization** - we have real signal
- Normalize all samples before PLL processing

### AGC Implementation

```go
// Update AGC with magnitude
mag := math32.Hypot(i, q)
if mag > 0 {
    agc.Process(mag) // Updates internal gain
}

// Apply normalization
scaled := agc.Process(mag) // Returns normalized magnitude
gain := scaled / (mag + 1e-6)
return i * gain, q * gain
```

**Critical**: AGC must be applied **before** PLL, not after. This ensures PLL sees normalized signals.

---

## Phase Error Handling

### Phase Wrap-Around

**Always use `wrapPi()` for phase differences**:

```go
func wrapPi(angle float32) float32 {
    for angle > math32.Pi {
        angle -= twoPi
    }
    for angle < -math32.Pi {
        angle += twoPi
    }
    return angle
}
```

**Why**: Phase errors can be > π, causing incorrect PLL updates.

### Phase Error Validation

**Validate phase errors are reasonable** before using for PLL updates:

```go
maxPhaseError := twoPi / constellationSize * 0.7
if math32.Abs(phaseError) < maxPhaseError {
    // Normal update - likely real phase error
    pll.Update(phaseError)
} else {
    // Large error - likely noise spike
    pll.Update(phaseError * 0.3) // Reduced update
}
```

### Phase Ambiguity

**MPSK has M-fold phase ambiguity**:
- QPSK: 4 possible phase offsets (0°, 90°, 180°, 270°)
- 8PSK: 8 possible phase offsets

**Solutions**:
1. **Preamble**: Use known pattern to resolve ambiguity
2. **Differential encoding**: Encode data in phase differences (DPSK)

---

## Signal Detection

### Coarse Detection

**Quick energy check** before expensive processing:

```go
// Coarse detection: very lenient threshold
coarseThreshold := cfg.EnergyThresholdForSNR(targetSNR) * 0.1

// Coarse detection logic:
// - When no lock on a signal - coarse detection avoids calculations
// - When coarse detection triggers and lock is achieved - disable coarse detection until lock is lost
// - Coarse detection should NOT be active during active lock
useCoarseDetection := !active && !pllLocked
if useCoarseDetection && energy < coarseThreshold {
    return -1 // Skip expensive processing
}
```

**Key Points**:
- **When no lock**: Coarse detection avoids expensive calculations on noise-only samples
- **When locked**: Coarse detection is disabled to ensure all symbols are processed
- **State-based**: Only active when `!active && !pllLocked` to prevent skipping symbols during active transmission

**Why**: Avoids wasting CPU on noise-only samples, but ensures all symbols are processed once lock is achieved.

### Full Detection

**SNR-based threshold** for reliable detection:

```go
baseThreshold := cfg.EnergyThresholdForSNR(targetSNR)
// Use adaptive threshold based on state
threshold := calculateAdaptiveThreshold(baseThreshold, active, preambleCount)
if energy < threshold {
    // Handle silence or noise
    return -1
}
```

### Silence Detection

**Track silence count** to detect transmission gaps:

```go
if energy < threshold && !active {
    silenceCount++
    if silenceCount >= cfg.TransmissionGap {
        // Reset for new transmission
        resetLoops()
        unlockPLL()
    }
    return -1
}
```

**Key**: Only reset during silence, not during active transmission (low energy might be noise, not silence).

---

## Signal Tracking

### Active State Management

**Track active transmission state**:

```go
if !active && energy >= threshold {
    active = true // Signal detected
}
if active {
    silenceCount = 0 // Reset silence counter
}
```

**Why**: Prevents false resets during transmission (low energy symbols might be due to noise).

### PLL Lock State

**Track PLL lock separately from active state**:

```go
if !pllLocked {
    // Check if we can lock
    if phaseErrorCount >= 8 {
        avgPhaseError := phaseErrorSum / float32(phaseErrorCount)
        if avgPhaseError < maxAllowedError {
            pllLocked = true
            setTrackingBandwidth()
        }
    }
}
```

**Key**: Don't force lock - validate phase error is reasonable first.

---

## Reset Behavior

### Modulator Reset

**Reset between packets** to ensure clean state:

```go
func (m *Modulator) Reset() {
    m.buffered = 0
    m.preambleCount = 0
    // Reset any stateful components
}
```

**Test Pattern**:
```go
mod := NewModulator(cfg)
// Transmit packet 1
mod.Process(buffer1, bits1)
mod.Reset() // Reset between packets
// Transmit packet 2
mod.Process(buffer2, bits2)
```

### Demodulator Reset

**Reset on silence detection**:

```go
func (d *Demodulator) Reset() {
    d.buffered = 0
    d.active = false
    d.preambleCount = 0
    d.unlockPLL()
    d.resetLoops() // Reset PLL and AGC
    // Reset SNR tracking
    d.signalPowerSum = 0
    d.noisePowerSum = 0
    d.snrSampleCount = 0
}
```

### Reset Testing

**Test PLL reset between packets**:

```go
// Buffer: padding + "FOO" + padding + "BAR" + padding
// Use separate demodulators for each packet
demod1 := NewDemodulator(cfg)
demod2 := NewDemodulator(cfg)

// Each demodulator should independently acquire lock
chunks1 := splitIntoChunksFromBuffer(buffer1, 5)
for _, chunk := range chunks1 {
    demod1.Process(outputBits1, buffer.New(chunk))
}

chunks2 := splitIntoChunksFromBuffer(buffer2, 5)
for _, chunk := range chunks2 {
    demod2.Process(outputBits2, buffer.New(chunk))
}
```

**Validates**: Each demodulator can independently acquire PLL lock.

---

## SOLID Principles

### Single Responsibility

**Each component does one thing**:
- `integrateSymbol()`: Only integrates samples
- `quantizePhase()`: Only quantizes phase to symbol
- `applyAGC()`: Only applies AGC normalization
- `decodeSymbol()`: Orchestrates the process

**Why**: Makes testing and debugging easier - can test each component independently.

### Interface Segregation

**Small, focused interfaces**:
```go
type BitsWriter interface {
    Write([]byte) (int, error)
    BitsLength() int
}
```

**Not**: One large interface with many methods.

### Dependency Inversion

**Accept interfaces, return structs**:
```go
func Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int
```

**Not**: Concrete types in function signatures.

### Benefits for Testing

**SOLID principles enable**:
1. **Unit testing**: Test each component in isolation
2. **Mocking**: Easy to mock interfaces
3. **Debugging**: Isolate issues to specific components
4. **Refactoring**: Change implementation without breaking tests

**Example**: When PLL wasn't locking, we could test:
- Symbol integration (correct?)
- Phase calculation (correct?)
- PLL update (correct?)
- AGC normalization (correct?)

Each could be tested independently.

---

## Testing Strategies

### Noise Sweeps

**Test SNR from high to theoretical limit**:

```go
snrValues := []float32{30, 25, 20, 15, 12, 10}
for _, snr := range snrValues {
    t.Run(fmt.Sprintf("%.0fdB", snr), func(t *testing.T) {
        runRoundtrip(t, cfg, payload, snr)
    })
}
```

**Why**:
- Validates performance at various SNR levels
- Identifies minimum working SNR
- Catches regressions at specific SNR levels

**Theoretical Limits**:
- QPSK: ~10dB SNR (coherent detection)
- 16-QAM: ~15dB SNR
- 64-QAM: ~20dB SNR

**Practical Limits**: Usually 2-3dB higher due to:
- PLL acquisition overhead
- AGC convergence time
- Phase ambiguity resolution
- Noise in preamble detection

### Packet Size Sweeps

**Test PLL stability for long packets**:

```go
packetSizes := []int{512, 1024, 2048, 4096}
snrDb := float32(16.0) // Moderate SNR

for _, size := range packetSizes {
    t.Run(fmt.Sprintf("%d_bytes", size), func(t *testing.T) {
        payload := makePayload(size)
        runRoundtrip(t, cfg, payload, snrDb)
    })
}
```

**Why**:
- Validates PLL can maintain lock over extended periods
- Catches phase drift issues
- Tests buffer management for long sequences
- Ensures no memory leaks or buffer overflows

### Modulation Order Sweeps

**Test different constellation sizes**:

```go
bitsPerSymbol := []int{2, 3, 4, 6, 8} // QPSK, 8PSK, 16QAM, 64QAM, 256QAM

for _, bps := range bitsPerSymbol {
    cfg.BitsPerSymbol = bps
    require.NoError(t, cfg.Validate())
    
    t.Run(fmt.Sprintf("%d-QAM", 1<<bps), func(t *testing.T) {
        runRoundtrip(t, cfg, payload, snrDb)
    })
}
```

**Why**:
- Validates constellation table generation
- Tests quantization accuracy
- Ensures PLL works with different symbol rates
- Catches issues with higher-order modulations

### Test Structure

**Recommended test organization**:

```
roundtrip_test.go      - Basic roundtrip tests (1, 7, 33 bytes)
snr_limits_test.go     - SNR sweep tests
long_packet_test.go    - Packet size sweep tests
pll_reset_test.go      - PLL reset/re-acquisition tests
modulation_order_test.go - Constellation size sweep tests
```

**Each test file focuses on one aspect**, making it easy to:
- Run specific test categories
- Debug specific issues
- Understand test coverage

---

## Key Takeaways

1. **Zero-copy buffers**: Use slice views, not copies
2. **SNR-based thresholds**: Adapt to test conditions
3. **AGC before PLL**: Normalize signals first
4. **PLL retuning**: Adjust parameters for AGC-normalized signals
5. **Coarse detection**: Skip expensive processing on noise
6. **SOLID principles**: Enable independent testing and debugging
7. **Comprehensive sweeps**: Test SNR, packet size, modulation order
8. **Reset testing**: Validate PLL re-acquisition
9. **Streaming simulation**: Test real-world behavior
10. **Documentation**: Record tuning decisions and rationale

---

## References

- `x/math/dsp/modulation/mpsk/config.go` - PLL tuning parameters
- `x/math/dsp/modulation/qam/config.go` - QAM-specific tuning
- `x/math/dsp/modulation/mpsk/SPEC.md` - PLL information theory
- `x/math/dsp/modulation/mpsk/roundtrip_test.go` - Test examples
- `x/math/filter/pll` - PLL implementation
- `x/math/filter/agc` - AGC implementation

