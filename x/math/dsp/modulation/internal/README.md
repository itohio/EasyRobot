# Modulation Internal Processors

This package provides in-place processors for complex64 signal processing in QAM/PSK demodulation.

## Implemented Processors

### 1. FIR (Finite Impulse Response Filter)
- **Purpose**: Matched filtering, pulse shaping, anti-aliasing
- **Operation**: Applies FIR filter to both I and Q components independently
- **Usage**: Filters complex64 samples using filter coefficients

### 2. AGC (Automatic Gain Control)
- **Purpose**: Normalizes signal magnitude to target level
- **Operation**: Measures magnitude and adjusts gain automatically
- **Usage**: Stabilizes signal amplitude for consistent demodulation

### 3. Costas Loop
- **Purpose**: Carrier recovery for QAM/PSK signals
- **Operation**: Extracts carrier phase and frequency offset using decision-directed phase detection
- **Usage**: Recovers carrier from suppressed-carrier modulated signals
- **Control Methods**: `Phase()`, `FrequencyOffset()`

### 4. Matched Filter Integrator
- **Purpose**: Integrate samples over symbol period using matched filter envelope
- **Operation**: Multiplies samples by pulse shape and accumulates over symbol duration
- **Usage**: Pre-processes IQ samples before symbol decision
- **Control Methods**: `SymbolCount()`, `SamplesPerSymbol()`

### 5. Phase Rotator
- **Purpose**: Rotate IQ samples by phase offset
- **Operation**: Applies rotation matrix to complex samples
- **Usage**: Phase correction, constellation rotation
- **Control Methods**: `SetPhase()`, `Phase()`, `RotatePhase()`

### 6. Symbol Timing Recovery (Gardner Algorithm)
- **Purpose**: Adjust sampling instants to optimal symbol timing
- **Operation**: Interpolates samples based on Gardner timing error detection
- **Usage**: Handles timing jitter, clock drift, and misalignment
- **Control Methods**: `TimingPhase()`, `TimingError()`, `SetLoopBandwidth()`

### 7. I/Q Imbalance Corrector
- **Purpose**: Correct gain and phase imbalance between I and Q channels
- **Operation**: Applies correction matrix to balance channels
- **Usage**: Corrects hardware imperfections in mixers/ADCs
- **Control Methods**: `SetCorrectionMatrix()`, `CorrectionMatrix()`, `EstimatedImbalance()`

### 8. DC Blocking Filter
- **Purpose**: Remove DC offset from IQ samples
- **Operation**: High-pass IIR filter to remove bias
- **Usage**: Removes DC bias that can affect demodulation
- **Control Methods**: `DCOffset()`, `SetAlpha()`, `Alpha()`

### 9. CFO Corrector (Carrier Frequency Offset)
- **Purpose**: Correct carrier frequency offset
- **Operation**: Mixes with complex exponential to derotate signal
- **Usage**: Coarse frequency offset correction
- **Control Methods**: `SetFrequencyOffset()`, `FrequencyOffset()`, `Phase()`

### 10. AFC (Automatic Frequency Control)
- **Purpose**: Track and correct slow frequency drift
- **Operation**: Estimates frequency from phase derivatives
- **Usage**: Handles long-term frequency drift
- **Control Methods**: `EstimatedOffset()`, `SetEstimatedOffset()`, `SetLoopBandwidth()`

## Additional Operations Needed for QAM/PSK Demodulation

Based on analysis of existing QAM/PSK demodulators, the following additional in-place operations would be useful:

### 4. Matched Filter Integrator
- **Purpose**: Integrate samples over symbol period using matched filter envelope
- **Operation**: Multiply samples by pulse shape (raised cosine, etc.) and accumulate
- **Current**: Done per-symbol in demodulators using `integrateSymbol()`
- **Benefit**: Pre-process IQ samples before symbol decision
- **Priority**: High - used in every symbol decision

### 5. Phase Rotator
- **Purpose**: Rotate IQ samples by a fixed or adaptive phase offset
- **Operation**: Apply rotation matrix: I' = I*cos(θ) - Q*sin(θ), Q' = I*sin(θ) + Q*cos(θ)
- **Current**: Implemented in `pll.Loop.Rotate()` for per-symbol rotation
- **Benefit**: Phase correction, constellation rotation
- **Priority**: Medium - useful for phase alignment before demodulation

### 6. Symbol Timing Recovery (Gardner Algorithm)
- **Purpose**: Adjust sampling instants to optimal symbol timing
- **Operation**: Interpolate/decimate samples based on timing error from Gardner TED
- **Current**: Assumed perfect timing in existing demodulators
- **Benefit**: Handles timing jitter, clock drift, and misalignment
- **Priority**: High - critical for real-world scenarios with timing errors

### 7. I/Q Imbalance Corrector
- **Purpose**: Correct gain and phase imbalance between I and Q channels
- **Operation**: Apply correction matrix: [I' Q'] = [a b; c d] * [I Q]
- **Current**: Not explicitly implemented (assumed balanced)
- **Benefit**: Corrects hardware imperfections in mixers/ADCs
- **Priority**: Medium - important for hardware with I/Q imbalances

### 8. DC Blocking Filter (High-Pass)
- **Purpose**: Remove DC offset from IQ samples
- **Operation**: High-pass filter (FIR or IIR) to remove bias
- **Current**: Not explicitly implemented
- **Benefit**: Removes DC bias that can affect demodulation accuracy
- **Priority**: Low - can use existing FIR high-pass filter

### 9. Frequency Offset Corrector (Coarse CFO)
- **Purpose**: Large-scale frequency offset correction before fine PLL
- **Operation**: Mix with estimated offset frequency using complex exponential
- **Current**: Handled entirely by PLL/Costas loop
- **Benefit**: Two-stage correction (coarse + fine) for wider acquisition range
- **Priority**: Low - Costas loop already handles this

### 10. Automatic Frequency Control (AFC)
- **Purpose**: Track and correct slow frequency drift
- **Operation**: Estimate frequency from phase derivatives
- **Current**: Partially handled by PLL frequency tracking
- **Benefit**: Handles long-term frequency drift
- **Priority**: Low - PLL already tracks frequency

## Implementation Status

All processors are now implemented:

1. ✅ **FIR** - Matched filtering
2. ✅ **AGC** - Gain control  
3. ✅ **Costas Loop** - Carrier recovery
4. ✅ **Matched Filter Integrator** - Symbol integration
5. ✅ **Phase Rotator** - Phase correction
6. ✅ **Symbol Timing Recovery** - Timing correction (Gardner algorithm)
7. ✅ **I/Q Imbalance Corrector** - Channel correction
8. ✅ **DC Blocking Filter** - Bias removal (IIR high-pass)
9. ✅ **CFO Corrector** - Carrier frequency offset correction
10. ✅ **AFC** - Automatic frequency control

## Usage Pattern

All processors implement the `Processor` interface:

```go
type Processor interface {
    Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int
    Reset()
}
```

Processors can be chained together for a complete demodulation pipeline:

```go
// Example: FIR → AGC → Costas Loop
firProc := internal.NewFIR(coeffs...)
agcProc := internal.NewAGC(1.0, 0.001)
costasProc := internal.NewCostasLoop(symbolRate, loopBw, damping, modOrder)

// Chain processors (processors modify dst in-place)
processed := firProc.Process(dst, src)
processed = agcProc.Process(dst, dst)  // dst becomes src for next stage
processed = costasProc.Process(dst, dst)
```

