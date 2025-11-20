# IIR/FIR Filter Specification

## Overview

The IIR/FIR filter package provides efficient digital filtering capabilities for signal processing applications. FIR and IIR filters are implemented in separate subpackages (`fir` and `iir`) with automatic coefficient calculation for common filter types. Filters focus on the core `Process()` method for efficiency and do not implement the legacy Filter interface.

## Components

### FIR Filter (`fir/` package)

**Purpose**: Finite Impulse Response digital filter for signal processing

**Description**: Implements FIR filters using direct convolution with circular buffers. FIR filters are inherently stable and can provide linear phase response. Suitable for applications requiring precise control over phase characteristics.

**Mathematical Model**:

FIR filter equation:
- `y[n] = Σ(k=0 to M-1) h[k] * x[n-k]`
- Where `h[k]` are the filter coefficients, `x[n]` is input, `y[n]` is output

**Type Definition**:
```go
type FIR struct {
    // Filter coefficients (impulse response)
    coeffs []float32

    // Internal circular buffer for input history
    buffer []float32
    bufIdx int

    // Filter order (length of coeffs - 1)
    order int
}
```

**Operations**:

1. **Initialization**:
   - `NewFIR(order int) *FIR`: Create FIR filter with given order
   - `NewFIRWithCoeffs(coeffs []float32) *FIR`: Create FIR filter with custom coefficients

2. **Coefficient Calculation**:
   - `LowPass(order int, cutoff, sampleRate float32) *FIR`: Low-pass filter design
   - `HighPass(order int, cutoff, sampleRate float32) *FIR`: High-pass filter design
   - `BandPass(order int, lowCutoff, highCutoff, sampleRate float32) *FIR`: Band-pass filter design
   - `BandStop(order int, lowCutoff, highCutoff, sampleRate float32) *FIR`: Band-stop filter design
   - `GeneralBand(order int, bands []BandSpec, sampleRate float32) *FIR`: General band filter

3. **Filtering**:
   - `Process(input float32) float32`: Process single sample
   - `ProcessBuffer(input, output []float32)`: Process buffer of samples
   - `Reset()`: Reset internal state

**Characteristics**:
- Linear phase response possible
- Always stable
- Higher order needed for sharp transitions
- Efficient circular buffer implementation
- Generic implementation for bounds checking optimization

### IIR Filter (`iir/` package)

**Purpose**: Infinite Impulse Response digital filter for signal processing

**Description**: Implements IIR filters using Direct Form II with efficient buffer management. IIR filters can achieve sharp frequency responses with lower orders than FIR filters.

**Mathematical Model**:

IIR filter difference equation:
- `y[n] = Σ(k=0 to M-1) b[k] * x[n-k] - Σ(k=1 to N-1) a[k] * y[n-k]`
- Where `b[k]` are feedforward coefficients, `a[k]` are feedback coefficients

**Type Definition**:
```go
type IIR struct {
    // Feedforward coefficients (numerator)
    b []float32

    // Feedback coefficients (denominator, a[0] = 1.0)
    a []float32

    // Internal buffers for input/output history (Direct Form II)
    w []float32 // Delay line for intermediate values

    // Filter order
    order int
}
```

**Operations**:

1. **Initialization**:
   - `NewIIR(order int) *IIR`: Create IIR filter with given order
   - `NewIIRWithCoeffs(b, a []float32) *IIR`: Create IIR filter with custom coefficients

2. **Coefficient Calculation**:
   - `ButterworthLowPass(order int, cutoff, sampleRate float32) *IIR`: Butterworth low-pass
   - `ButterworthHighPass(order int, cutoff, sampleRate float32) *IIR`: Butterworth high-pass
   - `ButterworthBandPass(order int, lowCutoff, highCutoff, sampleRate float32) *IIR`: Butterworth band-pass
   - `ButterworthBandStop(order int, lowCutoff, highCutoff, sampleRate float32) *IIR`: Butterworth band-stop
   - `ChebyshevILowPass(order int, cutoff, sampleRate, rippleDb float32) *IIR`: Chebyshev I low-pass
   - `ChebyshevIHighPass(order int, cutoff, sampleRate, rippleDb float32) *IIR`: Chebyshev I high-pass
   - `ChebyshevIBandPass(order int, lowCutoff, highCutoff, sampleRate, rippleDb float32) *IIR`: Chebyshev I band-pass
   - `ChebyshevIBandStop(order int, lowCutoff, highCutoff, sampleRate, rippleDb float32) *IIR`: Chebyshev I band-stop

3. **Filtering**:
   - `Process(input float32) float32`: Process single sample
   - `ProcessBuffer(input, output []float32)`: Process buffer of samples
   - `Reset()`: Reset internal state

**Characteristics**:
- Can achieve sharp responses with lower orders
- Potential phase nonlinearity
- May become unstable with poor coefficient design
- Efficient direct form II implementation
- Generic implementation for bounds checking optimization

### Filter Design (`design.go`)

**Purpose**: Coefficient calculation utilities for filter design

**Description**: Provides mathematical functions for calculating filter coefficients using established digital filter design methods.

**Supported Designs**:

1. **FIR Designs**:
   - Windowed sinc method
   - Remez exchange algorithm (future)
   - Frequency sampling method (future)

2. **IIR Designs**:
   - Butterworth: Maximally flat magnitude response
   - Chebyshev I: Equiripple passband, monotonic stopband
   - Chebyshev II: Monotonic passband, equiripple stopband (future)
   - Elliptic: Equiripple passband and stopband (future)

**Band Specifications**:
```go
type BandSpec struct {
    StartFreq float32  // Start frequency in Hz
    EndFreq   float32  // End frequency in Hz
    Gain      float32  // Desired gain (1.0 = pass, 0.0 = stop)
    Weight    float32  // Relative importance weight
}
```

## Design Decisions

### Architecture

1. **Buffer Management**:
   - Pre-allocated circular buffers for input/output history
   - Efficient indexing to avoid array shifts
   - Bounds checking optimized with generics where beneficial

2. **Coefficient Storage**:
   - FIR: Single coefficient array (impulse response)
   - IIR: Separate feedforward (b) and feedback (a) arrays
   - Normalized coefficients (a[0] = 1.0 for IIR)

3. **API Design**:
   - Focus on `Process()` method for core filtering functionality
   - Single sample processing for real-time applications
   - Buffer processing for batch operations
   - Separate subpackages to avoid interface complexity

### Performance

1. **Memory Allocation**:
   - Pre-allocate all buffers in constructors
   - Reuse buffers across processing calls
   - Minimal allocations in hot processing paths

2. **Numerical Stability**:
   - Proper coefficient normalization
   - Stability checks for IIR filters
   - High-precision coefficient calculation

3. **Optimization**:
   - Generic implementations to potentially eliminate bounds checks
   - Direct form II for IIR (most efficient)
   - Circular buffer indexing for FIR

### Compatibility

1. **Embedded Systems**:
   - `float32` precision throughout
   - Deterministic execution time
   - Minimal memory footprint

2. **Interface Consistency**:
   - Compatible with existing `Filter` interface
   - Works with `vec.Vector` for I/O
   - Follows package naming conventions

## Implementation Notes

### Current Implementation

- FIR filters with windowed sinc design
- IIR filters with Butterworth and Chebyshev I designs
- Circular buffer implementations
- Single sample and buffer processing modes
- Basic stability checking for IIR filters

### Missing Features

- Advanced IIR designs (Chebyshev II, Elliptic)
- Optimal FIR design (Remez algorithm)
- Real-time filter reconfiguration
- Multi-rate filtering
- Fixed-point implementations
- GPU acceleration support

### Usage Examples

#### FIR Low-pass Filter

```go
import "github.com/itohio/EasyRobot/x/math/filter/fir"

// Create 50th order low-pass FIR filter with 1000Hz cutoff at 44100Hz sample rate
filter := fir.NewFIRLowPass(50, 1000.0, 44100.0)

// Process single sample
output := filter.Process(inputSample)

// Process buffer
filter.ProcessBuffer(inputSamples, outputSamples)
```

#### IIR Butterworth Band-pass Filter

```go
import "github.com/itohio/EasyRobot/x/math/filter/iir"

// Create 4th order Butterworth band-pass filter from 300-3400Hz at 44100Hz
filter := iir.NewIIRButterworthBandPass(4, 300.0, 3400.0, 44100.0)

// Process samples
for _, sample := range inputSamples {
    outputSample := filter.Process(sample)
    // Use outputSample...
}
```

#### Custom FIR Filter

```go
// Design custom band filter
bands := []filter.BandSpec{
    {StartFreq: 0, EndFreq: 300, Gain: 1.0, Weight: 1.0},     // Pass low frequencies
    {StartFreq: 300, EndFreq: 3400, Gain: 1.0, Weight: 1.0},   // Pass voice band
    {StartFreq: 3400, EndFreq: 22050, Gain: 0.0, Weight: 1.0}, // Stop high frequencies
}

filter := filter.NewFIRGeneralBand(100, bands, 44100.0)
```

## Questions

1. Should we support more advanced IIR designs (Chebyshev II, Elliptic)?
2. Should we implement optimal FIR design algorithms (Remez exchange)?
3. How to handle filter stability checking and coefficient validation?
4. Should we support cascaded filter designs for higher orders?
5. Should we add support for fixed-point arithmetic?
6. How to optimize for SIMD operations?
7. Should we support real-time coefficient updates?
8. Should we add filter analysis functions (frequency response, group delay)?
9. How to handle different window functions for FIR design?
10. Should we support multi-channel filtering?
