# FIR Filter Specification

## Overview

The `fir` package provides efficient Finite Impulse Response digital filter implementations. FIR filters are inherently stable and can provide linear phase response, making them ideal for applications requiring precise control over phase characteristics.

## Mathematical Model

FIR filter equation:
- `y[n] = Σ(k=0 to M-1) h[k] * x[n-k]`
- Where `h[k]` are the filter coefficients, `x[n]` is input, `y[n]` is output

## Type Definition

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

## API

### Constructor

```go
New(coeffs ...float32) *FIR
```
Creates a new FIR filter with the specified coefficients.

**Parameters:**
- `coeffs`: Filter coefficients as variadic float32 arguments

**Example:**
```go
// Create FIR filter with 3 coefficients [h0, h1, h2]
filter := fir.New(0.1, 0.5, 0.1)
```

### Methods

```go
Process(input float32) float32
```
Processes a single input sample and returns the filtered output.

```go
ProcessBuffer(input, output []float32)
```
Processes a buffer of input samples and stores results in output buffer.

```go
Reset()
```
Resets the internal filter state.

## Filter Design Functions

### Low-pass Filter
```go
NewLowPass(order int, cutoffHz, sampleRate float32) *FIR
```

### High-pass Filter
```go
NewHighPass(order int, cutoffHz, sampleRate float32) *FIR
```

### Band-pass Filter
```go
NewBandPass(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *FIR
```

### Band-stop Filter
```go
NewBandStop(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *FIR
```

### General Band Filter
```go
NewGeneralBand(order int, bands []BandSpec, sampleRate float32) *FIR
```

## Band Specification

```go
type BandSpec struct {
    StartFreq float32 // Start frequency in Hz
    EndFreq   float32 // End frequency in Hz
    Gain      float32 // Desired gain (1.0 = pass, 0.0 = stop)
    Weight    float32 // Relative importance weight
}
```

## Characteristics

- Linear phase response possible
- Always stable
- Higher order needed for sharp transitions
- Efficient circular buffer implementation
- Optimized for real-time processing

## Usage Examples

### Using Filter Design Functions

```go
import "github.com/itohio/EasyRobot/x/math/filter/fir"

// Create a low-pass FIR filter
filter := fir.NewLowPass(50, 1000.0, 44100.0)

// Process single sample
output := filter.Process(inputSample)

// Process buffer
filter.ProcessBuffer(inputSamples, outputSamples)

// Reset filter state
filter.Reset()
```

### Using Custom Coefficients

```go
import "github.com/itohio/EasyRobot/x/math/filter/fir"

// Create FIR filter with custom coefficients
// Moving average filter: [0.2, 0.2, 0.2, 0.2, 0.2]
filter := fir.New(0.2, 0.2, 0.2, 0.2, 0.2)

output := filter.Process(inputSample)
```</contents>
</xai:function_call">Create FIR SPEC.md
