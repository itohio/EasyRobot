# IIR Filter Specification

## Overview

The `iir` package provides efficient Infinite Impulse Response digital filter implementations. IIR filters can achieve sharp frequency responses with lower orders than FIR filters, making them suitable for applications where computational efficiency is critical.

## Mathematical Model

IIR filter difference equation:
- `y[n] = Σ(k=0 to M-1) b[k] * x[n-k] - Σ(k=1 to N-1) a[k] * y[n-k]`
- Where `b[k]` are feedforward coefficients, `a[k]` are feedback coefficients

Note: `a[0]` is always normalized to 1.0.

## Type Definition

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

## API

### Constructor

```go
New(coeffs ...float32) *IIR
```
Creates a new IIR filter with the specified coefficients.

**Parameters:**
- `coeffs`: Filter coefficients as [b0, b1, ..., bn, a1, a2, ..., am] where a0 = 1.0

**Coefficient Format:**
For a transfer function H(z) = (b0 + b1*z^-1 + ... + bn*z^-n) / (1 + a1*z^-1 + ... + am*z^-m)

**Example:**
```go
// Create 2nd order IIR filter: H(z) = (0.1 + 0.2*z^-1 + 0.1*z^-2) / (1 + 0.3*z^-1 + 0.1*z^-2)
// coeffs = [b0, b1, b2, a1, a2] = [0.1, 0.2, 0.1, 0.3, 0.1]
filter := iir.New(0.1, 0.2, 0.1, 0.3, 0.1)
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

### Butterworth Filters

#### Low-pass
```go
NewButterworthLowPass(order int, cutoffHz, sampleRate float32) *IIR
```

#### High-pass
```go
NewButterworthHighPass(order int, cutoffHz, sampleRate float32) *IIR
```

#### Band-pass
```go
NewButterworthBandPass(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *IIR
```

#### Band-stop
```go
NewButterworthBandStop(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *IIR
```

### Chebyshev I Filters

#### Low-pass
```go
NewChebyshevILowPass(order int, cutoffHz, sampleRate, rippleDb float32) *IIR
```

#### High-pass
```go
NewChebyshevIHighPass(order int, cutoffHz, sampleRate, rippleDb float32) *IIR
```

#### Band-pass
```go
NewChebyshevIBandPass(order int, lowCutoffHz, highCutoffHz, sampleRate, rippleDb float32) *IIR
```

#### Band-stop
```go
NewChebyshevIBandStop(order int, lowCutoffHz, highCutoffHz, sampleRate, rippleDb float32) *IIR
```

## Characteristics

- Can achieve sharp responses with lower orders than FIR
- Potential phase nonlinearity
- May become unstable with poor coefficient design
- Efficient Direct Form II implementation
- Optimized for real-time processing

## Usage Examples

### Using Filter Design Functions

```go
import "github.com/itohio/EasyRobot/x/math/filter/iir"

// Create a Butterworth low-pass IIR filter
filter := iir.NewButterworthLowPass(4, 1000.0, 44100.0)

// Process single sample
output := filter.Process(inputSample)

// Process buffer
filter.ProcessBuffer(inputSamples, outputSamples)

// Reset filter state
filter.Reset()
```

### Using Custom Coefficients

```go
import "github.com/itohio/EasyRobot/x/math/filter/iir"

// Create 1st order low-pass IIR filter
// H(z) = (0.1 + 0.1*z^-1) / (1 + 0.8*z^-1)
// coeffs = [b0, b1, a1] = [0.1, 0.1, 0.8]
filter := iir.New(0.1, 0.1, 0.8)

output := filter.Process(inputSample)
```

## Coefficient Format

For the `New(coeffs...)` constructor, coefficients should be provided as:
```
[b0, b1, b2, ..., bn, a1, a2, a3, ..., am]
```

Where:
- `b0, b1, ..., bn`: Feedforward (numerator) coefficients
- `a1, a2, ..., am`: Feedback (denominator) coefficients (a0 = 1.0)

Example for a 2nd order filter:
```go
// H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
filter := iir.New(b0, b1, b2, a1, a2)
```</contents>
</xai:function_call">Create IIR SPEC.md
