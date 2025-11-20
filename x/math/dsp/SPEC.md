# DSP Package Specification

The `dsp` package provides comprehensive digital signal processing functionality for EasyRobot, optimized for fp32 operations and designed to work seamlessly with the vector and matrix interfaces.

## Overview

The package includes:
- Fast Fourier Transform (FFT) for 1D and 2D signals
- Convolution operations using FFT
- Windowing functions for signal processing
- Signal generation utilities
- Measurement and analysis functions
- Shift/rotation estimation for 2D signals

All operations work with `vecTypes.Vector` and `matTypes.Matrix` interfaces, ensuring compatibility with the broader EasyRobot math ecosystem.

## FFT Operations

### Unified FFT API

The FFT API provides separate types for 1D and 2D transforms with destination-based operations for optimal performance.

#### 1D FFT

```go
// Create 1D FFT processor for length 1024
fft1D := dsp.NewFFT1D(1024)

// Create source and destination buffers
signal := vec.New(1024)
spectrum := vec.New(1024)
reconstructed := vec.New(1024)

// Fill signal with data...
// Forward FFT
fft1D.Forward(signal, spectrum)

// Inverse FFT
fft1D.Backward(spectrum, reconstructed)
```

#### 2D FFT

```go
// Create 2D FFT processor for 256x256 matrices
fft2D := dsp.NewFFT2D(256, 256)

// Create source and destination buffers
signal2D := mat.New(256, 256)
spectrum2D := mat.New(256, 256)
reconstructed2D := mat.New(256, 256)

// Fill signal2D with data...
// Forward 2D FFT
fft2D.Forward(signal2D, spectrum2D)

// Inverse 2D FFT
fft2D.Backward(spectrum2D, reconstructed2D)
```

## Convolution

### 1D Convolution

```go
// Create FFT processor
fft1D := dsp.NewFFT1D(1024) // Must be >= len(signal) + len(kernel) - 1

// Create signals and result buffer
signal := vec.New(100)
kernel := vec.New(10)
result := vec.New(109) // len(signal) + len(kernel) - 1

// Fill signal and kernel...
// Convolve (destination-based)
fft1D.Convolve(signal, kernel, result)
```

### 2D Convolution

```go
// Create FFT processor
fft2D := dsp.NewFFT2D(256, 256) // Must be >= signal + kernel dimensions

// Create signals and result buffer
signal2D := mat.New(100, 100)
kernel2D := mat.New(10, 10)
result2D := mat.New(109, 109) // signal + kernel - 1

// Fill signal2D and kernel2D...
// Convolve (destination-based)
fft2D.Convolve(signal2D, kernel2D, result2D)
```

## Window Functions

### Available Window Types

- `Rectangular`: Uniform weighting
- `Sine`: Sine window
- `Lanczos`: Lanczos window
- `Triangular`: Triangular window
- `Hann`: Hann window (cosine taper)
- `BartlettHann`: Bartlett-Hann window
- `Hamming`: Hamming window
- `Blackman`: Blackman window
- `BlackmanHarris`: Blackman-Harris window
- `Nuttall`: Nuttall window
- `BlackmanNuttall`: Blackman-Nuttall window
- `FlatTop`: Flat-top window
- `Gaussian`: Gaussian window (requires sigma parameter)
- `Tukey`: Tukey window (requires alpha parameter)

### Usage Examples

```go
windows := dsp.NewWindows()

// Apply Hann window to vector
params := dsp.WindowParams{Type: dsp.Hann}
windows.ApplyToVector(signal, params)

// Create window vector
windowVec := windows.CreateWindowVector(1024, params)

// Apply separable window to matrix
windows.ApplyToMatrix(signal2D, params)

// Create 2D window matrix
window2D := windows.CreateWindowMatrix(256, 256, params)
```

### Parametric Windows

```go
// Gaussian window with sigma=0.5
gaussianParams := dsp.WindowParams{
    Type:   dsp.Gaussian,
    Param:  0.5,
}
windows.ApplyToVector(signal, gaussianParams)

// Tukey window with alpha=0.2
tukeyParams := dsp.WindowParams{
    Type:   dsp.Tukey,
    Param:  0.2,
}
windows.ApplyToVector(signal, tukeyParams)
```

## Signal Generation

### 1D Signals

```go
gen1D := dsp.NewSignalGenerator1D()

// Basic signals
zeros := gen1D.Zeros(1024)
ones := gen1D.Ones(1024)
constant := gen1D.Constant(1024, 3.14)

// Ramps and steps
ramp := gen1D.Ramp(1024)
rampNorm := gen1D.RampNormalized(1024)
step := gen1D.Step(1024, 512)

// Oscillators
sine := gen1D.Sinusoid(1024, 10.0, 1.0, 0.0, 1000.0)
cosine := gen1D.Cosine(1024, 10.0, 1.0, 0.0, 1000.0)
square := gen1D.Square(1024, 10.0, 1.0, 1000.0)
triangle := gen1D.Triangle(1024, 10.0, 1.0, 1000.0)

// Noise
uniformNoise := gen1D.NoiseUniform(1024, 0.1)
gaussianNoise := gen1D.NoiseGaussian(1024, 0.1)

// Special signals
impulse := gen1D.Impulse(1024, 512)
exponential := gen1D.Exponential(1024, 0.01)
chirp := gen1D.Chirp(1024, 1.0, 50.0, 1.0, 1000.0)
```

### 2D Signals

```go
gen2D := dsp.NewSignalGenerator2D()

// Basic 2D signals
zeros2D := gen2D.Zeros(256, 256)
ones2D := gen2D.Ones(256, 256)
constant2D := gen2D.Constant(256, 256, 1.0)

// Patterns
ramp2D := gen2D.Ramp(256, 256)
checkerboard := gen2D.Checkerboard(256, 256, 32)
gaussian2D := gen2D.Gaussian(256, 256, 50.0)

// 2D sinusoids
sinusoid2D := gen2D.Sinusoid(256, 256, 10.0, 10.0, 1.0)

// 2D noise
uniformNoise2D := gen2D.NoiseUniform(256, 256, 0.1)
gaussianNoise2D := gen2D.NoiseGaussian(256, 256, 0.1)
```

## Measurements

### Signal Measurements

```go
measurements := dsp.NewMeasurements()

// 1D measurements
result1D := measurements.Measure1D(signal)
fmt.Printf("RMS: %f, Peak: %f, SNR: %f dB\n",
    result1D.RMS, result1D.Peak, result1D.SNR)

// 2D measurements
result2D := measurements.Measure2D(signal2D)
fmt.Printf("RMS: %f, Mean: %f, Variance: %f\n",
    result2D.RMS, result2D.Mean, result2D.Variance)
```

### Frequency Analysis

```go
// Goertzel algorithm for single frequency detection
freqMeas := measurements.Goertzel(signal, 100.0, 1000.0)
fmt.Printf("Frequency: %f Hz, Amplitude: %f, Phase: %f rad\n",
    freqMeas.Frequency, freqMeas.Amplitude, freqMeas.Phase)
```

### Correlation Analysis

```go
// 1D cross-correlation
correlation1D := measurements.CrossCorrelate1D(signal1, signal2)
peakCorr1D := measurements.FindPeakCorrelation(correlation1D)
fmt.Printf("Max correlation: %f at lag %d\n",
    peakCorr1D.Coefficient, peakCorr1D.Lag)

// 2D cross-correlation
correlation2D := measurements.CrossCorrelate2D(image1, image2)
maxCorr, dy, dx := measurements.FindPeakCorrelation2D(correlation2D)
fmt.Printf("Max correlation: %f at shift (%d, %d)\n", maxCorr, dy, dx)
```

### Shift Estimation

```go
// Phase correlation for sub-pixel shift estimation
shiftY, shiftX := measurements.EstimateShift2D(image1, image2)
fmt.Printf("Estimated shift: (%d, %d) pixels\n", shiftY, shiftX)
```

## Digital Modulation

### IQ Modulation/Demodulation

```go
// IQ Encoder for modulation
encoder := dsp.NewIQEncoder(1000000.0, 100000.0) // 1MHz sample rate, 100kHz carrier

// Create baseband I/Q signals
I := vec.New(1000)
Q := vec.New(1000)
// Fill I and Q with data...

// Modulate onto carrier
modulated := encoder.Modulate(I, Q)

// IQ Decoder for demodulation
decoder := dsp.NewIQDecoder(1000000.0, 100000.0)

// Set up low-pass filter for baseband filtering
import "github.com/itohio/EasyRobot/x/math/filter/fir"
lowPass := fir.NewLowPass(32, 10000.0, 1000000.0) // 32 taps, 10kHz cutoff, 1MHz sample rate
decoder.SetFilter(lowPass)

// Demodulate back to I/Q (filtering applied automatically)
I_demod, Q_demod := decoder.Demodulate(modulated)
```

### PSK (Phase Shift Keying) Demodulation

```go
// PSK Decoder
pskDecoder := dsp.NewPSKDecoder(1000000.0, 100000.0, 10000.0, 2) // BPSK

// Demodulate PSK signal
symbols := pskDecoder.Demodulate(receivedSignal)

// symbols will contain 0s and 1s for BPSK
```

### FSK (Frequency Shift Keying) Demodulation

```go
// FSK Decoder
fskDecoder := dsp.NewFSKDecoder(1000000.0, 90000.0, 110000.0, 10000.0) // 90kHz/110kHz

// Demodulate FSK signal
symbols := fskDecoder.Demodulate(receivedSignal)

// symbols will contain 0s and 1s for FSK
```

### ASK (Amplitude Shift Keying) Demodulation

```go
// ASK Decoder
askDecoder := dsp.NewASKDecoder(1000000.0, 100000.0, 10000.0)

// Demodulate ASK signal
symbols := askDecoder.Demodulate(receivedSignal)

// symbols will contain 0s and 1s for ASK
```

### QAM (Quadrature Amplitude Modulation)

```go
// QAM-16 Encoder (4 bits per symbol)
qamEncoder := dsp.NewQAM16Encoder(sampleRate, carrierFreq, symbolRate)

// Encode bytes to amplitudes
encoder := qamEncoder.(dsp.ModulatorEncoder)
signal := vec.New(1000)
encoder.Encode(signal, []byte("Hello World"))

// QAM-16 Decoder
qamDecoder := dsp.NewQAM16Decoder(sampleRate, carrierFreq, symbolRate)
decoder := qamDecoder.(dsp.ModulatorDecoder)

// Set end-of-packet callback for protocol awareness
decoder.OnIsEndOfPacket(func(decoded []byte) bool {
    return len(decoded) >= 11 // "Hello World" is 11 bytes
})

// Decode amplitudes to bytes
decoded := make([]byte, 11)
decoder.Decode(decoded, signal)

// decoded now contains the original bytes
```

## Modulator Interfaces

The DSP package provides interfaces for implementing custom modulators:

```go
// Encoder interface for modulation
type ModulatorEncoder interface {
    Encode(amplitudes vecTypes.Vector, data []byte) bool
    Reset()
}

// Decoder interface for demodulation
type ModulatorDecoder interface {
    Decode(data []byte, amplitudes vecTypes.Vector) bool
    Reset()
    OnIsEndOfPacket(callback func([]byte) bool)
}
```

## Filter Requirements for Decoders

Digital modulation decoders require appropriate filtering to remove noise and unwanted frequency components. All decoders support the `filter.Processor[float32]` interface for optimal performance.

### Required Filter Types and Tuning

#### 1. Low-Pass Filters for Baseband Processing

**Purpose**: Remove high-frequency noise and carrier harmonics after demodulation.

**Recommended Implementation**: FIR low-pass filter from `x/math/filter/fir` package.

**Frequency Tuning Guidelines**:
- **PSK/QAM Decoders**: `cutoff = symbolRate/2`
  - Removes demodulation artifacts above Nyquist frequency
  - Example: For 25kHz symbol rate, use 12.5kHz cutoff
- **ASK Decoder (Envelope)**: `cutoff = symbolRate/2`
  - Smooths envelope detection output
  - Prevents symbol timing jitter

**Example Setup**:
```go
import "github.com/itohio/EasyRobot/x/math/filter/fir"

// For PSK decoder with 25kHz symbol rate
decoder := dsp.NewPSKDecoder(1000000, 100000, 25000, 4) // QPSK
lowPass := fir.NewLowPass(32, 12500, 1000000) // 32 taps, 12.5kHz cutoff, 1MHz sample rate
decoder.SetFilter(lowPass)
```

#### 2. Band-Pass Filters for Carrier Extraction (Optional)

**Purpose**: Improve carrier synchronization in noisy environments.

**Recommended Implementation**: FIR band-pass filter.

**Frequency Tuning**:
- **Center Frequency**: Carrier frequency
- **Bandwidth**: `symbolRate` to `2*symbolRate`
- **Example**: For 100kHz carrier, 25kHz symbol rate: 87.5kHz - 112.5kHz passband

#### 3. High-Pass Filters for DC Removal (Optional)

**Purpose**: Remove DC offset and low-frequency interference.

**Recommended Implementation**: FIR high-pass filter.

**Frequency Tuning**:
- **Cutoff**: `10-100 Hz` depending on application
- **Example**: 50Hz cutoff for audio applications

### Filter Performance Considerations

- **FIR vs IIR**: FIR filters preferred for linear phase response
- **Order/Length**: 16-64 taps typically sufficient for most applications
- **Sample Rate**: Filters should match decoder sample rate
- **Reset**: Call `filter.Reset()` when resetting decoders for continuous processing

### Integration with x/math/filter

```go
// FIR Low-pass filter (recommended)
lowPass := fir.NewLowPass(order, cutoffHz, sampleRate)
decoder.SetFilter(lowPass)

// FIR Band-pass filter
bandPass := fir.NewBandPass(order, lowCutoffHz, highCutoffHz, sampleRate)

// IIR filter (alternative, if phase linearity not critical)
iirFilter := iir.New(coeffs...) // Custom IIR coefficients
```

## Roundtrip Testing

The package includes comprehensive roundtrip tests for modulation schemes:

```go
// Example: QAM-16 roundtrip test with noise and padding
func TestQAM16Roundtrip(t *testing.T) {
    // 1. Encode bytes to amplitudes
    // 2. Add random padding to beginning
    // 3. Add -10dB noise
    // 4. Decode back and verify bytes match
    // 5. Uses streaming interface with callbacks
}
```

## Performance Considerations

### FFT Optimization
- Input signals are automatically zero-padded to the next power of 2 for optimal FFT performance
- 2D FFTs are computed as separate 1D FFTs on rows and columns
- All FFT operations use in-place computation where possible

### Memory Management
- Matrix operations use the existing `mat.New()` for memory allocation
- Vector operations use `vec.NewVector()` for consistency
- Consider reusing buffers for repeated operations to minimize allocations

### Numerical Precision
- All operations use `float32` for compatibility with EasyRobot's math primitives
- FFT implementation uses Cooley-Tukey algorithm with bit-reversal permutation
- Window functions are computed with high precision coefficients

## Error Handling

The package uses Go's standard error handling patterns:
- FFT operations may return errors for invalid input dimensions
- Window functions validate parameters before application
- Measurement functions handle edge cases (empty signals, etc.)

## Integration with EasyRobot Math Primitives

The DSP package is designed to work seamlessly with EasyRobot's existing math infrastructure:

- Uses `x/math/primitive/fp32` for low-level operations
- Compatible with `vecTypes.Vector` and `matTypes.Matrix` interfaces
- Leverages existing buffer pooling and memory management

## Example: Complete Signal Processing Pipeline

```go
// Create a noisy signal
gen := dsp.NewSignalGenerator()
cleanSignal := vec.New(1024)
noise := vec.New(1024)
noisySignal := vec.New(1024)

// Generate signals
gen.Sinusoid1D(1024, 10.0, 1.0, 0.0, 1000.0, cleanSignal)
gen.NoiseGaussian1D(1024, 0.1, noise)

// Add noise to signal (using vec operations)
fp32.Add(cleanSignal, noise, noisySignal)

// Apply window
windows := dsp.NewWindows()
windows.ApplyToVector(noisySignal, dsp.WindowParams{Type: dsp.Hann})

// FFT analysis
fft1D := dsp.NewFFT1D(1024)
spectrum := vec.New(1024)
fft1D.Forward(noisySignal, spectrum)

// Measure signal quality
measurements := dsp.NewMeasurements()
quality := measurements.Measure1D(noisySignal)
fmt.Printf("Signal quality - RMS: %f, SNR: %f dB\n", quality.RMS, quality.SNR)

// Frequency analysis
fundamental := measurements.Goertzel(noisySignal, 10.0, 1000.0)
fmt.Printf("Fundamental: %f Hz, Amplitude: %f\n",
    fundamental.Frequency, fundamental.Amplitude)
```

## Testing and Benchmarks

The package includes comprehensive tests and benchmarks:

```bash
# Run tests
go test ./x/math/dsp/...

# Run benchmarks
go test -bench=. ./x/math/dsp/...

# Run with CPU profiling
go test -bench=. -cpuprofile=cpu.prof ./x/math/dsp/
```

## Future Extensions

Planned enhancements include:
- Additional window functions (Kaiser, Dolph-Chebyshev)
- Spectrogram computation with windowed FFT
- Filter design (FIR, IIR)
- Wavelet transforms
- Real-time processing pipelines
