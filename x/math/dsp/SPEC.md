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
- Line coding (Manchester encoding/decoding)

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
// Apply Hann window to vector
params := dsp.WindowParams{Type: dsp.Hann}
dsp.ApplyWindowVector(signal, params)

// Create window vector into a destination buffer
windowBuf := make([]float32, 1024)
windowVec := dsp.CreateWindowVector(1024, params, windowBuf)

// Apply separable window to matrix
dsp.ApplyWindowMatrix(signal2D, params)

// Create 2D window matrix (allocates when dst is nil)
window2D := dsp.CreateWindowMatrix(256, 256, params, nil)
```

### Parametric Windows

```go
// Gaussian window with sigma=0.5
gaussianParams := dsp.WindowParams{
    Type:   dsp.Gaussian,
    Param:  0.5,
}
dsp.ApplyWindowVector(signal, gaussianParams)

// Tukey window with alpha=0.2
tukeyParams := dsp.WindowParams{
    Type:   dsp.Tukey,
    Param:  0.2,
}
dsp.ApplyWindowVector(signal, tukeyParams)
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
whiteNoise := gen1D.NoiseWhite(1024, 0.1)      // White noise (flat spectrum)
pinkNoise := gen1D.NoisePink(1024, 0.1)        // Pink noise (1/f, -3dB/octave)
brownNoise := gen1D.NoiseBrown(1024, 0.1)      // Brown noise (1/f^2, -6dB/octave)

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
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/iq"
)

// IQ Encoder for modulation
encoder := iq.NewEncoder(1000000.0, 100000.0) // 1MHz sample rate, 100kHz carrier

// Create baseband I/Q signals
I := vec.New(1000)
Q := vec.New(1000)
// Fill I and Q with data...

// Modulate onto carrier
modulated := encoder.Modulate(I, Q)

// IQ Decoder for demodulation
decoder := iq.NewDecoder(1000000.0, 100000.0)

// Set up low-pass filter for baseband filtering
import "github.com/itohio/EasyRobot/x/math/filter/fir"
lowPass := fir.NewLowPass(32, 10000.0, 1000000.0) // 32 taps, 10kHz cutoff, 1MHz sample rate
decoder.SetFilter(lowPass)

// Demodulate back to I/Q (filtering applied automatically)
I_demod, Q_demod := decoder.Demodulate(modulated)
```

### PSK (Phase Shift Keying) Encoding/Decoding

```go
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/mpsk"
)

// PSK Encoder (BPSK, QPSK, 8-PSK, 16-PSK)
pskEncoder := mpsk.NewEncoder(1000000.0, 100000.0, 10000.0, 2) // BPSK
// Modulation order: 2=BPSK, 4=QPSK, 8=8-PSK, 16=16-PSK

// Generate PSK modulated signal from data
data := []byte("Hello")
amplitude := float32(1.0)
signalLength := 1000
modulatedSignal := pskEncoder.GenerateSignal(data, amplitude, signalLength)

// PSK Decoder
pskDecoder := mpsk.NewDecoder(1000000.0, 100000.0, 10000.0, 2) // BPSK

// Demodulate PSK signal
symbols := pskDecoder.Demodulate(receivedSignal)

// symbols will contain 0s and 1s for BPSK
```

### FSK (Frequency Shift Keying) Encoding/Decoding

```go
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
)

// Binary FSK Encoder (2 frequencies)
fskEncoder := mfsk.NewEncoder(1000000.0, 90000.0, 110000.0, 10000.0) // 90kHz/110kHz

// Multi-FSK Encoder (2, 4, 8, or 16 frequencies)
mFskEncoder := mfsk.NewMEncoder(1000000.0, 10000.0, 90000.0, 100000.0, 110000.0, 120000.0) // 4-FSK
// Supports 2, 4, 8, or 16 frequencies (must be power of 2)

// Generate FSK modulated signal from data
data := []byte("Hello")
amplitude := float32(1.0)
signalLength := 1000
modulatedSignal := fskEncoder.GenerateSignal(data, amplitude, signalLength)

// FSK Decoder
fskDecoder := mfsk.NewDecoder(1000000.0, 90000.0, 110000.0, 10000.0) // 90kHz/110kHz

// Demodulate FSK signal
symbols := fskDecoder.Demodulate(receivedSignal)

// symbols will contain 0s and 1s for binary FSK
```

#### FSK SNR Performance Characteristics

The FSK decoder has been tested for robustness at various Signal-to-Noise Ratio (SNR) levels. Performance characteristics:

**Preamble Detection:**
- **30-20 dB SNR**: 100% preamble detection success rate
- **18-12 dB SNR**: 98-100% preamble detection success rate
- **10-8 dB SNR**: 92-96% preamble detection success rate
- **6-0 dB SNR**: 42-90% preamble detection success rate

**Minimum SNR Requirements:**
- **Target**: 20 dB SNR (100% preamble detection)
- **Minimum**: 12 dB SNR (98% preamble detection)
- **Marginal**: 8-10 dB SNR (92-96% preamble detection)

**Decoder Features for Noise Robustness:**
1. **Low-Pass Filtering**: Automatic low-pass filter (16 taps, cutoff at 4x max frequency or 80% Nyquist) applied before frequency detection
2. **Adaptive Thresholds**: Frequency detection uses adaptive magnitude thresholds based on signal strength
3. **Preamble Detection**: Requires 12 consecutive preamble symbols (highest frequency) for clean signals, 8 for noisy signals
4. **START/END Markers**: 4-symbol START marker (lowest frequency) and 4-symbol END marker for packet boundaries
5. **Lenient Symbol Detection**: Allows some non-preamble symbols during preamble detection to handle noise

**Test Results Summary:**
- Preamble detection works reliably down to 12 dB SNR
- Bit decoding quality degrades significantly below 20 dB SNR
- Current implementation achieves target of 20 dB SNR for reliable operation
- Further improvements needed for reliable operation below 12 dB SNR

**Test Programs:**
- `tmp/fsk_snr_preamble.go`: Tests preamble detection at various SNR levels
- `tmp/fsk_snr_bit_quality.go`: Tests bit decoding quality (symbol-by-symbol) at various SNR levels

### ASK (Amplitude Shift Keying) Demodulation

```go
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/ask"
)

// ASK Decoder
askDecoder := ask.NewDecoder(1000000.0, 100000.0, 10000.0)

// Demodulate ASK signal
symbols := askDecoder.Demodulate(receivedSignal)

// symbols will contain 0s and 1s for ASK
```

### QAM (Quadrature Amplitude Modulation)

```go
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation"
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/qam"
)

// QAM-16 Encoder (4 bits per symbol)
qamEncoder := qam.NewEncoder(sampleRate, carrierFreq, symbolRate)

// Generate QAM modulated signal from data
data := []byte("Hello World")
amplitude := float32(1.0)
signalLength := 1000
modulatedSignal := qamEncoder.GenerateSignal(data, amplitude, signalLength)

// Alternative: Encode bytes to amplitudes using interface
encoder := qamEncoder.(modulation.ModulatorEncoder)
signal := vec.New(1000)
encoder.Encode(signal, []byte("Hello World"))

// QAM-16 Decoder
qamDecoder := qam.NewDecoder(sampleRate, carrierFreq, symbolRate)
decoder := qamDecoder.(modulation.ModulatorDecoder)

// Set end-of-packet callback for protocol awareness
decoder.OnIsEndOfPacket(func(decoded []byte) bool {
    return len(decoded) >= 11 // "Hello World" is 11 bytes
})

// Decode amplitudes to bytes
decoded := make([]byte, 11)
decoder.Decode(decoded, signal)

// decoded now contains the original bytes
```

### Manchester Line Coding

Manchester encoding is a line coding scheme where each data bit is encoded as a transition, providing built-in clock recovery and DC balance. The implementation uses optimized 4-bit lookup tables (64 bytes total) for high performance.

#### Encoding Variants

**Standard Manchester:**
- `0` → `01` (low-to-high transition)
- `1` → `10` (high-to-low transition)

**IEEE 802.3 Manchester:**
- `0` → `10` (high-to-low transition)
- `1` → `01` (low-to-high transition)

#### Basic Usage

```go
// Standard Manchester Encoder
encoder := dsp.NewManchesterEncoder()

// Encode bytes to Manchester-encoded bytes
data := []byte{0xAA, 0x55}
encoded := make([]byte, len(data)*2) // Output is 2x input size
bytesWritten := encoder.Encode(encoded, data)

// Standard Manchester Decoder
decoder := dsp.NewManchesterDecoder()

// Decode Manchester-encoded bytes back to original
decoded := make([]byte, len(encoded)/2) // Output is half input size
bytesDecoded := decoder.Decode(decoded, encoded)
```

#### IEEE 802.3 Variant

```go
// IEEE 802.3 Manchester Encoder
encoder := dsp.NewManchesterEncoderIEEE8023()

// IEEE 802.3 Manchester Decoder
decoder := dsp.NewManchesterDecoderIEEE8023()
```

#### Handling Misaligned Signals

Manchester encoding guarantees a transition in the middle of each bit period, which helps with clock recovery. However, if decoding starts at the wrong bit position, the decoder may misinterpret symbols. The decoder supports bit offset adjustment for misaligned signals:

```go
decoder := dsp.NewManchesterDecoder()

// Set bit offset for misaligned signals (0-7 bits)
decoder.SetBitOffset(3) // Skip first 3 bits before decoding

// Decode with offset applied
decoded := make([]byte, len(encoded)/2)
decoder.Decode(decoded, encoded)

// Reset offset
decoder.Reset()
```

**When to use bit offset:**
- Signal stream doesn't start at a byte boundary
- Clock recovery has identified the correct symbol boundary
- Processing partial byte streams

**Note:** For best results, use clock recovery/synchronization to find the correct symbol boundary before decoding. The bit offset is a simple mechanism for handling known misalignments.

#### Performance Characteristics

**Memory Usage:**
- **Lookup Tables**: 64 bytes total (16 bytes × 4 tables)
  - `encodeTable4Std`: 16 bytes
  - `encodeTable4IEEE`: 16 bytes
  - `decodeTable4Std`: 16 bytes
  - `decodeTable4IEEE`: 16 bytes
- **96% reduction** compared to 8-bit lookup tables (1.5KB → 64 bytes)

**Encoding Performance:**
- Processes 4 bits at a time using lookup tables
- 2 lookups per input byte (high 4 bits, low 4 bits)
- No bit manipulation loops in hot path

**Decoding Performance:**
- Fast path: processes 2 input bytes simultaneously when aligned
- Uses 4-bit lookup tables for efficient symbol decoding
- Falls back to bit-by-bit decoding for misaligned signals

**Cache Efficiency:**
- 64-byte lookup tables fit in a single CPU cache line
- Excellent cache locality and hit rates

#### Roundtrip Example

```go
encoder := dsp.NewManchesterEncoder()
decoder := dsp.NewManchesterDecoder()

// Original data
original := []byte{0x12, 0x34, 0x56, 0x78}

// Encode
encoded := make([]byte, len(original)*2)
encoder.Encode(encoded, original)

// Decode
decoded := make([]byte, len(original))
decoder.Decode(decoded, encoded)

// decoded now matches original
```

#### Use Cases

Manchester encoding is commonly used in:
- **Ethernet (IEEE 802.3)**: Physical layer encoding
- **RFID systems**: Data transmission with clock recovery
- **Serial communication**: DC-balanced signaling
- **Embedded systems**: Simple line coding with built-in synchronization

**Advantages:**
- Built-in clock recovery (transition in middle of each bit)
- DC balance (equal number of high and low states)
- Error detection (invalid symbols: 00 or 11)
- Simple implementation

**Trade-offs:**
- 2x bandwidth requirement (each bit becomes 2 bits)
- Not the most bandwidth-efficient encoding

#### Composition with Other Encoders/Decoders

Manchester encoder/decoder can be composed with other byte-level encoders/decoders or modulator encoders/decoders for flexible signal processing chains.

**Byte-Level Composition:**

```go
// Chain multiple byte encoders together
manchester1 := dsp.NewManchesterEncoder()
manchester2 := dsp.NewManchesterEncoder() // or another encoder
composedEncoder := dsp.NewComposedByteEncoder(manchester1, manchester2)

// Encode through both encoders
data := []byte{0xAA, 0x55}
encoded := make([]byte, len(data)*4) // Estimate size
composedEncoder.Encode(encoded, data)

// Chain multiple byte decoders (applied in reverse order)
composedDecoder := dsp.NewComposedByteDecoder(decoder1, decoder2)
decoded := make([]byte, len(encoded)/4)
composedDecoder.Decode(decoded, encoded)
```

**Composition with Modulator Encoders/Decoders:**

```go
// Chain Manchester encoder with BFSK encoder
manchesterEncoder := dsp.NewManchesterEncoder()
fskEncoder := dsp.NewFSKEncoder(sampleRate, freqLow, freqHigh, symbolRate)

// Create composed encoder: bytes -> Manchester -> bytes -> BFSK -> amplitudes
composedEncoder := dsp.NewModulatorEncoderWithByteEncoder(manchesterEncoder, fskEncoder)

data := []byte("Hello")
amplitudes := vec.New(1000)
composedEncoder.Encode(amplitudes, data)

// Chain BFSK decoder with Manchester decoder
fskDecoder := dsp.NewFSKDecoder(sampleRate, freqLow, freqHigh, symbolRate)
manchesterDecoder := dsp.NewManchesterDecoder()

// Create composed decoder: amplitudes -> BFSK -> bytes -> Manchester -> bytes
composedDecoder := dsp.NewModulatorDecoderWithByteDecoder(fskDecoder, manchesterDecoder)

decoded := make([]byte, len(data))
composedDecoder.Decode(decoded, amplitudes)
```

**Interface Compliance:**

- `ManchesterEncoder` implements `ByteEncoder` interface
- `ManchesterDecoder` implements `ByteDecoder` interface
- `ComposedByteEncoder` implements `ByteEncoder` interface
- `ComposedByteDecoder` implements `ByteDecoder` interface
- `ModulatorEncoderWithByteEncoder` implements `ModulatorEncoder` interface
- `ModulatorDecoderWithByteDecoder` implements `ModulatorDecoder` interface

This allows flexible composition of encoding/decoding stages in your signal processing pipeline.

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
dsp.ApplyWindowVector(noisySignal, dsp.WindowParams{Type: dsp.Hann})

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
