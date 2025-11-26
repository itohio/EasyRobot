# Implementation Plan – `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\ofdm`

## 1. Overview
- Deliver OFDM modem stages that follow the same processor pattern as the DSSS and MFSK modems (`chain.Processor`, nil-destination sizing, streaming state).
- Reuse existing DSP building blocks (FFT/IFFT, QAM/PSK constellation mapping, fp32 vector ops) instead of ad-hoc math.
- Provide analytical helpers (bandwidth/SNR/separation) and a deterministic test suite covering roundtrip + noise scenarios.
- Support configurable FFT sizes (64, 128, 256, 512), cyclic prefix lengths, and subcarrier modulation schemes (BPSK, QPSK, 16-QAM, 64-QAM).

## 2. Core Components

### 2.1 Configuration & Shared Math
- Implement `Config` in `ofdm/config.go` mirroring DSSS/MFSK structure:
  - Fields: samplerate, FFT size (N), cyclic prefix length (N_CP), bits per subcarrier, active/data/pilot subcarrier counts, constellation type, pilot pattern, preamble pattern, equalizer type, energy thresholds.
  - Validation ensures FFT size is power of 2, CP length < N, active subcarriers ≤ N, bits per subcarrier matches constellation.
  - Helper methods: `SymbolRate()`, `SubcarrierSpacing()`, `SamplesPerSymbol()`, `SamplesForBits(bits)`, `RequiredBandwidth()`, `MinimumSampleRate()`, `ProcessingGain()`, `OutputSNR(inputSNR)`, `SymbolSeparationMetric()`, `IsSNRSufficient(inputSNR, targetBER)`.
  - Provide `MaxSamplesForBits(bits)`/`MaxBytesForSamples(samples)` for nil-destination sizing in mod/dem.
- Constellation mapping: precompute QAM/PSK constellation tables (BPSK, QPSK, 16-QAM, 64-QAM) with Gray coding.
- Subcarrier allocation: define active/data/pilot subcarrier indices (skip DC, guard bands, insert pilots).
- Bit mapping helpers: Gray-coded conversions between bits and constellation symbols.

### 2.2 Modulator (Processor[float32, byte])
- Struct state: config, constellation table, subcarrier allocation, IFFT instance, symbol buffer (complex64, size N), CP buffer (complex64, size N_CP), sample buffer (float32, size N+N_CP), preamble pattern, preamble generated flag, current symbol subcarriers, symbol index, bit buffer, bit position, scratch `[1]byte`.
- `Process(dst, src)`:
  1. Validate `src.(buffer.BitsReader)` and `BitsLength()==cfg.BitsPerSubcarrier`.
  2. If `dst==nil`, estimate `(remainingPreamble + ceil(bits/k))*SamplesPerSymbol` using sizeReader adapter.
  3. Loop while `dst` has space:
     - If preamble not generated, generate preamble symbols (known pattern), perform IFFT, add CP, write samples.
     - Fetch symbol bits (`nextSymbol` handles reading one byte into scratch and masking to subcarrier width).
     - Map bits to constellation symbol (QAM/PSK lookup).
     - Allocate symbol to active subcarrier (skip DC, insert pilots at designated positions).
     - When all active subcarriers filled, perform IFFT on symbol buffer.
     - Add cyclic prefix (copy last N_CP samples, prepend to symbol).
     - Convert complex64 samples to float32 (take real part, or I/Q interleaved if complex baseband).
     - Write partial samples respecting `buffer.Writer` limits; store `pendingSample` to resume on next call.
  4. Return total samples written.
- Maintain state for streaming across multiple `Process` calls (partial symbols, bit buffer).
- `Reset()` clears pending state, regenerates preamble from start.

### 2.3 Demodulator (Processor[byte, float32])
- Struct: config, FFT instance, bounded FIFO (couple of OFDM symbols), state machine (`idle`, `sync`, `track`), channel estimate buffer (complex64, size N), symbol buffer (complex64, size N), sample buffer (float32, size N+N_CP), correlation buffer (complex64, size N), preamble pattern, running energy stats, frame detection state, CFO estimate, timing offset, equalizer state, scratch `[1]byte`.
- `Process(dst, src)`:
  1. Validate `dst.(buffer.BitsWriter)` and `BitsLength()==cfg.BitsPerSubcarrier`.
  2. Nil `dst`: return `(expectedSymbolsRemaining)*BitsPerSubcarrier*N_data/8`.
  3. Append samples to FIFO until either source drains or FIFO hits cap (drop consumed samples promptly).
  4. While FIFO contains ≥ `N + N_CP` and dst has capacity:
     - **Idle**: compute RMS via `fp32.Energy`. If > threshold, transition to `sync`, reset CFO/timing.
     - **Sync**: correlate with preamble pattern (sliding window, compute correlation peak). Once detected, estimate coarse timing and CFO, initialize channel estimate, switch to `track`.
     - **Track**:
       * Remove CP from symbol window.
       * Perform FFT on time-domain samples → frequency-domain symbols.
       * Update channel estimate using pilot subcarriers (if available).
       * Equalize symbols: `X̂[m] = Y[m] / Ĥ[m]` (ZF) or MMSE.
       * Correct residual CFO (rotate symbols).
       * Demap symbols to bits (nearest constellation point, Gray decoding).
       * Call `dst.Write` with decoded bytes.
       * Update timing offset if needed (early/late correlation on CP).
     - Advance FIFO by one OFDM symbol (include timing adjustment).
  5. If energy drops below threshold for `TransmissionTimeout` symbols, revert to idle and reset frame detection.
  6. Return total bytes written.
- `Reset()` empties FIFO, resets FFT/channel/timing/CFO, state=idle.

## 3. Supporting Systems

### 3.1 Composable Processors (modulation/internal)

**Key Insight**: Work with `complex64` buffers internally using composable processors from `modulation/internal`. This enables efficient, reusable signal processing stages.

**Existing Processors to Reuse**:
- **AGC** (`internal.AGC`): Normalize signal magnitude for consistent demodulation
- **CFO Corrector** (`internal.CFOCorrector`): Coarse carrier frequency offset correction
- **Phase Rotator** (`internal.PhaseRotator`): Fine phase correction
- **DC Blocking** (`internal.DCBlocking`): Remove DC bias
- **I/Q Imbalance** (`internal.IQImbalance`): Correct I/Q channel imbalances

**New Processors to Create in `modulation/internal`**:
- **FFT Processor** (`internal.FFTProcessor`): Forward FFT on complex64 buffers (N:1, changes sample count)
- **IFFT Processor** (`internal.IFFTProcessor`): Inverse FFT on complex64 buffers (N:1, changes sample count)
- **Cyclic Prefix Adder** (`internal.CPAdder`): Add CP to OFDM symbols (N → N+N_CP)
- **Cyclic Prefix Remover** (`internal.CPRemover`): Remove CP from OFDM symbols (N+N_CP → N)
- **Channel Estimator** (`internal.ChannelEstimator`): Pilot-based channel estimation with interpolation
- **Equalizer** (`internal.Equalizer`): ZF or MMSE equalization (in-place on frequency-domain symbols)

**Processor Pattern**:
All processors implement `internal.Processor` interface:
```go
type Processor interface {
    Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int
    Reset()
}
```

**Composition Example** (Demodulator pipeline):
```go
// Convert float32 samples to complex64
samplesToComplex := // ... converter ...

// Apply AGC
agc := internal.NewAGC(1.0, 0.001)
agc.Process(complexBuf, samplesToComplex)

// Remove DC
dcBlock := internal.NewDCBlocking(0.99)
dcBlock.Process(complexBuf, complexBuf)

// Coarse CFO correction
cfo := internal.NewCFOCorrector(estimatedCFO, sampleRate)
cfo.Process(complexBuf, complexBuf)

// Remove CP
cpRemover := internal.NewCPRemover(N, N_CP)
cpRemover.Process(fftBuf, complexBuf)  // Changes sample count

// FFT
fftProc := internal.NewFFTProcessor(N)
fftProc.Process(freqBuf, fftBuf)  // Changes sample count

// Channel estimation (uses pilots)
channelEst := internal.NewChannelEstimator(pilotIndices, pilotSymbols)
channelEst.Process(channelBuf, freqBuf)

// Equalization
equalizer := internal.NewEqualizer("ZF", channelBuf)
equalizer.Process(equalizedBuf, freqBuf)  // In-place on freqBuf

// Fine CFO correction (phase rotation per subcarrier)
phaseRot := internal.NewPhaseRotator(residualPhase)
phaseRot.Process(equalizedBuf, equalizedBuf)
```

### 3.2 FFT/IFFT Processors

**Implementation**: Create `internal.FFTProcessor` and `internal.IFFTProcessor` that wrap `x/math/dsp/fft1dc.FFT1DC` or similar.

**FFT Processor**:
- Input: `N` complex64 samples (time domain)
- Output: `N` complex64 samples (frequency domain)
- Uses `processInPlace` helper for efficient buffer handling
- Pre-allocates FFT instance (no allocations in hot path)

**IFFT Processor**:
- Input: `N` complex64 samples (frequency domain)
- Output: `N` complex64 samples (time domain)
- Includes 1/N normalization (verify FFT library scaling)

### 3.3 Cyclic Prefix Processors

**CP Adder**:
- Input: `N` complex64 samples (OFDM symbol)
- Output: `N + N_CP` complex64 samples (symbol + CP)
- Copies last `N_CP` samples, prepends to symbol
- Changes sample count, must read/write explicitly

**CP Remover**:
- Input: `N + N_CP` complex64 samples (symbol + CP)
- Output: `N` complex64 samples (OFDM symbol)
- Strips first `N_CP` samples, keeps last `N` samples
- Changes sample count, must read/write explicitly

### 3.4 Channel Estimation Processor

**Channel Estimator**:
- Input: `N` complex64 samples (frequency-domain received symbols)
- Output: `N` complex64 samples (channel estimate `H[m]`)
- Uses pilot subcarriers with known symbols to estimate `H[p] = Y[p] / X_p`
- Interpolates to data subcarriers (linear or DFT-based)
- Can be used in-place or with separate output buffer

### 3.5 Equalization Processor

**Equalizer**:
- Input: `N` complex64 samples (frequency-domain received symbols `Y[m]`)
- Output: `N` complex64 samples (equalized symbols `X̂[m]`)
- ZF: `X̂[m] = Y[m] / Ĥ[m]`
- MMSE: `X̂[m] = Y[m] · Ĥ*[m] / (|Ĥ[m]|² + σ²)`
- Uses channel estimate from `ChannelEstimator`
- In-place processing (modifies input buffer)

### 3.6 Other Supporting Systems

- **Constellation mapping**: Implement QAM/PSK symbol tables with Gray coding in `ofdm/constellation.go`.
- **Subcarrier allocation**: Define active/data/pilot subcarrier indices in `ofdm/subcarrier.go`.
- **Synchronization**: Preamble correlation, timing recovery, CFO estimation in `ofdm/sync.go`.
- **Noise utilities**: AWGN helper leveraging `dsp.NoiseWhiteGenerator` for SNR-controlled tests plus `appendSilence(samples)` helper.
- **Analytical helpers**: tests verifying Config methods (bandwidth/SNR/separation) against python3/numpy references (per `.cursorrules`).

## 4. Test Plan
1. **Roundtrip (∞ SNR)**:
   - 1 byte + 0.5 s silence before/after; split samples into three chunks, feed demod.Process thrice, ensure recovered byte matches and no preamble bytes emitted.
   - 4 bytes, 33 bytes – confirm streaming across multiple Process invocations and correct sample sizing.
2. **Noise scenarios** (7 bytes payload):
   - **Theoretical SNR limits** (with 2-3 dB implementation margin):
     - BPSK (1 bit/subcarrier): minimum ~12-13 dB SNR for BER = 10^-5
     - QPSK (2 bits/subcarrier): minimum ~12-13 dB SNR for BER = 10^-5
     - 16-QAM (4 bits/subcarrier): minimum ~17-18 dB SNR for BER = 10^-5
     - 64-QAM (6 bits/subcarrier): minimum ~22-23 dB SNR for BER = 10^-5
   - **Test scenarios**:
     - QPSK: Test at 15+ dB, 12+ dB (near limit), 20+ dB (comfortable margin)
     - 16-QAM: Test at 20+ dB, 17+ dB (near limit), 25+ dB (comfortable margin)
     - 64-QAM: Test at 25+ dB (current implementation requires higher SNR due to channel estimation and equalization complexity)
   - Tests automatically skip if SNR is below theoretical minimum with margin
   - Lower SNR tests (10 dB, 5 dB) may be skipped if below theoretical limits
3. **Sizing behaviour**: `Modulator.Process(nil, src)` equals actual samples produced; demod nil-dst returns expected bytes.
4. **Buffer exhaustion**: limit destination sample/byte buffers to ensure both processors resume properly without resetting state.
5. **Transmission detection**: ensure demod remains silent during leading/trailing silence and before preamble lock.
6. **Analytical helpers**: table-driven tests for `RequiredBandwidth`, `IsSNRSufficient`, `SymbolSeparationMetric`.
7. **CFO test**: inject CFO ±Δf/4, verify acquisition and tracking.
8. **Multipath test**: two-tap channel with delay < CP length, verify ISI mitigation.

Common test params: 96 kHz sample rate, N=64 FFT size, N_CP=16, QPSK modulation, `dsp.FFT` for IFFT/FFT.

## 5. Documentation
- Update `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\ofdm\SPEC.md` to mirror DSSS/MFSK spec structure:
  - Overview, Config table, Modulator/Demodulator sections (with streaming diagrams), FFT/IFFT description, analytical formulas, and test scenarios.
  - Reference the `math/dsp/fft` package for FFT/IFFT operations.
- Ensure GoDoc comments on all exported symbols mention processor semantics (BitsReader/BitsWriter expectations, nil-destination sizing).

## 6. Implementation Order

### Phase 1: Core Infrastructure
1. Build Config + analytical helpers (`ofdm/config.go`).
2. Implement constellation mapping (QAM/PSK tables with Gray coding) (`ofdm/constellation.go`).
3. Implement subcarrier allocation helpers (active/data/pilot indices) (`ofdm/subcarrier.go`).

### Phase 2: Composable Processors (modulation/internal)
4. **FFT Processor** (`internal/fft_processor.go`):
   - Wrap `x/math/dsp/fft1dc.FFT1DC` or similar
   - Implement `Processor` interface
   - Use `processInPlace` helper for efficient buffer handling
   - Pre-allocate FFT instance (no allocations in hot path)
   
5. **IFFT Processor** (`internal/ifft_processor.go`):
   - Wrap IFFT functionality
   - Include 1/N normalization
   - Same pattern as FFT processor

6. **Cyclic Prefix Adder** (`internal/cp_adder.go`):
   - Add CP to OFDM symbols (N → N+N_CP)
   - Changes sample count, must read/write explicitly
   - Similar pattern to `MatchedFilter` (buffering for sample count changes)

7. **Cyclic Prefix Remover** (`internal/cp_remover.go`):
   - Remove CP from OFDM symbols (N+N_CP → N)
   - Changes sample count, must read/write explicitly

8. **Channel Estimator** (`internal/channel_estimator.go`):
   - Pilot-based channel estimation
   - Linear interpolation to data subcarriers
   - Can process in-place or with separate output

9. **Equalizer** (`internal/equalizer.go`):
   - ZF and MMSE equalization
   - In-place processing on frequency-domain symbols
   - Uses channel estimate from `ChannelEstimator`

### Phase 3: Modulator
10. Implement Modulator (`ofdm/modulator.go`):
    - Use `IFFTProcessor` for IFFT
    - Use `CPAdder` for cyclic prefix
    - Compose processors efficiently
    - Handle float32 ↔ complex64 conversion
    - Sizing tests

### Phase 4: Demodulator
11. Implement synchronization (preamble detection, timing, CFO) (`ofdm/sync.go`).
12. Implement Demodulator (`ofdm/demodulator.go`):
    - Compose processors: AGC → DC Blocking → CFO → CP Remover → FFT → Channel Est → Equalizer → Phase Rot
    - Use existing `internal` processors where applicable
    - Handle complex64 ↔ float32 conversion
    - Frame detection and state machine

### Phase 5: Testing
13. Write AWGN/silence utilities and full roundtrip/noise tests.
14. Update SPEC/docs and run full test suite (`go test ./...`).

## 7. Key Implementation Details

### 7.1 FFT/IFFT Usage

**IFFT (Modulator)**:
```go
import "github.com/itohio/EasyRobot/x/math/dsp/fft"

// Pre-allocate IFFT instance
ifft := fft.NewIFFT(N)  // N = FFT size

// For each OFDM symbol:
// 1. Fill symbol buffer with frequency-domain symbols (complex64)
// 2. Perform IFFT
ifft.Transform(symbolBuffer)  // In-place transform

// 3. Convert complex64 to float32 (take real part for real output, or I/Q interleaved)
for i := range N {
    samples[i] = float32(real(symbolBuffer[i]))
    // Or for complex baseband:
    // samples[2*i] = float32(real(symbolBuffer[i]))
    // samples[2*i+1] = float32(imag(symbolBuffer[i]))
}
```

**FFT (Demodulator)**:
```go
import "github.com/itohio/EasyRobot/x/math/dsp/fft"

// Pre-allocate FFT instance
fftInst := fft.NewFFT(N)

// For each OFDM symbol (after CP removal):
// 1. Convert float32 samples to complex64
for i := range N {
    symbolBuffer[i] = complex(samples[i], 0)  // Real input
    // Or for complex baseband:
    // symbolBuffer[i] = complex(samples[2*i], samples[2*i+1])
}

// 2. Perform FFT
fftInst.Transform(symbolBuffer)  // In-place transform

// 3. Extract frequency-domain symbols
// symbolBuffer now contains Y[m] for m=0..N-1
```

**Note**: Verify FFT library scaling (IFFT typically includes 1/N normalization).

### 7.2 Constellation Mapping

**QPSK Example**:
```go
// Gray-coded QPSK constellation
var qpskConstellation = []complex64{
    complex(1.0/math32.Sqrt(2), 1.0/math32.Sqrt(2)),   // 00 → +1+j
    complex(-1.0/math32.Sqrt(2), 1.0/math32.Sqrt(2)),  // 01 → -1+j
    complex(-1.0/math32.Sqrt(2), -1.0/math32.Sqrt(2)), // 10 → -1-j
    complex(1.0/math32.Sqrt(2), -1.0/math32.Sqrt(2)),  // 11 → +1-j
}

// Map 2 bits to QPSK symbol
func mapBitsToSymbol(bits byte) complex64 {
    idx := bits & 0x03  // 2 bits
    return qpskConstellation[idx]
}

// Demap QPSK symbol to 2 bits (nearest neighbor)
func demapSymbolToBits(symbol complex64) byte {
    // Find nearest constellation point
    minDist := float32(math32.MaxFloat32)
    bestIdx := 0
    for i, c := range qpskConstellation {
        dist := abs(symbol - c)
        if dist < minDist {
            minDist = dist
            bestIdx = i
        }
    }
    return byte(bestIdx)
}
```

### 7.3 Cyclic Prefix

**Add CP (Modulator)**:
```go
// symbolBuffer contains N IFFT samples
// cpBuffer will contain N_CP samples (last N_CP samples of symbol)

// Copy last N_CP samples to CP buffer
copy(cpBuffer, symbolBuffer[N-N_CP:])

// Prepend CP to symbol
copy(outputBuffer, cpBuffer)        // CP first
copy(outputBuffer[N_CP:], symbolBuffer)  // Then symbol
```

**Remove CP (Demodulator)**:
```go
// inputBuffer contains N + N_CP samples
// Remove first N_CP samples, keep last N samples

copy(symbolBuffer, inputBuffer[N_CP:N_CP+N])
```

### 7.4 Channel Estimation

**Pilot-Based Estimation**:
```go
// For pilot subcarrier p with known symbol X_p
func estimateChannel(pilotSubcarriers []int, pilotSymbols []complex64, receivedSymbols []complex64) []complex64 {
    channelEstimate := make([]complex64, N)
    
    // Estimate at pilot positions
    for i, p := range pilotSubcarriers {
        channelEstimate[p] = receivedSymbols[p] / pilotSymbols[i]
    }
    
    // Interpolate to data subcarriers (linear interpolation)
    for m := range N {
        if channelEstimate[m] == 0 {  // Not a pilot
            // Find nearest pilots
            // ... linear interpolation logic ...
        }
    }
    
    return channelEstimate
}
```

### 7.5 Equalization

**Zero-Forcing (ZF)**:
```go
func equalizeZF(received []complex64, channelEstimate []complex64) []complex64 {
    equalized := make([]complex64, len(received))
    for m := range len(received) {
        if abs(channelEstimate[m]) > 1e-6 {  // Avoid division by zero
            equalized[m] = received[m] / channelEstimate[m]
        }
    }
    return equalized
}
```

**MMSE**:
```go
func equalizeMMSE(received []complex64, channelEstimate []complex64, noiseVariance float32) []complex64 {
    equalized := make([]complex64, len(received))
    for m := range len(received) {
        h := channelEstimate[m]
        hMagSq := real(h)*real(h) + imag(h)*imag(h)
        if hMagSq > 1e-6 {
            equalized[m] = received[m] * complex(real(h), -imag(h)) / complex(hMagSq + noiseVariance, 0)
        }
    }
    return equalized
}
```

### 7.6 Preamble Detection

**Correlation-Based Detection**:
```go
func detectPreamble(samples []float32, preamblePattern []complex64) (bool, int) {
    N := len(preamblePattern)
    maxCorr := float32(0)
    bestOffset := -1
    
    // Slide window over samples
    for offset := 0; offset <= len(samples) - 2*N; offset++ {
        // Correlate with known preamble pattern
        corr := float32(0)
        for i := range N {
            // Convert samples to complex, correlate
            // ... correlation logic ...
        }
        if corr > maxCorr {
            maxCorr = corr
            bestOffset = offset
        }
    }
    
    threshold := float32(0.5)  // Adjust based on noise
    if maxCorr > threshold {
        return true, bestOffset
    }
    return false, -1
}
```

## 8. File Structure

### 8.1 OFDM Package

```
x/math/dsp/modulation/ofdm/
├── SPEC.md                    (this file's companion spec)
├── IMPLEMENTATION_PLAN.md     (this file)
├── config.go                  (Configuration structures and validation)
├── constellation.go           (QAM/PSK constellation mapping)
├── subcarrier.go              (Subcarrier allocation helpers)
├── modulator.go               (OFDM modulator implementation)
├── demodulator.go             (OFDM demodulator implementation)
├── sync.go                    (Preamble detection, timing, CFO)
├── modulator_test.go          (Modulator unit tests)
├── demodulator_test.go        (Demodulator unit tests)
├── roundtrip_test.go          (End-to-end tests)
└── noise_test.go              (Noise resilience tests)
```

### 8.2 New Processors in modulation/internal

```
x/math/dsp/modulation/internal/
├── ... (existing processors: agc.go, cfo.go, phase_rotator.go, etc.)
├── fft_processor.go           (NEW: Forward FFT processor)
├── ifft_processor.go          (NEW: Inverse FFT processor)
├── cp_adder.go               (NEW: Cyclic prefix adder)
├── cp_remover.go             (NEW: Cyclic prefix remover)
├── channel_estimator.go      (NEW: Channel estimation processor)
└── equalizer.go              (NEW: ZF/MMSE equalization processor)
```

**Note**: Channel estimation and equalization could alternatively be implemented as helper functions in `ofdm/channel.go` and `ofdm/equalizer.go` if they don't need to be composable processors. However, implementing them as processors enables reuse in other modulation schemes and better testability.

## 9. Key Design Decisions

1. **FFT Size**: Support power-of-2 sizes (64, 128, 256, 512) for efficient FFT computation.
2. **Cyclic Prefix**: Configurable length, typically N/4 to N/8, must exceed channel delay spread.
3. **Subcarrier Allocation**: Skip DC subcarrier, allocate guard bands, insert pilots at fixed positions.
4. **Constellation**: Support BPSK, QPSK, 16-QAM, 64-QAM with Gray coding.
5. **Channel Estimation**: Pilot-based with linear interpolation (can be extended to DFT-based).
6. **Equalization**: ZF (simple) and MMSE (better performance, requires noise variance).
7. **Synchronization**: Preamble correlation for frame detection, pilot-based fine timing and CFO.
8. **State Management**: Maintain state across `Process` calls for streaming (partial symbols, bit buffer).
9. **Nil Destination**: Calculate theoretical maximum output size for buffer sizing.
10. **Transmission Detection**: Only write bytes when frame is detected and active.

## 10. Dependencies

- `github.com/itohio/EasyRobot/x/buffer` - BitsReader, BitsWriter, Writer, Reader interfaces
- `github.com/itohio/EasyRobot/x/chain` - Processor interface
- `github.com/chewxy/math32` - Float32 math operations
- `github.com/itohio/EasyRobot/x/math/primitive/fp32` - Float32 primitives (Energy, etc.)
- `github.com/itohio/EasyRobot/x/math/dsp` - FFT/IFFT (`fft1dc.FFT1DC` for complex64)
- `github.com/itohio/EasyRobot/x/math/dsp/modulation/internal` - Composable processors (AGC, CFO, Phase Rotator, DC Blocking, etc.)
- `github.com/itohio/EasyRobot/x/math/dsp` - Noise generators for testing

## 11. Test Parameters

### Standard Test Configuration
- **Sample Rate**: 96,000 Hz
- **FFT Size**: 64 (N=64)
- **Cyclic Prefix**: 16 samples (N_CP=16, N/4)
- **Subcarrier Spacing**: 96,000 / 64 = 1,500 Hz
- **Symbol Duration**: 64 / 96,000 = 0.667 ms (excluding CP)
- **Total Symbol Duration**: 80 / 96,000 = 0.833 ms (including CP)
- **Active Subcarriers**: 52 (skip DC and guards)
- **Data Subcarriers**: 48
- **Pilot Subcarriers**: 4
- **Bits Per Subcarrier**: 2 (QPSK)
- **Constellation**: QPSK (Gray-coded)
- **Equalizer**: ZF (simple) or MMSE (better performance)
- **Preamble**: Two identical OFDM symbols for correlation

### Noise Test Parameters
- **SNR Levels**: 20 dB, 15 dB, 12 dB (QPSK), 17 dB (16-QAM), 22 dB (64-QAM)
- **Noise Type**: White Gaussian noise (AWGN)
- **Test Data**: 7 bytes
- **Constellation**: QPSK, 16-QAM, 64-QAM

## 12. Notes

### 12.1 Composable Processor Benefits

- **Efficiency**: `processInPlace` helper enables zero-copy processing when `dst` implements `Viewer` interface
- **Reusability**: Processors can be shared across different modulation schemes (OFDM, QAM, PSK)
- **Testability**: Each processor can be unit tested independently
- **Composability**: Chain processors together for complete signal processing pipelines
- **Memory Efficiency**: Processors work on `complex64` buffers internally, avoiding unnecessary conversions

### 12.2 OFDM-Specific Considerations

- FFT/IFFT operations dominate processing complexity (~O(N log N) per symbol).
- CP overhead reduces spectral efficiency but provides ISI mitigation and timing tolerance.
- Channel estimation accuracy depends on pilot density and SNR.
- CFO must be corrected to avoid ICI (inter-carrier interference).
- Equalization compensates for frequency-selective fading.
- Preamble detection requires sufficient SNR for reliable correlation.
- Frame structure: preamble (for sync) + payload OFDM symbols.
- Demodulator buffer kept small (2-3 OFDM symbols) for efficient processing.
- Transmission detection prevents writing noise as data.
- Processing gain from FFT: ~10*log10(N) dB (e.g., N=64 → ~18 dB).

### 12.3 Implementation Tips

- **Buffer Management**: Use `buffer.Viewer[complex64]` when possible for in-place processing
- **Sample Count Changes**: FFT, IFFT, CP add/remove change sample counts - must read/write explicitly (not use `processInPlace`)
- **Hot Path Optimization**: Pre-allocate all buffers, avoid allocations in hot paths
- **Type Conversions**: Minimize float32 ↔ complex64 conversions - work in complex64 domain internally
- **Processor Chaining**: Compose processors efficiently: `agc.Process(buf, buf)` for in-place chains

### 12.4 Testing Considerations

- Lower SNR tests (10 dB, 5 dB) may not be feasible depending on constellation and channel conditions.
- Use `CalculateSNRLimit()` and `IsSNRSufficient()` to determine if test SNR levels are theoretically achievable.
- Use `CalculateOutputSNR()` to verify processing gain improvement.
- Use `EstimateBER()` to predict bit error rates for given SNR conditions.
- Symbol separation depends on constellation: QPSK has better separation than 64-QAM at same power.
- Test each processor independently before composing into full demodulator pipeline.

