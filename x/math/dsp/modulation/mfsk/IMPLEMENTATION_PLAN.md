# Implementation Plan – `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\mfsk`

## 1. Overview
- Deliver coherent MFSK modem stages that follow the same processor pattern as the DSSS modem (`chain.Processor`, nil-destination sizing, streaming state).
- Reuse existing DSP building blocks (raised-cosine generators, Goertzel detectors, fp32 vector ops) instead of ad-hoc math.
- Provide analytical helpers (bandwidth/SNR/separation) and a deterministic test suite covering roundtrip + noise scenarios.
- Replace the obsolete PLL in `x/math/filter/pll` with a useful float32 implementation that can be shared by MFSK and future modulators.

## 2. Core Components
### 2.1 Configuration & Shared Math
- Implement `Config` in `mfsk/config.go` mirroring DSSS structure:
  - Fields: samplerate, carrier, tone spacing, bits per symbol, samples per symbol, amplitude, envelope type, preamble pattern/length, PLL/timing gains, energy thresholds.
  - Validation ensures tones fit within Nyquist, `BitsPerSymbol` matches number of tones, `SamplesPerSymbol > 0`.
  - Helper methods: `SymbolRate()`, `SamplesPerSymbol()`, `SamplesForBits(bits)`, `RequiredBandwidth()`, `MinimumSampleRate()`, `ProcessingGain()`, `OutputSNR(inputSNR)`, `SymbolSeparationMetric()`, `IsSNRSufficient(inputSNR, targetBER)`.
  - Provide `MaxSamplesForBits(bits)`/`MaxBytesForSamples(samples)` for nil-destination sizing in mod/dem.
- Tone/envelope cache: precompute phase increments for each tone; build symbol envelope via `dsp.NewRaisedCosineGenerator` (rectangular fallback).
- Bit mapping helpers: gray-coded conversions between bits and tone indices.

### 2.2 Modulator (Processor[float32, byte])
- Struct state: config, envelope slice, tone phases, `dsp.CosineGenerator`, pending symbol/sample counters, preamble index, scratch `[1]byte`.
- `Process(dst, src)`:
  1. Validate `src.(buffer.BitsReader)` and `BitsLength()==cfg.BitsPerSymbol`.
  2. If `dst==nil`, estimate `(remainingPreamble + ceil(bits/k))*SamplesPerSymbol` using sizeReader adapter.
  3. Loop while `dst` has space:
     - Fetch symbol (`nextSymbol` handles alternating preamble before payload, reading one byte into scratch and masking to symbol width).
     - Render symbol: fill scratch slice of `SamplesPerSymbol` by summing tone sinusoids with envelope. Use `dsp.CosineGenerator.GenerateWithOp` for carrier multiplication.
     - Write partial samples respecting `buffer.Writer` limits; store `pendingSample` to resume on next call.
  4. Return total samples written.
- Maintain phase continuity per tone and for the shared carrier oscillator.
- `Reset()` clears pending state, regenerates carrier, replays preamble from start.

## 3. Demodulator (Processor[byte, float32])
- Struct: config, bounded FIFO (couple of symbols), state machine (`idle`, `sync`, `track`), PLL instance (from new `math/filter/pll`), timing loop, Goertzel templates per tone, running energy stats, preamble progress, scratch `[1]byte`.
- `Process(dst, src)`:
  1. Validate `dst.(buffer.BitsWriter)` and `BitsLength()==cfg.BitsPerSymbol`.
  2. Nil `dst`: return `(expectedSymbolsRemaining)*BitsPerSymbol/8`.
  3. Append samples to FIFO until either source drains or FIFO hits cap (drop consumed samples promptly).
  4. While FIFO contains ≥ `SamplesPerSymbol` and dst has capacity:
     - **Idle**: compute RMS via `fp32.Energy`. If > threshold, transition to `sync`, reset PLL/timing.
     - **Sync**: correlate (via cached Goertzel templates) against alternating preamble tones. Each successful symbol increments `preambleProgress`; once reached, drop preamble symbol without writing and jump to `track`.
     - **Track**:
       * For each tone, run `dsp.Goertzel` (already implemented in `x/math/dsp`) over the symbol window to obtain magnitude/phase.
       * Choose tone via max magnitude; convert to bits and call `dst.Write`.
       * Update PLL with phase error from chosen tone (new PLL lives in `x/math/filter/pll`), update timing loop using early/late split of the same window (adjust by ±1 sample when drift accumulates).
     - Advance FIFO by one symbol (include timing adjustment).
  5. If energy drops below threshold for `TransmissionTimeout` symbols, revert to idle and reset preamble detection.
  6. Return total bytes written.
- `Reset()` empties FIFO, resets PLL/timing/preamble, state=idle.

## 4. Supporting Systems
- **Useful PLL**:
  - Implement a reusable PLL in `D:\Projects\itohio\golang\EasyRobot\x\math\filter\pll` (new subpackage).
  - API: `type Loop struct { ... }`, `func NewLoop(sampleRate, loopBandwidth, damping float32) *Loop`, `func (l *Loop) Step(phaseError float32) (sin, cos float32)` etc.
  - Replace the existing “useless” implementation entirely and update mfsk imports to the new loop.
- **Timing loop**: small helper in `mfsk/timing.go` performing early/late energy comparison with configurable gain.
- **Noise utilities**: AWGN helper leveraging `dsp.NoiseWhiteGenerator` for SNR-controlled tests plus `appendSilence(samples)` helper.
- **Analytical helpers**: tests verifying Config methods (bandwidth/SNR/separation) against python3/numpy references (per `.cursorrules`).

## 5. Test Plan
1. **Roundtrip (∞ SNR)**:
   - 1 byte + 0.5 s silence before/after; split samples into three chunks, feed demod.Process thrice, ensure recovered byte matches and no preamble bytes emitted.
   - 4 bytes, 33 bytes – confirm streaming across multiple Process invocations and correct sample sizing.
2. **Noise scenarios** (7 bytes payload):
   - **Theoretical SNR limits** (with 2.5 dB implementation margin):
     - 2-FSK: minimum ~15 dB SNR for BER = 10^-5
     - 4-FSK: minimum ~17 dB SNR for BER = 10^-5
     - 8-FSK: minimum ~20-22 dB SNR for BER = 10^-5 (higher due to frequency resolution limits)
   - **Test scenarios**:
     - 2-FSK: Test at 15+ dB, 12+ dB (near limit), 20+ dB (comfortable margin)
     - 4-FSK: Test at 17+ dB, 15+ dB (near limit), 20+ dB (comfortable margin)
     - 8-FSK: Test at 25+ dB (current implementation requires higher SNR due to Goertzel frequency resolution limits with 256 tones)
   - Tests automatically skip if SNR is below theoretical minimum with margin
   - Lower SNR tests (10 dB, 5 dB) may be skipped if below theoretical limits
3. **Sizing behaviour**: `Modulator.Process(nil, src)` equals actual samples produced; demod nil-dst returns expected bytes.
4. **Buffer exhaustion**: limit destination sample/byte buffers to ensure both processors resume properly without resetting state.
5. **Transmission detection**: ensure demod remains silent during leading/trailing silence and before preamble lock.
6. **Analytical helpers**: table-driven tests for `RequiredBandwidth`, `IsSNRSufficient`, `SymbolSeparationMetric`.

Common test params: 96 kHz sample rate, 20 kHz carrier (tone spacing derived), raised-cosine envelope, `dsp.Goertzel` for tone detection.

## 6. Documentation
- Update `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\mfsk\SPEC.md` to mirror DSSS spec structure:
  - Overview, Config table, Modulator/Demodulator sections (with streaming diagrams), PLL description, analytical formulas, and test scenarios.
  - Reference the new `math/filter/pll` subpackage as a reusable component.
- Ensure GoDoc comments on all exported symbols mention processor semantics (BitsReader/BitsWriter expectations, nil-destination sizing).

## 7. Implementation Order
1. Build Config + analytical helpers.
2. Implement new PLL in `x/math/filter/pll` and remove legacy stub.
3. Add envelope/tone cache helpers and Gray mapping.
4. Implement Modulator + sizing tests.
5. Implement Timing loop + Demodulator (use `dsp.Goertzel` for tone detection).
6. Write AWGN/silence utilities and full roundtrip/noise tests.
7. Update SPEC/docs and run full test suite (`go test ./...`).

