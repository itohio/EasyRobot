Implementation Plan — `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\qam`
================================================================================================

## 1. References & prerequisites

1. Align with `D:\Projects\itohio\golang\EasyRobot\.cursorrules` (float32, math32, streaming processors, minimal allocations, hot-path optimizations).
2. Follow `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\qam\SPEC.md`.
3. Reuse building blocks from:
   - `x/math/dsp`: raised/root raised cosine generators, IQ oscillators, noise sources, matched filters/Goertzel, envelope builders.
   - `x/math/primitive/fp32`: vector ops for accumulation, peak detection, ring buffers with minimal allocation.
   - `x/math/filter/pll`: PLL already validated in MFSK/MPSK; reuse for carrier/phase tracking.
   - `x/buffer`: `BitsReader`, `BitsWriter`, `BitRing`, sample buffer utilities.
   - `x/chain`: `Processor` interface semantics (streaming, nil-destination sizing, partial consumption).

## 2. Goals

- Streaming M-QAM **Modulator** and **Demodulator** supporting Gray-mapped square constellations (4,16,64; extendable to others).
- Analytical helpers: bandwidth, symbol separation (amplitude grid spacing), theoretical Eb/N0 & SNR with configurable margins (note: each M has different SNR thresholds).
- Deterministic roundtrip & noise tests mirroring requirements (1 byte, 4 bytes, 33 bytes, multiple SNRs, BitsPerSymbol ∈ {2,4,8} for 7-byte 15 dB scenario).
- Spectrally optimal symbol envelopes (raised cosine) produced via `dsp` generators.
- Preamble handling (alternating constellation points) for sync; demodulator never emits preamble bits.
- Reuse PLL from `filter/pll`.

## 3. Config & shared math

### 3.1 `Config` structure

- Fields similar to MFSK/MPSK: `SampleRate`, `CarrierHz`, `BitsPerSymbol`, `SamplesPerSymbol`, `Amplitude`, `Envelope`, `RaisedCosineBeta`, `PreambleLength`, `EnergyThreshold`, `TransmissionGap`, `LoopBandwidth`, `LoopDamping`, `SilenceSamples`.
- Validation:
  - `BitsPerSymbol ∈ [2..8]` (4-QAM/QPSK upward). For BitsPerSymbol odd (e.g., 3 bits per symbol) square QAM not possible; enforce even values? (Plan: support pow-of-four constellations 4,16,64 (2,4,6 bits). Add validation `constellationSize = 1 << BitsPerSymbol`; ensure sqrt(M) integer).
  - `SamplesPerSymbol` large enough to satisfy Nyquist for chosen sample rate & guard relative to highest frequency component (carrier ± half bandwidth).
  - Default envelope `raised_cosine` β=0.35.
- Helper methods:
  - `SymbolRate()`, `SamplesForBits(bits)`, `MaxBytesForSamples(samples)`.
  - `ConstellationLevels()` returns amplitude levels per dimension.
  - `ConstellationNormalization()` returns amplitude scaling (√[3/(2(L²-1))]).
  - `RequiredBandwidth()` (tone + roll-off, B ≈ SymbolRate*(1+β)).
  - `SymbolSeparationMetric()` (Euclidean spacing between adjacent constellation points).
  - `MinimumEbN0(targetBER)` per QAM formula (M-QAM theoretical thresholds). Provide values for M=4 (~9.6 dB), 16 (~15 dB), 64 (~21 dB), adjust by BitsPerSymbol.
  - `MinimumSNR(targetBER, marginDb)` convert Eb/N0 to SNR via `SNR = Eb/N0 + 10*log10(BitsPerSymbol)` minus bandwidth ratio if needed; include margin parameter (2–3 dB typical, higher for large M).
  - `IsSNRSufficient(snrDb, targetBER, marginDb)`.

### 3.2 Constellation cache

- Precompute:
  - `levels = []float32{-(L-1), ..., +(L-1)}` odd steps.
  - `constellation = []complex64` of normalized amplitude.
  - Gray mapping tables: `symbolToBits`, `bitsToSymbol`.
  - Envelope samples for `SamplesPerSymbol`.

## 4. Modulator

### 4.1 Behavior

1. `NewModulator(cfg Config)` validates config, precomputes constellation + envelope.
2. `Process(dst buffer.Writer[float32], src buffer.Reader[byte]) int`:
   - Validate `src` implements `buffer.BitsReader`, `BitsLength == BitsPerSymbol`.
   - Nil `dst`: return `(preambleSymbols + payloadSymbols) * SamplesPerSymbol`.
   - Streaming loop:
     - If preamble not exhausted, emit alternating symbols (e.g., `symbol=0` and `symbol=M-1` for strong transitions).
     - Otherwise read one symbol from `src.Read(tmp[:])`, mask to `M-1`.
     - Convert to I/Q values via precomputed tables, apply envelope & carrier.
     - Maintain `phaseAccumulator`, `pendingSymbol`, `pendingSample`.
     - Write float32 samples to `dst`; stop when `dst` full or `src` empty.
3. After `src` drained, return samples written; next call continues using stored state.
4. `Reset()` resets accumulators, preamble counters, scratch buffers.

### 4.2 Implementation hints

- Use stack-based `[1]byte` scratch for symbol read.
- Envelope: `dsp.NewRaisedCosineGenerator(center, scale, beta, amplitudeScaling)`.
- Carrier mixing: `phase += phaseIncr` with `math32` sin/cos; consider precomputed `cosPhase`, `sinPhase` arrays to avoid trig per sample (like MFSK).

## 5. Demodulator

### 5.1 Behavior

1. `NewDemodulator(cfg Config)` sets up FIFO buffer, PLL, constellation tables.
2. `Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int`:
   - Validate `dst` implements `buffer.BitsWriter`.
   - If `dst == nil`, return best estimate of bytes: `fifo.Len()/SamplesPerSymbol * BitsPerSymbol / 8`.
   - Read from `src` into FIFO until at least `SamplesPerSymbol` available or `src` exhausted.
   - While enough samples:
     - Compute symbol energy; if below threshold, reset active state/preamble, drop to next block.
     - Apply PLL phase correction to symbol window (mix by `e^{-jθ}` sample-wise or rotate matched-filter output).
     - Matched filter (dot with envelope) to produce `Y`.
     - During preamble: compare detected symbol vs expected; increment preamble counters; do not emit bits.
     - After lock:
       - Quantize I/Q to nearest levels; map to bits.
       - Write bits to `dst` using `[1]byte` scratch; stop if `dst` full.
       - Update PLL phase error (decision-directed).
     - Slide FIFO by `SamplesPerSymbol`.
   - Maintain `TransmissionGap`: after consecutive low-energy windows, stop decoding until next burst.
3. `Reset()` clears FIFO, PLL, preamble counters.

### 5.2 State & loops

- FIFO size: `SamplesPerSymbol * 2`.
- Phase tracking: reuse `filter/pll.Loop` (call `Update(phaseError)` each symbol).
- Timing: initial version uses integer sample assumption; include fields for future Gardner/M&M integration.
- AGC stub: maintain running estimate of signal power to normalize amplitude before quantization (exponential average).

## 6. Analytical helpers & console utilities

- `ConstellationLevels()` & `ConstellationScale()` for mapping symbol bits -> float32 amplitude.
- `BandwidthInfo()` returning `(symbolRate, rolloffBandwidth, totalBandwidth)`.
- Theoretical SNR table function, e.g., `RequiredSNRForM(M, targetBER)` returning baseline Eb/N0 + margin (document values for M=4,16,64).
- Symbol separation metric defined as Euclidean distance between adjacent constellation points after normalization (√( (ΔI)^2 + (ΔQ)^2 )).

## 7. Roundtrip & noise tests

Helper functions duplicated/adapted from MFSK/MPSK modules:

- `buildBits`, `bytesToSymbols`, `symbolsToBytes`.
- `surroundWithSilence(samples, cfg)` adds 0.5 s silence before/after.
- `splitIntoChunks(samples, parts)` with `parts=3`.
- `addNoise(samples, snrDb)`.

### 7.1 Required tests (Go + Testify)

1. `TestRoundtrip_OneByte_NoNoise`  
   - Payload `{0x5a}`, BitsPerSymbol = default (e.g., 4 QAM?).  
   - Expect perfect recovery.
2. `TestRoundtrip_FourBytes_NoNoise`.
3. `TestRoundtrip_ThirtyThreeBytes_NoNoise`.
4. `TestRoundtrip_Noise_20dB` (payload 7 bytes, BitsPerSymbol default).
5. `TestRoundtrip_Noise_15dB`.
6. `TestRoundtrip_Noise_10dB`: run only if `cfg.IsSNRSufficient(10 dB)` true; otherwise `t.Skip`.
7. `TestRoundtrip_Noise_5dB`: same conditional skip.
8. `TestRoundtrip_Noise_BitsPerSymbol`  
   - Loop BitsPerSymbol ∈ {2,4,8}. (For 8 bits per symbol -> 256-QAM; ensure `SamplesPerSymbol` adjusted).  
   - For each, run 7-byte payload at 15 dB SNR.  
   - Skip scenarios below theoretical SNR limit, logging reason.

All tests use `SampleRate = 96_000`, `CarrierHz = 20_000`. Surround with 0.5 s silence before demod.

### 7.2 Additional tests

- `TestModulator_EstimateMatchesActual`.
- `TestDemodulator_NilDstSizing`.
- `TestConfig_BandwidthAndSNR`.
- Dedicated high-order QAM noise tests (similar to new MFSK 8/16 FSK tests) for 64-QAM verifying SNR requirements (≥22 dB).

## 8. Implementation order

1. **Config & helpers** (constellation math, normalization, bandwidth, SNR). Write table-driven tests.
2. **Waveform cache** (constellation + envelope).
3. **Modulator** (streaming, nil sizing, preamble). Add tests for partial buffers.
4. **Demodulator** (FIFO, energy detect, PLL, quantization). Start with integer timing; include AGC stub.
5. **Roundtrip tests** (All scenarios). Factor helper functions from previous modulators for reuse.
6. **Higher-order QAM tests** verifying SNR gating and noise tolerance.
7. **Documentation**: update SPEC if new constraints discovered; ensure README/test instructions updated if needed.

## 9. Key considerations / risks

- **High-order QAM sensitivity**: ensure `MinimumSNR` gating prevents impossible tests at low SNRs; log skip reasons.
- **AGC accuracy**: amplitude errors degrade decisions; maintain running RMS to normalize matched filter outputs before quantization.
- **PLL stability**: large constellations require tight phase lock; tune `LoopBandwidth` per BitsPerSymbol.
- **Performance**: avoid allocations inside symbol loops; reuse scratch arrays `[SamplesPerSymbol]float32`.
- **Streaming correctness**: ensure both modulator/demodulator resume seamlessly across `Process` calls (partial symbol writes, FIFO handling).

## 10. Deliverables checklist

- [ ] `Config` with analytical helpers & tests.
- [ ] Constellation/envelope cache builder.
- [ ] Streaming modulator.
- [ ] Streaming demodulator with PLL + AGC stub.
- [ ] Roundtrip + noise tests for required scenarios.
- [ ] Additional high-order QAM tests (SNR gating).
- [ ] Documentation updates (if SPEC needs adjustments).

This plan mirrors proven MFSK/MPSK workflows while incorporating QAM-specific requirements: amplitude quantization, AGC, higher SNR sensitivity, and square constellations with Gray mapping. It satisfies all user constraints and prepares for end-to-end roundtrip validation under multiple noise levels.

