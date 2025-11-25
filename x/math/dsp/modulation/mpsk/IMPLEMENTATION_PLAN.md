Implementation Plan — `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\mpsk`
================================================================================================

## 1. References & prerequisites

1. Read and align with `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\mpsk\SPEC.md`.
2. Follow project-wide constraints from `D:\Projects\itohio\golang\EasyRobot\.cursorrules`.
3. Reuse existing building blocks:
   - `x/math/dsp`: raised/root raised cosine generators, IQ oscillators, noise sources, Goertzel/matched-filter utilities, energy calculators.
   - `x/math/primitive/fp32`: vector helpers for accumulation, complex math, buffering without allocations.
   - `x/math/filter/pll`: float32 PLL already validated in MFSK; required for MPSK carrier tracking.
4. Respect streaming interfaces:
   - `x/buffer`: `BitsReader`, `BitsWriter`, `BitRing`, sample buffers.
   - `x/chain`: `Processor[Dst, Src]` contract (`Process(dst, src) int`, nil-destination sizing semantics).

## 2. Goals / Success criteria

- Implement streaming M-PSK **Modulator** and **Demodulator** that satisfy buffer/processor requirements described by the user.
- Provide analytical helpers (bandwidth, symbol separation, theoretical SNR limit).
- Deliver deterministic roundtrip tests (modulator → optional noise → demodulator) covering all scenarios listed by the user.
- Prefer spectrally optimal envelopes (Raised Cosine / Root Raised Cosine) provided by `dsp` generators.
- Ensure demodulator never emits payload bits during silence / preamble-only periods.

## 3. Architecture overview

### 3.1 Shared Config (`Config`)

- Parameters: `SampleRate`, `CarrierHz`, `BitsPerSymbol`, `SamplesPerSymbol`, `Amplitude`, `Envelope`, `RaisedCosineBeta`, `PreambleLength`, `EnergyThreshold`, `TransmissionGap`, `LoopBandwidth`, `LoopDamping`, `SilenceSamples`.
- Validation must ensure:
  - `BitsPerSymbol` ∈ {1..8}.
  - `SamplesPerSymbol` high enough for Nyquist with current `CarrierHz` and `M = 2^BitsPerSymbol`.
  - `ToneSpacing` equivalent for PSK → `SymbolRate()` controls bandwidth; still provide helper for theoretical occupied BW.
  - PLL gains default via `filter/pll` helper.
- Helper methods:
  - `SymbolRate()`, `SamplesForBits(bits)`, `RequiredBandwidth()`, `SymbolSeparationMetric()` (phase spacing), `MinimumEbN0(targetBER)`, `MinimumSNR(targetBER, marginDb)`, `IsSNRSufficient(...)`.

### 3.2 Preamble & framing

- Modulator prepends alternating constellation phases (e.g., `0`, `M/2`) for coarse phase/timing acquisition.
- Demodulator tracks preamble separately; payload bits emitted only once lock is achieved.
- `PreambleLength` configurable, default 16 symbols.

## 4. Modulator design

### 4.1 Responsibilities

1. Validate `src` implements `buffer.BitsReader` and `BitsLength()` equals `cfg.BitsPerSymbol`.
2. If `dst == nil`, compute `(preambleSymbols + payloadSymbols)*SamplesPerSymbol` and return.
3. Stream-friendly `Process` loop:
   - Read one symbol (one `BitsReader.Read(tmp[:])`), mask to `M-1`.
   - Convert symbol index to phase `phase = basePhase[idx]`.
   - Render symbol samples into scratch buffer using chosen envelope (default raised cosine) and `dsp` oscillator (complex exponential for PSK). Maintain:
     - `phaseAccumulator` for continuous carrier.
     - `pendingSymbol`, `pendingSample` offsets to resume mid-symbol.
   - Write as many float32 samples as `dst` accepts; stop when `src` empty or `dst` full.
4. State to persist across calls:
   - Pending sample pointer.
   - Last phase accumulator to guarantee continuity.
   - Remaining preamble symbols count.
5. Silence insertion:
   - Optionally add `SilenceSamples` before/after payload (used by tests to produce 0.5 s guard). Could be part of helper that surrounds buffers in tests rather than modulator.

### 4.2 Data paths / helpers

- Precompute constellation table `constPhasors[m] = cos/sin`.
- Envelope generator per symbol using `dsp.NewRaisedCosineGenerator`.
- `math32` trigonometric helpers per `.cursorrules`.

## 5. Demodulator design

### 5.1 Responsibilities

1. Validate `dst` implements `buffer.BitsWriter` with matching `BitsLength()`.
2. Maintain compact FIFO (≤ 2 symbols) to avoid large allocations.
3. Streaming `Process`:
   - Read from `src` until at least `SamplesPerSymbol` available or `src` exhausted.
   - Energy detect vs `EnergyThreshold`.
   - During silence: reset lock state, drop buffered preamble metrics.
   - Once energy present, run preamble correlation:
     - Use simple alternating symbol detector while `preambleCount < PreambleLength`.
     - Do not emit bits during preamble.
   - After lock: for each symbol window:
     - Apply PLL phase correction using `filter/pll`.
     - Integrate (matched filter) across `SamplesPerSymbol` samples.
     - Decision: map `arg(Y)` to nearest M-PSK symbol.
     - Write symbol bits to `dst` (using `BitsWriter.Write` on `[1]byte` scratch).
     - Update PLL (decision-directed error) and optional timing loop (initial version can assume perfect timing with `SamplesPerSymbol` integer, but plan includes placeholder for timing tracking).
4. Transmission end detection: when energy falls below threshold for `TransmissionGap` consecutive symbols, stop emitting bits until energy returns.
5. Buffer management: after processing symbol, shift FIFO by `SamplesPerSymbol` (no large copies: use `copy` on kept slice).

### 5.2 Carrier & timing control

- Use `filter/pll.Loop` reused from MFSK:
  - `Update(phaseError)` returns current phase; use to rotate symbol samples, or rotate final integrator output.
  - Gains from `cfg.LoopBandwidth`, `cfg.LoopDamping`.
- Timing: start with integer-sample assumption; include TODO for Gardner/Mueller integration later but keep API ready (`timingPhase`, `timingError` fields).

## 6. Analytical helpers

1. **Bandwidth**:
   - PSK bandwidth ~ `R_sym*(1 + β)` (β = roll-off). Provide helper returning tone + rolloff bandwidth.
2. **Symbol separation**:
   - Angular distance `Δθ = 2π/M`, convert to abstract metric (useful for SNR discussions).
3. **SNR limit**:
   - Provide theoretical Eb/N0 approximations (reuse from MFSK plan but adjust constant for PSK).
4. **Sizing**:
   - `SamplesForBits(bits)` considering preamble; symmetrical `MaxBytesForSamples(samples)` for demod nil-dst sizing.

## 7. Testing strategy

All tests live under `D:\Projects\itohio\golang\EasyRobot\x\math\dsp\modulation\mpsk`.

### 7.1 Roundtrip harness

1. Helper functions:
   - `buildBits(bitsPerSymbol, symbols)` using `buffer.NewBits`.
   - `symbolsToBytes`, `bytesToSymbols` for verifying decoded data.
   - `surroundWithSilence(samples, cfg)` adds ±0.5 s (0.5 * 96 kHz) zeros.
   - `splitIntoChunks(samples, parts)` to simulate streaming `Process` calls (3 chunks).
   - `addNoise(samples, snrDb)` using `dsp.NewNoiseWhiteGenerator`.
2. `runRoundtrip(t, cfg, payload, snrDb)` flow:
   - Instantiate modulator/demodulator with same config.
   - `mod.Process(nil, bits)` to size.
   - Actual modulation to buffer, add silence + noise.
   - Split into 3 pieces; call `demod.Process(outputBits, chunkReader)` per chunk.
   - After processing, read decoded bits and compare with payload.

### 7.2 Required test cases

| Test | Config | Payload | SNR |
|------|--------|---------|-----|
| `TestRoundtrip_OneByte_NoNoise` | Default (BitsPerSymbol=2) | `{0x5a}` | `Inf` |
| `TestRoundtrip_FourBytes_NoNoise` | same | `{0x10,0x20,0x30,0x40}` | `Inf` |
| `TestRoundtrip_ThirtyThreeBytes_NoNoise` | same | 33-byte pseudo-random | `Inf` |
| `TestRoundtrip_Noise_20dB` | `payload=makePayload(7)` | 7 bytes | `20 dB` |
| `TestRoundtrip_Noise_15dB` | same | 7 bytes | `15 dB` |
| `TestRoundtrip_Noise_10dB` | if `cfg.IsSNRSufficient(10 dB, targetBER)` returns true | 7 bytes | `10 dB` |
| `TestRoundtrip_Noise_5dB` | same check for `5 dB`; skip with log if below theoretical limit | 7 bytes | `5 dB` |
| `TestRoundtrip_Noise_BitsPerSymbol` | Loop BitsPerSymbol ∈ {1,2,3} (2,4,8-PSK) | 7 bytes | `15 dB` each |

All tests run at `SampleRate = 96_000`, `CarrierHz = 20_000`.

### 7.3 Additional tests

- `TestConfig_BandwidthCalculations`: table-driven values for required bandwidth vs expected formulas.
- `TestConfig_SNRThresholds`: ensure `IsSNRSufficient` respects theoretical minimum + margin.
- `TestModulator_EstimateMatchesActual`: `Process(nil, ...)` equals actual samples produced.
- `TestDemodulator_NilDstSizing`: `Process(nil, src)` returns expected bytes.

## 8. Noise & envelope handling

- Use raised cosine envelope to reduce spectral splatter; default `Envelope = "raised_cosine"` with `RaisedCosineBeta = 0.35`.
- Tests wrap modulation output with silence before adding AWGN noise to avoid ramp transients.
- For SNR injection:
  - Compute instantaneous power, derive noise amplitude `sqrt(signalPower / 10^(snrDb/10))`.
  - Use `dsp.NoiseWhiteGenerator`.

## 9. Implementation steps

1. **Config & helpers**  
   - Port relevant helpers from MFSK, adapt formulas to PSK (tone count replaced with constellation size).  
   - Add unit tests for sizing and analytical helpers.
2. **Tone/constellation table (`waveform.go`)**  
   - Precompute constellation phasors and envelope once per config.
3. **Modulator (`modulator.go`)**  
   - Implement streaming Process logic, nil-destination sizing, preamble, state resets.
4. **Demodulator (`demodulator.go`)**  
   - FIFO management, energy detector, preamble guard, matched-filter integration, PLL updates.  
   - Add TODO hooks for timing recovery but keep first version integer-sample.
5. **Roundtrip tests**  
   - Implement helper functions and required test cases.
6. **Noise-specific tests**  
   - 8-/16-PSK targeted tests similar to new MFSK tests.
7. **Documentation updates**  
   - Summarize plan in SPEC if new insights learned during coding.

## 10. Risk & mitigation

- **Phase/timing lock instability**: start with simple integrator; use raised cosine envelope to reduce ISI; add generous preamble.
- **Performance**: keep allocations steady by reusing buffers (`[1]byte` scratch, FIFO slices). Move heavy math to reusable scratch arrays.
- **High-order PSK sensitivity**: tests skip low SNR scenarios when theoretical limit not met; log skip reasons.

## 11. Deliverables checklist

- [ ] Config + helpers + unit tests.
- [ ] Constellation/envelope cache.
- [ ] Streaming modulator (`chain.Processor` compliant).
- [ ] Streaming demodulator with PLL.
- [ ] Roundtrip + noise tests per matrix above.
- [ ] Documentation updates (SPEC references, README snippet if necessary).
- [ ] Future work notes (timing loop, soft decision outputs).

This plan mirrors the proven MFSK workflow, adapting core learnings (streaming buffers, PLL reuse, nil-destination sizing, analytical helpers) to the PSK domain while satisfying the new functional and testing requirements.

