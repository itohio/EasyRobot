# DSSS Implementation Analysis: Roundtrip Decoding Failures

## Executive Summary

The DSSS modulator/demodulator implementation has several critical issues preventing correct roundtrip operation. The primary problems are phase synchronization failures, lack of symbol timing recovery, and missing carrier recovery mechanisms. The code assumes perfect phase and timing alignment between transmitter and receiver, which is unrealistic even in simulation.

## Critical Issues

not using math32 library.

### 1. Phase Continuity and Wrapping

**Location**: `Modulator.Encode()`

**Problem**:
```go
dsss.carrierPhase += 1.0
phase := 2 * math.Pi * dsss.carrierFreq * dsss.carrierPhase / dsss.sampleRate
```

The carrier phase accumulates indefinitely without wrapping. While mathematically treating it as a sample counter is valid, this leads to:
- Floating-point precision degradation over long sequences
- Potential overflow issues in extended operation
- Unnecessary magnitude growth

**Impact**: Medium - can cause gradual drift in long transmissions

**Fix**:
```go
dsss.carrierPhase = math32.Mod(dsss.carrierPhase + 1.0, dsss.sampleRate)
```

---

### 2. Phase Synchronization Failure ⚠️ **CRITICAL**

**Location**: `Modulator.Encode()` vs `Demodulator.correlateWithCode()`

**Problem**:

**Modulator** maintains continuous phase state:
```go
// State variable persists across Encode() calls
dsss.carrierPhase += 1.0
phase := 2 * math.Pi * dsss.carrierFreq * dsss.carrierPhase / dsss.sampleRate
```

**Demodulator** calculates phase independently:
```go
globalIdx := dsss.processedSamples + idx
phase := 2 * math.Pi * dsss.carrierFreq * float32(globalIdx) / dsss.sampleRate
```

**Critical Issues**:
1. The demodulator assumes phase starts at 0, but modulator can have arbitrary phase
2. Any Reset() call or new instance creates phase discontinuity
3. No mechanism to estimate or track actual carrier phase
4. Real-world signals have arbitrary phase offsets due to propagation delays

**Impact**: **CRITICAL** - This is likely the primary cause of roundtrip failure

**Explanation**: 

Coherent demodulation requires knowing the exact phase of the received carrier. Your code performs:
```
transmitted: A * chip * cos(ωt + φ₀)
received:    A * chip * cos(ωt + φ₀ + φᵤₙₖₙₒwₙ)
correlation: ∫ received * cos(ωt) dt
```

The unknown phase offset φᵤₙₖₙₒwₙ causes:
- Correlation results to be rotated by cos(φᵤₙₖₙₒwₙ)
- Sign ambiguity when phase approaches ±90°
- Complete signal loss at 90° phase offset

---

### 3. Missing Carrier Recovery

**Location**: `Demodulator.correlateWithCode()`

**Problem**: The demodulator uses coherent detection without any carrier recovery mechanism:

```go
carrier := float32(math.Cos(float64(phase)))
sum += samples[idx] * chipValue * carrier
```

This assumes perfect knowledge of carrier phase, which is never available in real systems.

**Standard Solutions**:
- **Costas Loop**: PLL-based carrier tracking for BPSK/QPSK
- **Decision-Directed Loop**: Uses detected symbols to update phase estimate
- **Differential Encoding**: Encode information in phase transitions (DBPSK)
- **Non-Coherent Detection**: Square-law detector (performance penalty)

**Impact**: **CRITICAL** - Makes real-world operation impossible

---

### 4. Symbol Timing Recovery Missing

**Location**: `Demodulator.Decode()`

**Problem**:
```go
for len(dsss.pending) >= dsss.samplesPerSymbol {
    window := dsss.pending[:dsss.samplesPerSymbol]
    // Process exactly samplesPerSymbol samples
    dsss.pending = dsss.pending[dsss.samplesPerSymbol:]
}
```

Assumptions:
- Symbol boundaries are perfectly aligned at sample 0
- No timing offset between transmitter and receiver
- Sample clocks are perfectly synchronized
- No multipath or delay

**Reality**:
- Symbol timing is arbitrary and unknown
- Need to search for optimal sampling instant
- Requires synchronization preamble
- Clock drift requires continuous tracking

**Impact**: **HIGH** - Even small timing offsets cause inter-symbol interference

**Standard Solutions**:
- Early-late gate synchronizer
- Gardner timing error detector
- Mueller-Müller algorithm
- Preamble-based coarse acquisition

---

### 5. Correlation Energy Normalization

**Location**: `Demodulator.correlateWithCode()`

**Problem**:
```go
func (dsss *Demodulator) correlateWithCode(samples []float32, code Code) float32 {
    var sum float32
    // ... accumulates raw correlation
    return sum  // Not normalized
}
```

Issues:
- Correlation magnitude depends on signal amplitude
- Threshold decisions become amplitude-dependent
- AGC (Automatic Gain Control) is required but not implemented
- Symbol energy varies with amplitude, affecting detection

**Impact**: **MEDIUM** - Degrades performance with amplitude variations

**Fix**:
```go
func (dsss *Demodulator) correlateWithCode(samples []float32, code Code) float32 {
    var sum float32
    var energy float32
    
    for chip := 0; chip < dsss.spreadingFactor; chip++ {
        chipStart := chip * dsss.samplesPerChip
        chipValue := code.GetBipolar(chip)
        
        for s := 0; s < dsss.samplesPerChip; s++ {
            idx := chipStart + s
            globalIdx := dsss.processedSamples + idx
            phase := 2 * math.Pi * dsss.carrierFreq * float32(globalIdx) / dsss.sampleRate
            carrier := float32(math.Cos(float64(phase)))
            
            sample := samples[idx] * carrier
            sum += sample * chipValue
            energy += sample * sample
        }
    }
    
    if energy > 1e-9 {
        return sum / float32(math.Sqrt(float64(energy)))
    }
    return 0
}
```

---

### 6. Binary DSSS Detection Logic

**Location**: `Demodulator.detectSymbol()`

**Problem**:
```go
if dsss.modulationOrder == 2 {
    corr := dsss.correlateWithCode(samples, dsss.code)
    if corr >= 0 {
        return 1
    }
    return 0
}
```

This correlates with a single code and checks sign. During encoding:
```go
if symbolIndex == 0 {
    chipValue = -chipValue // Bit 0 → transmit -code
} else {
    chipValue = +chipValue // Bit 1 → transmit +code
}
```

**Analysis**: The logic is theoretically correct but **completely broken by phase offset**:
- At 0° phase: correlation = +A (correct)
- At 90° phase: correlation = 0 (ambiguous)
- At 180° phase: correlation = -A (inverted)
- At arbitrary phase θ: correlation = A·cos(θ) (unreliable)

**Impact**: **CRITICAL** when combined with phase synchronization failure

---

### 7. Energy Threshold Detection

**Location**: `Demodulator.symbolHasEnergy()`

**Problem**:
```go
const symbolEnergyThreshold = 1e-3

func (dsss *Demodulator) symbolHasEnergy(samples []float32) bool {
    var sum float64
    for _, v := range samples {
        sum += float64(v * v)
    }
    rms := math.Sqrt(sum / float64(len(samples)))
    return float32(rms) >= dsss.energyThreshold
}
```

Issues:
- Hard-coded threshold assumes specific signal amplitude
- No automatic threshold adjustment
- No noise floor estimation
- Can miss valid symbols or accept noise

**Impact**: **LOW** - mainly affects noise handling

---

## Root Cause Analysis

The roundtrip test failure stems from a fundamental architectural flaw: **assuming perfect synchronization**.

### Synchronization Requirements for DSSS

A working DSSS receiver requires four synchronized domains:

1. **Carrier Frequency**: ω_rx ≈ ω_tx (handled by design)
2. **Carrier Phase**: φ_rx ≈ φ_tx (⚠️ **MISSING**)
3. **Symbol Timing**: Symbol boundaries aligned (⚠️ **MISSING**)
4. **Frame Synchronization**: Bit/byte boundaries (partially handled by bit buffer)

Your implementation has **none** of the required synchronization mechanisms beyond frequency matching.

### Why Roundtrip Fails

Even in a perfect simulation with no noise, the system fails because:

```
Modulator:
  - Generates signal with phase φ₀ (from carrierPhase state)
  - Symbol boundaries at arbitrary sample positions

Demodulator:
  - Assumes phase = 0
  - Assumes symbol boundaries at sample 0, samplesPerSymbol, 2×samplesPerSymbol, ...
  
Result:
  - Phase offset causes correlation to be scaled by cos(φ₀)
  - Wrong symbol timing causes integration over wrong samples
  - Combined effect: correlation values are meaningless
  - Detection: random bit errors
```

---

## Testing Strategy

### Phase 1: Eliminate Carrier (Baseband Testing)

Remove carrier modulation entirely to isolate DSSS logic:

**Modulator**:
```go
// Replace carrier modulation with direct chip output
ampData[samplesWritten] = dsss.amplitude * chipValue
// Remove: * carrier
```

**Demodulator**:
```go
// Remove carrier demodulation
sum += samples[idx] * chipValue
// Remove: * carrier
```

This tests:
- ✓ Spreading code generation
- ✓ Bit-to-symbol mapping
- ✓ Symbol-to-bit decoding
- ✓ Buffer management
- ✗ Carrier modulation (deferred)
- ✗ Phase synchronization (not needed)

### Phase 2: Add Carrier with Zero Phase

Force both modulator and demodulator to start at phase 0:

**Modulator**:
```go
func (dsss *Modulator) Reset() {
    dsss.chipIndex = 0
    dsss.sampleIndexInChip = 0
    dsss.carrierPhase = 0  // Ensure zero phase
    dsss.bitBuffer.Clear()
}
```

**Demodulator**:
```go
func (dsss *Demodulator) Reset() {
    // ... existing code ...
    dsss.processedSamples = 0  // Ensure phase alignment
}
```

Test immediately after Reset() on both.

### Phase 3: Implement Differential Encoding

Replace absolute phase detection with differential encoding:

**Modulator**:
```go
type Modulator struct {
    // ... existing fields ...
    previousSymbol int  // Add state
}

// Encode symbol as XOR with previous
currentSymbol := symbolValue ^ dsss.previousSymbol
// transmit currentSymbol
dsss.previousSymbol = currentSymbol
```

**Demodulator**:
```go
type Demodulator struct {
    // ... existing fields ...
    previousSymbol int  // Add state
}

// Decode differential
receivedSymbol := detectSymbol(window)
actualSymbol := receivedSymbol ^ dsss.previousSymbol
dsss.previousSymbol = receivedSymbol
```

This makes the system immune to fixed phase offsets (but not rotating phase).

### Phase 4: Add Preamble and Synchronization

Implement a known preamble for:
- Coarse timing acquisition
- Phase estimation
- AGC training

---

## Immediate Action Plan

### Priority 1: Basic Roundtrip (No Carrier)

1. **Remove carrier modulation** from both modulator and demodulator
2. **Verify spreading/despreading** works correctly
3. **Test with known patterns**: all zeros, all ones, alternating bits

### Priority 2: Carrier with Perfect Sync

1. **Force phase = 0** in both modulator and demodulator
2. **Test immediately after Reset()** on both
3. **Verify correlation** produces correct signs

### Priority 3: Energy Normalization

1. **Implement normalized correlation** as shown above
2. **Test with varying amplitudes** (0.1, 0.5, 1.0, 2.0)

### Priority 4: Production Features

1. **Differential encoding** for phase immunity
2. **Preamble-based sync** for timing acquisition
3. **Carrier tracking** for phase stability

---

## Code Fixes

### Fix 1: Baseband Mode (Testing)

Add a flag to disable carrier:

```go
type Modulator struct {
    // ... existing fields ...
    useCarrier bool
}

func NewModulator(..., useCarrier bool) *Modulator {
    // ...
    mod.useCarrier = useCarrier
    return mod
}

// In Encode():
var sample float32
if dsss.useCarrier {
    phase := 2 * math.Pi * dsss.carrierFreq * dsss.carrierPhase / dsss.sampleRate
    carrier := float32(math.Cos(float64(phase)))
    sample = dsss.amplitude * chipValue * carrier
    dsss.carrierPhase += 1.0
} else {
    sample = dsss.amplitude * chipValue
}
ampData[samplesWritten] = sample
```

Mirror in demodulator.

### Fix 2: Normalized Correlation

Replace `correlateWithCode()` as shown in Issue #5 above.

### Fix 3: Phase Wrapping

```go
dsss.carrierPhase = float32(math.Mod(float64(dsss.carrierPhase + 1.0), float64(dsss.sampleRate)))
```

### Fix 4: Differential BPSK (Simple Phase Immunity)

```go
type Modulator struct {
    // ... existing fields ...
    lastSymbol int
}

func (dsss *Modulator) Encode(...) int {
    // ... existing code to get symbolValue ...
    
    // Differential encoding: current = previous XOR data
    encodedSymbol := dsss.lastSymbol ^ symbolValue
    dsss.lastSymbol = encodedSymbol
    
    // Use encodedSymbol for spreading
    // ...
}
```

```go
type Demodulator struct {
    // ... existing fields ...
    lastSymbol int
}

func (dsss *Demodulator) detectSymbol(samples []float32) int {
    receivedSymbol := // ... existing detection ...
    
    // Differential decoding: data = current XOR previous
    dataSymbol := receivedSymbol ^ dsss.lastSymbol
    dsss.lastSymbol = receivedSymbol
    
    return dataSymbol
}
```

---

## Architecture Recommendations

For a production-grade DSSS system:

### 1. Layered Synchronization

```
┌─────────────────────────────────────┐
│  Frame Sync (preamble detection)    │
├─────────────────────────────────────┤
│  Symbol Timing Recovery              │
├─────────────────────────────────────┤
│  Carrier Recovery (Costas loop)      │
├─────────────────────────────────────┤
│  Matched Filter / Correlator         │
├─────────────────────────────────────┤
│  DSSS Despreading                    │
└─────────────────────────────────────┘
```

### 2. Signal Processing Chain

**Transmitter**:
```
Data → FEC Encode → Scramble → Symbol Map → Spread → Pulse Shape → Up-convert → Transmit
```

**Receiver**:
```
Receive → Down-convert → Matched Filter → Timing Sync → Despread → Carrier Sync → Decode → FEC Decode → Data
```

### 3. Required Components Not Present

- ❌ Pulse shaping filter (RRC, raised cosine)
- ❌ Matched filter (maximizes SNR)
- ❌ Timing error detector
- ❌ Phase error detector
- ❌ Loop filters (for tracking)
- ❌ Preamble generator/detector
- ❌ AGC
- ❌ Forward Error Correction

---

## Conclusion

The DSSS implementation is fundamentally broken for real-world use due to missing synchronization mechanisms. However, the core spreading/despreading logic appears sound.

**Recommended Path Forward**:

1. ✅ **Short term**: Test in baseband mode (no carrier) to verify core DSSS logic
2. ✅ **Medium term**: Implement differential encoding for phase robustness
3. ✅ **Long term**: Add proper synchronization (preamble, timing recovery, carrier tracking)

The good news: the code structure is clean and the spreading factor calculations are correct. The issues are in the synchronization domain, which can be added incrementally without restructuring the core implementation.