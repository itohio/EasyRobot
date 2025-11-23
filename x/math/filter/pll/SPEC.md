# PLL (Phase-Locked Loop) Specification

## Overview

A general-purpose Phase-Locked Loop (PLL) for carrier phase and frequency tracking in digital communications. Used by M-FSK, M-PSK, QAM, and M-ASK demodulators.

## Purpose

The PLL tracks the carrier phase and frequency offset, providing:
- Phase correction for coherent detection
- Frequency offset (CFO) tracking
- Phase continuity across symbols

## Mathematical Model

### Discrete-Time PLL

The PLL is a PI (Proportional-Integral) controller:

```
Phase Error: ε[n] = arg(Z[n]) - θ[n]
Frequency Update: ω[n+1] = ω[n] + Ki * ε[n]
Phase Update: θ[n+1] = θ[n] + Kp * ε[n] + ω[n+1]
```

Where:
- `Z[n]` is the complex correlator output for the detected symbol
- `θ[n]` is the current phase estimate
- `ω[n]` is the frequency offset estimate
- `Kp` is the proportional gain
- `Ki` is the integral gain

### Loop Bandwidth

The loop bandwidth `B_loop` determines tracking speed vs noise:
- `B_loop ≈ 0.01-0.1 * symbol_rate` (typical)
- Higher bandwidth: faster tracking, more noise
- Lower bandwidth: slower tracking, less noise

### PI Controller Gains

From control theory, for a second-order loop:
```
Kp = 2 * ζ * ωn
Ki = ωn²
```

Where:
- `ζ` (zeta) is damping ratio (typically 0.707 for critical damping)
- `ωn` is natural frequency: `ωn = 2π * B_loop`

## API Design

### Interface

```go
type Filter struct {
    // Update updates PLL with phase error (in radians)
    Update(phaseError float32)
    
    // Phase returns current phase estimate (in radians)
    Phase() float32
    
    // FrequencyOffset returns current frequency offset estimate (in Hz)
    FrequencyOffset() float32
    
    // Reset resets PLL state
    Reset()
    
    // SetGains sets proportional and integral gains
    SetGains(kp, ki float32)
}
```

### Implementation

```go
type Filter struct {
    // PI controller gains
    kp float32 // Proportional gain
    ki float32 // Integral gain
    
    // State
    phase      float32 // Current phase estimate (radians)
    freqOffset float32 // Frequency offset accumulator (radians/sample)
    
    // Sample rate for frequency conversion
    sampleRate float32
}
```

## Usage Patterns

### For M-FSK

```go
// After detecting symbol with frequency k
freqIdx, magnitude := detector.DetectFrequency(samples)
correlatorOutput := computeCorrelator(samples, freqIdx) // Complex value
phaseError := math.Atan2(imag(correlatorOutput), real(correlatorOutput))
filter := pll.New(kp, ki, sampleRate)
filter.Update(phaseError)

// Apply phase correction to next symbol
correctedPhase := filter.Phase()
correctedSamples := rotatePhase(samples, -correctedPhase)
```

### For M-PSK/QAM

```go
// After detecting symbol
iVal, qVal := demodulateIQ(samples)
phaseError := computePhaseError(iVal, qVal, detectedSymbol)
filter := pll.New(kp, ki, sampleRate)
filter.Update(phaseError)

// Apply phase correction
correctedPhase := filter.Phase()
correctedIQ := rotateIQ(iVal, qVal, -correctedPhase)
```

### For M-ASK

```go
// ASK doesn't need phase tracking, but can use PLL for carrier recovery
envelope := math.Abs(complexSample)
phaseError := math.Atan2(imag(complexSample), real(complexSample))
filter := pll.New(kp, ki, sampleRate)
filter.Update(phaseError)
```

## Embedded Constraints

- **No allocations**: All state pre-allocated
- **Small memory**: ~16 bytes per PLL instance
- **Low CPU**: O(1) per update
- **Fixed-point optional**: Use float32 for simplicity

## Tuning Guidelines

### Initial Gains

For symbol rate `R_sym` and desired loop bandwidth `B_loop`:

```go
omega_n := 2 * math.Pi * B_loop / R_sym
zeta := 0.707 // Critical damping
kp := 2 * zeta * omega_n
ki := omega_n * omega_n
```

### Typical Values

- **BFSK/2-FSK**: `B_loop = 0.05 * R_sym`, `Kp = 0.1`, `Ki = 0.01`
- **8-FSK/16-FSK**: `B_loop = 0.02 * R_sym`, `Kp = 0.05`, `Ki = 0.005`
- **BPSK/QPSK**: `B_loop = 0.1 * R_sym`, `Kp = 0.2`, `Ki = 0.02`
- **QAM**: `B_loop = 0.05 * R_sym`, `Kp = 0.1`, `Ki = 0.01`

## Testing

- Unit test: PLL step response with known phase offset
- Unit test: PLL frequency tracking with constant CFO
- Integration test: PLL with M-FSK demodulator
- Integration test: PLL with M-PSK demodulator

## References

- "Digital Communications" by John Proakis
- "Phase-Locked Loops" by Roland Best
- "Software Defined Radio" by Tony J. Rouphael

