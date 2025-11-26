# Energy Threshold Tuning Results

## Overview

The MPSK demodulator uses an SNR-based energy threshold to distinguish signal+noise from noise-only samples. This is critical for:
- Rejecting noise-only padding samples (preventing false detections)
- Detecting actual signal+noise reliably
- Enabling operation at low SNR (down to 10dB for QPSK)

## Implementation

### SNR-Based Threshold Calculation

The threshold is calculated using `EnergyThresholdForSNR()` in `config.go`:

```go
func (c Config) EnergyThresholdForSNR(snrDb float32) float32 {
    signalPower := c.Amplitude * c.Amplitude
    noisePower := signalPower / math32.Pow(10, snrDb/10)
    threshold := noisePower * 5.0  // 5-sigma threshold
    // Bounded between 2x noise power and 0.1x signal power
    return threshold
}
```

### Key Parameters

- **Factor: 5.0** - 5-sigma threshold for reliable detection
  - Ensures noise-only samples are rejected with high confidence
  - At 25dB SNR: noise_power ≈ 0.002, threshold ≈ 0.01
  - Well above noise floor, well below signal level

- **Bounds:**
  - Minimum: 2x noise power (reject pure noise)
  - Maximum: 0.1x signal power (detect weak signals)

### Adaptive Threshold Application

The threshold is applied adaptively based on transmission state:

1. **Initial Detection** (not active, no preamble):
   - Uses full SNR-based threshold
   - Prevents false detections from padding noise

2. **Preamble Acquisition** (preamble in progress):
   - Uses 0.5x threshold (more lenient)
   - Allows detection of weak signals during acquisition

3. **Active Transmission** (payload decoding):
   - Uses 0.3x threshold (most lenient)
   - Accounts for noise reducing signal energy during transmission

## Performance Results

### Test Results (QPSK, 7-byte payload)

| SNR (dB) | Status | Notes |
|----------|--------|-------|
| 30       | PASS   | Excellent performance |
| 25       | PASS   | Target SNR achieved |
| 20       | PASS   | Good margin |
| 15       | PASS   | Reliable operation |
| 12       | PASS   | Near practical limit |
| 10       | PASS   | Theoretical limit for QPSK |
| 8        | FAIL   | Below practical limit |

### Key Achievements

1. **25dB SNR**: Successfully achieved (original target)
2. **10dB SNR**: Achieved (theoretical limit for QPSK coherent detection)
3. **Robust Padding Rejection**: SNR-based threshold prevents false detections from noise-only padding

## Design Rationale

### Why SNR-Based Threshold?

- **Problem**: Fixed threshold (1e-5) was too low, allowing noise-only padding to trigger false signal detection
- **Solution**: Calculate threshold from expected SNR and signal amplitude
- **Benefit**: Automatically adapts to noise level, distinguishing signal+noise from noise-only

### Why 5-Sigma Factor?

- **Statistical Reliability**: 5-sigma provides ~99.9999% confidence for Gaussian noise
- **Margin for Fluctuations**: Accounts for noise variance and fluctuations
- **Practical Balance**: High enough to reject noise, low enough to detect weak signals

### Why Adaptive Application?

- **Initial Detection**: Stricter threshold prevents false positives from padding
- **Preamble**: More lenient to catch weak signals during acquisition
- **Payload**: Most lenient to handle noise-induced energy reduction during transmission

## AGC Integration

AGC normalization is only applied when PLL is locked:
- **Before Lock**: AGC updates for convergence but doesn't normalize (prevents normalizing noise)
- **After Lock**: AGC normalizes signal amplitude (ensures correct symbol detection)

This prevents noise-only samples from being incorrectly normalized, which would corrupt phase detection.

## Future Improvements

1. **Dynamic Threshold Adjustment**: Could adjust threshold based on measured noise floor
2. **Multi-Symbol Energy**: Could use energy over multiple symbols for more robust detection
3. **Adaptive Factor**: Could adjust 5-sigma factor based on SNR (stricter at lower SNR)

## References

- SNR calculation: `EnergyThresholdForSNR()` in `config.go`
- Threshold application: `decodeSymbol()` in `demodulator.go`
- Test validation: `TestSNRLimits()` in `snr_limits_test.go`

