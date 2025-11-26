package qam

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
)

// Config holds all tunable parameters for the MQAM modem.
type Config struct {
	SampleRate       float32
	CarrierHz        float32
	BitsPerSymbol    int
	SamplesPerSymbol int
	Amplitude        float32

	Envelope         string
	RaisedCosineBeta float32
	PreambleLength   int
	EnergyThreshold  float32
	TransmissionGap  int
	LoopBandwidth    float32
	LoopDamping      float32
	SilenceSamples   int
}

// DefaultConfig returns a safe baseline configuration.
func DefaultConfig() Config {
	return Config{
		SampleRate:       96_000,
		CarrierHz:        20_000,
		BitsPerSymbol:    4,   // 16-QAM by default
		SamplesPerSymbol: 384, // Large oversampling for comfortable filtering
		Amplitude:        0.8,
		Envelope:         "raised_cosine",
		RaisedCosineBeta: 0.35,
		PreambleLength:   0, // QAM doesn't currently support preambles
		EnergyThreshold:  1e-5,
		TransmissionGap:  4,
		LoopBandwidth:    0, // Will be derived from SymbolRate if unset
		LoopDamping:      0.707,
		SilenceSamples:   0,
	}
}

// Validate checks the configuration and applies defaults.
func (c *Config) Validate() error {
	if c.SampleRate <= 0 {
		return errors.New("qam: sample rate must be positive")
	}
	if c.CarrierHz <= 0 {
		c.CarrierHz = 1_000
	}
	if c.BitsPerSymbol < 2 || c.BitsPerSymbol > 8 || c.BitsPerSymbol&1 == 1 {
		return fmt.Errorf("qam: bits per symbol must be an even value in [2..8], got %d", c.BitsPerSymbol)
	}
	if c.SamplesPerSymbol <= 0 {
		return errors.New("qam: samples per symbol must be positive")
	}
	if c.Amplitude <= 0 {
		c.Amplitude = 1
	}
	if c.PreambleLength < 0 {
		c.PreambleLength = 0
	}
	if c.EnergyThreshold <= 0 {
		c.EnergyThreshold = 1e-5
	}
	if c.TransmissionGap <= 0 {
		c.TransmissionGap = 4
	}
	if c.RaisedCosineBeta <= 0 {
		c.RaisedCosineBeta = 0.35
	}
	if c.LoopBandwidth <= 0 {
		// Compute optimal bandwidth based on target SNR (default 25dB)
		// and preamble length
		c.LoopBandwidth = c.OptimalLoopBandwidth(25.0)
	}
	if c.LoopDamping <= 0 {
		c.LoopDamping = c.OptimalLoopDamping(25.0)
	}
	if err := c.ensureWithinNyquist(); err != nil {
		return err
	}
	return nil
}

func (c Config) constellationSize() int {
	return 1 << c.BitsPerSymbol
}

func (c Config) axisBits() int {
	return c.BitsPerSymbol >> 1
}

func (c Config) levelsPerDimension() int {
	return 1 << c.axisBits()
}

// SymbolRate returns the nominal symbol rate in Hz.
func (c Config) SymbolRate() float32 {
	return c.SampleRate / float32(c.SamplesPerSymbol)
}

// OptimalLoopBandwidth returns the optimal loop bandwidth for a given target SNR (in dB).
// Based on empirical testing, this selects bandwidth that balances convergence speed
// with noise rejection. The bandwidth is adjusted based on preamble length:
// - Longer preambles allow wider bandwidth for faster acquisition
// - Shorter preambles require narrower bandwidth for better noise filtering
// 
// Note: With AGC applied before PLL, signals are normalized, allowing slightly wider
// bandwidth since amplitude variations are reduced. However, we keep conservative values
// to maintain stability.
func (c Config) OptimalLoopBandwidth(targetSNRDb float32) float32 {
	symbolRate := c.SymbolRate()
	
	// Base bandwidth from SNR table (for 25dB: 0.020 = 2% of symbol rate)
	// With AGC normalization before PLL, we can use slightly wider bandwidth
	// since amplitude variations are reduced, but we keep conservative values
	var baseBw float32
	switch {
	case targetSNRDb >= 25:
		baseBw = 0.025 // 2.5% - slightly wider with AGC normalization
	case targetSNRDb >= 20:
		baseBw = 0.060 // 6% - for 20-25dB
	case targetSNRDb >= 15:
		baseBw = 0.025 // 2.5% - for 15-20dB
	case targetSNRDb >= 10:
		baseBw = 0.015 // 1.5% - for 10-15dB
	case targetSNRDb >= 8:
		baseBw = 0.012 // 1.2% - for 8-10dB
	default: // 3-7 dB
		baseBw = 0.006 // 0.6% - for very low SNR
	}
	
	// Adjust based on preamble length
	// Longer preambles allow wider bandwidth for faster acquisition
	// Formula: bw = baseBw * (1 + 0.5 * min(preambleLength/20, 1))
	// This gives up to 1.5x wider bandwidth with 20+ symbol preambles
	preambleFactor := float32(1.0)
	if c.PreambleLength > 0 {
		// Scale from 1.0 (no preamble) to 1.5 (20+ symbols)
		normalizedPreamble := float32(c.PreambleLength) / 20.0
		if normalizedPreamble > 1.0 {
			normalizedPreamble = 1.0
		}
		preambleFactor = 1.0 + 0.5*normalizedPreamble
	}
	
	return baseBw * preambleFactor * symbolRate
}

// OptimalLoopDamping returns the optimal damping ratio for a given target SNR (in dB).
// Based on empirical testing from PLL noise performance analysis.
func (c Config) OptimalLoopDamping(targetSNRDb float32) float32 {
	switch {
	case targetSNRDb >= 25:
		return 1.00 // Optimal for 25-30dB
	case targetSNRDb >= 20:
		return 1.00 // For 20-25dB
	case targetSNRDb >= 15:
		return 0.707 // Standard for 15-20dB
	case targetSNRDb >= 10:
		return 1.50 // Higher damping for 10-15dB
	case targetSNRDb >= 8:
		return 0.50 // Lower damping for 8-10dB
	default: // 3-7 dB
		return 0.707 // Standard for very low SNR
	}
}

// OptimalAcquisitionBandwidth returns the loop bandwidth for acquisition phase.
// This is typically wider than tracking bandwidth for faster convergence.
// With AGC normalization, we can use slightly wider acquisition bandwidth
// since amplitude is normalized before PLL processing.
func (c Config) OptimalAcquisitionBandwidth(targetSNRDb float32) float32 {
	// Use wider bandwidth for acquisition (1.8x tracking bandwidth with AGC)
	// but cap at reasonable maximum
	acqBw := c.OptimalLoopBandwidth(targetSNRDb) * 1.8
	maxBw := c.SymbolRate() * 0.12 // Cap at 12% of symbol rate (slightly higher with AGC)
	if acqBw > maxBw {
		acqBw = maxBw
	}
	return acqBw
}

// OptimalTrackingBandwidth returns the loop bandwidth for tracking phase.
// This is narrower than acquisition bandwidth for better noise filtering.
func (c Config) OptimalTrackingBandwidth(targetSNRDb float32) float32 {
	// Use narrower bandwidth for tracking (0.5-0.7x acquisition bandwidth)
	return c.OptimalLoopBandwidth(targetSNRDb) * 0.6
}

// SamplesForBits returns the number of interleaved IQ samples needed to emit n bits including preamble.
func (c Config) SamplesForBits(bits int) int {
	symbols := 0
	if bits > 0 {
		symbols = (bits + c.BitsPerSymbol - 1) / c.BitsPerSymbol
	}
	totalSymbols := symbols + c.PreambleLength
	return totalSymbols * c.SamplesPerSymbol * 2
}

// MaxBytesForSamples returns the maximum number of recoverable bytes for the provided interleaved samples.
func (c Config) MaxBytesForSamples(samples int) int {
	if samples <= 0 {
		return 0
	}
	symbols := samples / (c.SamplesPerSymbol * 2)
	if symbols <= c.PreambleLength {
		return 0
	}
	payloadSymbols := symbols - c.PreambleLength
	return payloadSymbols * c.BitsPerSymbol / 8
}

// ConstellationLevels returns the normalized amplitude levels per dimension.
func (c Config) ConstellationLevels() []float32 {
	levels := c.levelsPerDimension()
	if levels == 0 {
		return nil
	}
	scale := c.ConstellationNormalization()
	result := make([]float32, levels)
	for i := range levels {
		centered := float32(-(levels - 1) + i*2)
		result[i] = centered * scale
	}
	return result
}

// ConstellationNormalization returns the normalization gain so the average symbol power equals 1.
func (c Config) ConstellationNormalization() float32 {
	l := float32(c.levelsPerDimension())
	if l <= 1 {
		return 1
	}
	return math32.Sqrt(3 / (2 * (l*l - 1)))
}

// RequiredBandwidth approximates the occupied bandwidth for the current parameters.
func (c Config) RequiredBandwidth() float32 {
	return c.SymbolRate() * (1 + c.RaisedCosineBeta)
}

// SymbolSeparationMetric returns the Euclidean separation between adjacent constellation points.
func (c Config) SymbolSeparationMetric() float32 {
	axisStep := c.ConstellationNormalization() * 2
	return axisStep
}

var qamEbN0Table = map[int]float32{
	2: 9.6,  // 4-QAM / QPSK
	4: 15.0, // 16-QAM
	6: 21.0, // 64-QAM
	8: 27.0, // 256-QAM
}

// MinimumEbN0 returns the approximate theoretical Eb/N0 (dB) for the target BER.
func (c Config) MinimumEbN0(targetBER float32) float32 {
	if targetBER <= 0 {
		targetBER = 1e-5
	}
	base, ok := qamEbN0Table[c.BitsPerSymbol]
	if !ok {
		// Fall back to linear extrapolation (roughly +3 dB per 2 extra bits)
		base = 9.6 + 3*float32(c.BitsPerSymbol-2)/2
	}
	if targetBER != 1e-5 {
		scale := math32.Log10(1e-5 / targetBER)
		base += 5 * scale
	}
	return base
}

// MinimumSNR returns the approximate minimum SNR (dB) for the given BER requirement and margin.
func (c Config) MinimumSNR(targetBER, marginDb float32) float32 {
	if marginDb <= 0 {
		marginDb = 2.5
	}
	ebn0 := c.MinimumEbN0(targetBER)
	return ebn0 + 10*math32.Log10(float32(c.BitsPerSymbol)) + marginDb
}

// IsSNRSufficient reports whether snrDb meets the requirement for the given BER and margin.
func (c Config) IsSNRSufficient(snrDb, targetBER, marginDb float32) bool {
	return snrDb >= c.MinimumSNR(targetBER, marginDb)
}

// EnergyThresholdForSNR calculates the minimum energy threshold to distinguish signal+noise from noise-only.
// At SNR dB, signal power is 10^(SNR/10) times noise power.
// We want a threshold that:
// - Is above noise-only power (to reject padding noise)
// - Is below signal+noise power (to detect actual signal)
// Returns threshold in energy units (average power over symbol period).
// For QAM, we use average symbol power from the constellation normalization.
func (c Config) EnergyThresholdForSNR(snrDb float32) float32 {
	// Expected signal power (for QAM with normalized constellation)
	// Average symbol power is normalized to 1.0 by ConstellationNormalization()
	// But we need to account for the actual amplitude scaling
	signalPower := c.Amplitude * c.Amplitude
	
	// Noise power at given SNR
	// SNR = 10*log10(signal_power / noise_power)
	// noise_power = signal_power / 10^(SNR/10)
	noisePower := signalPower / math32.Pow(10, snrDb/10)
	
	// Threshold should be between noise-only and signal+noise
	// Use noise_power * factor, where factor accounts for:
	// - Noise variance (use 3-5 sigma for reliable detection)
	// - Margin for noise fluctuations
	// At 25dB SNR: noise_power = signal_power / 316 ≈ 0.002
	// Threshold = noise_power * 5 = 0.01 (well above noise, well below signal)
	// This ensures we reject noise-only padding but detect signal+noise
	factor := float32(5.0) // 5-sigma threshold for reliable detection
	threshold := noisePower * factor
	
	// Ensure threshold is reasonable (not too high, not too low)
	// Minimum: 2x noise power (to reject pure noise)
	// Maximum: 0.1x signal power (to detect weak signals)
	minThreshold := noisePower * 2.0
	maxThreshold := signalPower * 0.1
	if threshold < minThreshold {
		threshold = minThreshold
	}
	if threshold > maxThreshold {
		threshold = maxThreshold
	}
	
	return threshold
}

func (c *Config) ensureWithinNyquist() error {
	bw := c.RequiredBandwidth()
	half := c.SampleRate / 2
	low := c.CarrierHz - bw/2
	high := c.CarrierHz + bw/2
	if low < 0 || high > half {
		return fmt.Errorf("qam: carrier %.2fHz with bandwidth %.2fHz exceeds Nyquist %.2fHz", c.CarrierHz, bw, half)
	}
	return nil
}
