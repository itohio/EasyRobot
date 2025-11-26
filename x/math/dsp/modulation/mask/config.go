package mask

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
)

// Config holds all tunable parameters for the M-ASK modem.
type Config struct {
	SampleRate       float32
	CarrierHz        float32
	BitsPerSymbol    int
	SamplesPerSymbol int
	Amplitude        float32

	// Amplitude levels (pre-computed)
	AmplitudeLevels []float32 // A_0, A_1, ..., A_{M-1}
	LevelSpacing     string    // "even" or "power_efficient"
	MinAmplitude     float32   // For power-efficient (non-zero minimum)
	Depth            float32   // Depth parameter (0-100%): 100% means lowest is 0, lower means non-zero minimum

	// Pulse shaping
	Envelope         string
	RaisedCosineBeta float32

	// Preamble
	PreambleLength  int
	PreamblePattern []int // Alternating pattern for sync

	// Demodulator parameters
	EnergyThreshold    float32
	TransmissionGap    int   // Silence detection gap (at least 4 symbols)
	LockLossThreshold  int   // Consecutive zero symbols to lose lock (default 20)
	AGCWindow          int
	CoherentMode       bool // true for coherent, false for envelope detection

	// Carrier recovery (for coherent mode)
	LoopBandwidth float32
	LoopDamping   float32

	// Timing recovery
	TimingLoopBandwidth float32
}

// DefaultConfig returns a safe baseline configuration.
func DefaultConfig() Config {
	return Config{
		SampleRate:       22_000,
		CarrierHz:        10_000,
		BitsPerSymbol:    2,
		SamplesPerSymbol: 220,
		Amplitude:        0.8,
		LevelSpacing:     "even",
		MinAmplitude:     0.1,
		Depth:            100.0, // 100% means lowest is 0 (full depth)
		Envelope:         "raised_cosine",
		RaisedCosineBeta: 0.35,
		PreambleLength:   20, // Default preamble length (lowest, highest alternating) - required for normalization
		EnergyThreshold:   1e-3,
		TransmissionGap:   4,  // At least 4 symbols of silence required
		LockLossThreshold: 20, // Lose lock after 20 consecutive zero symbols
		AGCWindow:        32,
		CoherentMode:     false, // Start with envelope detection
		LoopBandwidth:    0,     // Will be set in Validate()
		LoopDamping:      0.707,
	}
}

// Validate checks the configuration and applies defaults.
func (c *Config) Validate() error {
	if c.SampleRate <= 0 {
		return errors.New("mask: sample rate must be positive")
	}
	if c.CarrierHz <= 0 {
		c.CarrierHz = 1_000
	}
	if c.BitsPerSymbol <= 0 || c.BitsPerSymbol > 8 {
		return fmt.Errorf("mask: bits per symbol must be within 1..8, got %d", c.BitsPerSymbol)
	}
	if c.SamplesPerSymbol <= 0 {
		return errors.New("mask: samples per symbol must be positive")
	}
	if c.Amplitude <= 0 {
		c.Amplitude = 1
	}
	if c.Depth <= 0 || c.Depth > 100 {
		c.Depth = 100.0 // Default: 100% depth (lowest is 0)
	}
	// Preamble is required for amplitude normalization - cannot be disabled
	if c.PreambleLength <= 0 {
		c.PreambleLength = 20 // Default: 20 symbols (lowest/highest alternating)
	}
	if c.TransmissionGap < 4 {
		c.TransmissionGap = 4 // Minimum 4 symbols for silence detection
	}
	if c.LockLossThreshold <= 0 {
		c.LockLossThreshold = 20 // Default: lose lock after 20 consecutive zero symbols
	}
	if c.RaisedCosineBeta <= 0 {
		c.RaisedCosineBeta = 0.35
	}
	if c.EnergyThreshold <= 0 {
		c.EnergyThreshold = 1e-3
	}
	if c.TransmissionGap <= 0 {
		c.TransmissionGap = 4
	}
	if c.AGCWindow <= 0 {
		c.AGCWindow = 32
	}
	if c.LoopBandwidth <= 0 {
		c.LoopBandwidth = c.SymbolRate() * 0.05
	}
	if c.LoopDamping <= 0 {
		c.LoopDamping = 0.707
	}
	if c.TimingLoopBandwidth <= 0 {
		c.TimingLoopBandwidth = c.SymbolRate() * 0.01
	}

	// Calculate amplitude levels
	c.AmplitudeLevels = c.calculateAmplitudeLevels()

	// Set default preamble pattern if not set and preamble is enabled
	if c.PreambleLength > 0 && len(c.PreamblePattern) == 0 {
		M := c.constellationSize()
		// Preamble pattern: highest symbol, lowest symbol (alternating)
		c.PreamblePattern = []int{M - 1, 0} // Alternating highest/lowest
	}

	if err := c.ensureWithinNyquist(); err != nil {
		return err
	}
	return nil
}

func (c Config) constellationSize() int {
	return 1 << c.BitsPerSymbol
}

// SymbolRate returns the nominal symbol rate in Hz.
func (c Config) SymbolRate() float32 {
	return c.SampleRate / float32(c.SamplesPerSymbol)
}

// SamplesForBits returns the number of samples required to emit n bits including preamble.
func (c Config) SamplesForBits(bits int) int {
	if bits <= 0 {
		return c.PreambleLength * c.SamplesPerSymbol
	}
	symbols := (bits + c.BitsPerSymbol - 1) / c.BitsPerSymbol
	return (symbols + c.PreambleLength) * c.SamplesPerSymbol
}

// RequiredBandwidth approximates the occupied bandwidth for the current parameters.
func (c Config) RequiredBandwidth() float32 {
	switch c.Envelope {
	case "rectangular", "":
		return c.SymbolRate() / 2.0
	case "raised_cosine", "root_raised_cosine":
		return c.SymbolRate() * (1 + c.RaisedCosineBeta) / 2.0
	default:
		return c.SymbolRate() / 2.0
	}
}

// SymbolSeparationMetric returns the amplitude separation between adjacent symbols.
func (c Config) SymbolSeparationMetric() float32 {
	if len(c.AmplitudeLevels) < 2 {
		return 0
	}
	// For evenly spaced: separation = A_max / (M-1)
	if c.LevelSpacing == "even" {
		return c.Amplitude / float32(c.constellationSize()-1)
	}
	// For power-efficient: separation = (A_max - A_min) / (M-1)
	return (c.Amplitude - c.MinAmplitude) / float32(c.constellationSize()-1)
}

// MinimumEbN0 returns the theoretical minimum Eb/N0 (dB) for coherent M-ASK.
func (c Config) MinimumEbN0(targetBER float32) float32 {
	if targetBER <= 0 {
		targetBER = 1e-5
	}
	// Approximate coherent M-ASK requirement around BER=1e-5
	// Base for 2-ASK (OOK) ≈ 9.5 dB
	base := float32(9.5)
	bits := float32(c.BitsPerSymbol)
	if bits < 1 {
		bits = 1
	}
	// Add 10*log10(2*log2(M)) for higher-order ASK
	ebn0 := base + 10*math32.Log10(2*bits)
	if targetBER != 1e-5 {
		scale := math32.Log10(1e-5 / targetBER)
		ebn0 += 5 * scale
	}
	return ebn0
}

// MinimumSNR returns the theoretical minimum SNR (dB) required for the given BER and margin.
func (c Config) MinimumSNR(targetBER, marginDb float32) float32 {
	if marginDb <= 0 {
		marginDb = 2.5
	}
	ebn0 := c.MinimumEbN0(targetBER)
	// For non-coherent (envelope) detection, add ~3dB penalty
	if !c.CoherentMode {
		ebn0 += 3.0
	}
	// SNR ≈ Eb/N0 - 10*log10(BitsPerSymbol) + margin
	snr := ebn0 - 10*math32.Log10(float32(c.BitsPerSymbol)) + marginDb
	return snr
}

// IsSNRSufficient checks if the provided SNR (dB) meets the theoretical requirement.
func (c Config) IsSNRSufficient(snrDb, targetBER, marginDb float32) bool {
	return snrDb >= c.MinimumSNR(targetBER, marginDb)
}

// EnergyThresholdForSNR calculates the minimum energy threshold to distinguish signal+noise from noise-only.
// At SNR dB, signal power is 10^(SNR/10) times noise power.
// For M-ASK, signal power is based on the amplitude.
// Returns threshold in energy units (average power over symbol period).
// Returns 0 for infinite SNR (noise-free signals).
func (c Config) EnergyThresholdForSNR(snrDb float32) float32 {
	// For infinite SNR (noise-free), return 0 to use fixed threshold fallback
	if math32.IsInf(snrDb, 1) || snrDb <= 0 {
		return 0
	}

	// Expected signal power (for M-ASK, power is amplitude squared)
	signalPower := c.Amplitude * c.Amplitude

	// Noise power at given SNR
	// SNR = 10*log10(signal_power / noise_power)
	// noise_power = signal_power / 10^(SNR/10)
	noisePower := signalPower / math32.Pow(10, snrDb/10)

	// Threshold should be between noise-only and signal+noise
	// Use noise_power * factor for reliable detection (5-sigma threshold)
	factor := float32(5.0)
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

func (c *Config) calculateAmplitudeLevels() []float32 {
	M := c.constellationSize()
	levels := make([]float32, M)

	// Calculate minimum amplitude based on depth parameter
	// Depth 100% = lowest is 0, Depth 0% = lowest equals highest (no depth)
	depthFactor := c.Depth / 100.0
	minAmp := c.Amplitude * (1.0 - depthFactor)

	if c.LevelSpacing == "power_efficient" {
		// Use provided MinAmplitude if set, otherwise use calculated minAmp
		if c.MinAmplitude > 0 {
			minAmp = c.MinAmplitude
		}
		// A_m = A_min + m * (A_max - A_min) / (M-1)
		delta := (c.Amplitude - minAmp) / float32(M-1)
		for m := range M {
			levels[m] = minAmp + float32(m)*delta
		}
	} else {
		// Evenly spaced: A_m = A_min + m * (A_max - A_min) / (M-1)
		if M == 1 {
			levels[0] = c.Amplitude
		} else {
			delta := (c.Amplitude - minAmp) / float32(M-1)
			for m := range M {
				levels[m] = minAmp + float32(m)*delta
			}
		}
	}

	return levels
}

func (c *Config) ensureWithinNyquist() error {
	bw := c.RequiredBandwidth()
	half := c.SampleRate / 2
	low := c.CarrierHz - bw/2
	high := c.CarrierHz + bw/2
	if low < 0 || high > half {
		return fmt.Errorf("mask: carrier %.2fHz with bandwidth %.2fHz exceeds Nyquist %.2fHz", c.CarrierHz, bw, half)
	}
	return nil
}

