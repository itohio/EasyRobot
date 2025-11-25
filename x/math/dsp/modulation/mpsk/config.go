package mpsk

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
)

// Config holds all tunable parameters for the MPSK modem.
type Config struct {
	SampleRate       float32
	CarrierHz        float32
	BitsPerSymbol    int
	SamplesPerSymbol int
	Amplitude        float32

	Envelope           string
	RaisedCosineBeta   float32
	PreambleLength     int
	EnergyThreshold    float32
	TransmissionGap    int
	LoopBandwidth      float32
	LoopDamping        float32
	LoopTrackingFactor float32
	SilenceSamples     int
}

// DefaultConfig returns a safe baseline configuration.
func DefaultConfig() Config {
	return Config{
		SampleRate:         96_000,
		CarrierHz:          20_000,
		BitsPerSymbol:      2,
		SamplesPerSymbol:   384,
		Amplitude:          0.8,
		Envelope:           "raised_cosine",
		RaisedCosineBeta:   0.35,
		PreambleLength:     20, // Longer preamble for better PLL convergence in noise
		EnergyThreshold:    1e-5,
		TransmissionGap:    4,
		LoopBandwidth:      0, // Will be set to SymbolRate() * 0.05 in Validate()
		LoopDamping:        0.707,
		LoopTrackingFactor: 0.05,
		SilenceSamples:     0,
	}
}

// Validate checks the configuration and applies defaults.
func (c *Config) Validate() error {
	if c.SampleRate <= 0 {
		return errors.New("mpsk: sample rate must be positive")
	}
	if c.CarrierHz <= 0 {
		c.CarrierHz = 1_000
	}
	if c.BitsPerSymbol <= 0 || c.BitsPerSymbol > 8 {
		return fmt.Errorf("mpsk: bits per symbol must be within 1..8, got %d", c.BitsPerSymbol)
	}
	if c.SamplesPerSymbol <= 0 {
		return errors.New("mpsk: samples per symbol must be positive")
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
	if c.LoopTrackingFactor <= 0 || c.LoopTrackingFactor >= 1 {
		c.LoopTrackingFactor = 0.05
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

// OptimalLoopBandwidth returns the optimal loop bandwidth for a given target SNR (in dB).
// Based on empirical testing, this selects bandwidth that balances convergence speed
// with noise rejection. The bandwidth is adjusted based on preamble length:
// - Longer preambles allow wider bandwidth for faster acquisition
// - Shorter preambles require narrower bandwidth for better noise filtering
func (c Config) OptimalLoopBandwidth(targetSNRDb float32) float32 {
	symbolRate := c.SymbolRate()
	
	// Base bandwidth from SNR table (for 25dB: 0.020 = 2% of symbol rate)
	var baseBw float32
	switch {
	case targetSNRDb >= 25:
		baseBw = 0.020 // 2% - optimal for 25-30dB
	case targetSNRDb >= 20:
		baseBw = 0.050 // 5% - for 20-25dB
	case targetSNRDb >= 15:
		baseBw = 0.020 // 2% - for 15-20dB
	case targetSNRDb >= 10:
		baseBw = 0.010 // 1% - for 10-15dB
	case targetSNRDb >= 8:
		baseBw = 0.010 // 1% - for 8-10dB
	default: // 3-7 dB
		baseBw = 0.005 // 0.5% - for very low SNR
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
func (c Config) OptimalAcquisitionBandwidth(targetSNRDb float32) float32 {
	// Use wider bandwidth for acquisition (1.5-2x tracking bandwidth)
	// but cap at reasonable maximum
	acqBw := c.OptimalLoopBandwidth(targetSNRDb) * 1.5
	maxBw := c.SymbolRate() * 0.1 // Cap at 10% of symbol rate
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
	return c.SymbolRate() * (1 + c.RaisedCosineBeta)
}

// SymbolSeparationMetric returns angular separation between adjacent constellation points.
func (c Config) SymbolSeparationMetric() float32 {
	return twoPi / float32(c.constellationSize())
}

// MinimumEbN0 returns the theoretical minimum Eb/N0 (dB) for coherent M-PSK.
func (c Config) MinimumEbN0(targetBER float32) float32 {
	if targetBER <= 0 {
		targetBER = 1e-5
	}
	// Approximate coherent PSK requirement around BER=1e-5.
	// Base for BPSK ≈ 9.6 dB. Add 10*log10(BitsPerSymbol) for higher-order PSK.
	base := float32(9.6)
	bits := float32(c.BitsPerSymbol)
	if bits < 1 {
		bits = 1
	}
	ebn0 := base + 10*math32.Log10(bits)
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
	// SNR ≈ Eb/N0 + 10*log10(BitsPerSymbol) for PSK since bandwidth equals bit rate.
	snr := ebn0 + 10*math32.Log10(float32(c.BitsPerSymbol))
	return snr + marginDb
}

// IsSNRSufficient checks if the provided SNR (dB) meets the theoretical requirement.
func (c Config) IsSNRSufficient(snrDb, targetBER, marginDb float32) bool {
	return snrDb >= c.MinimumSNR(targetBER, marginDb)
}

func (c *Config) ensureWithinNyquist() error {
	bw := c.RequiredBandwidth()
	half := c.SampleRate / 2
	low := c.CarrierHz - bw/2
	high := c.CarrierHz + bw/2
	if low < 0 || high > half {
		return fmt.Errorf("mpsk: carrier %.2fHz with bandwidth %.2fHz exceeds Nyquist %.2fHz", c.CarrierHz, bw, half)
	}
	return nil
}
