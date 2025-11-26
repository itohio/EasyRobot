package mfsk

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
)

// Config holds all tunable parameters for the MFSK modem.
type Config struct {
	SampleRate       float32
	CarrierHz        float32
	ToneSpacingHz    float32
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
		ToneSpacingHz:    0,
		BitsPerSymbol:    2,
		SamplesPerSymbol: 384,
		Amplitude:        0.8,
		Envelope:         "raised_cosine",
		RaisedCosineBeta: 0.35,
		PreambleLength:   16,
		EnergyThreshold:  1e-3,
		TransmissionGap:  4,
		LoopBandwidth:    200,
		LoopDamping:      0.707,
		SilenceSamples:   0,
	}
}

// Validate checks the configuration and applies defaults.
func (c *Config) Validate() error {
	if c.SampleRate <= 0 {
		return errors.New("mfsk: sample rate must be positive")
	}
	if c.CarrierHz <= 0 {
		c.CarrierHz = 1_000
	}
	if c.ToneSpacingHz <= 0 {
		c.ToneSpacingHz = c.SymbolRate()
	}
	if c.BitsPerSymbol <= 0 || c.BitsPerSymbol > 8 {
		return fmt.Errorf("mfsk: bits per symbol must be within 1..8, got %d", c.BitsPerSymbol)
	}
	if c.SamplesPerSymbol <= 0 {
		return errors.New("mfsk: samples per symbol must be positive")
	}
	if c.Amplitude <= 0 {
		c.Amplitude = 1
	}
	if c.PreambleLength < 0 {
		c.PreambleLength = 0
	}
	if c.RaisedCosineBeta <= 0 {
		c.RaisedCosineBeta = 0.35
	}
	if c.LoopBandwidth <= 0 {
		c.LoopBandwidth = c.SymbolRate() * 0.05
	}
	if c.LoopDamping <= 0 {
		c.LoopDamping = 0.707
	}
	if err := c.ensureSpacingWithinNyquist(); err != nil {
		return err
	}
	c.ToneSpacingHz = c.SymbolRate()
	return nil
}

// toneCount returns the number of frequency tones (M) for M-FSK.
// For M-FSK: M = 2^BitsPerSymbol
// Examples: 1 bit → 2-FSK (2 tones), 2 bits → 4-FSK (4 tones), 3 bits → 8-FSK (8 tones)
func (c Config) toneCount() int {
	return 1 << c.BitsPerSymbol
}

// MFskOrder returns the M-FSK order (number of frequencies).
// This is the same as toneCount() but with a clearer name.
func (c Config) MFskOrder() int {
	return c.toneCount()
}

func (c Config) maxToneHz() float32 {
	return c.CarrierHz + c.ToneSpacingHz*float32(c.toneCount()-1)
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
// This includes the tone spacing range plus the raised-cosine rolloff.
func (c Config) RequiredBandwidth() float32 {
	toneBandwidth := c.ToneSpacingHz * float32(c.toneCount()-1)
	rolloffBandwidth := c.SymbolRate() * c.RaisedCosineBeta
	return toneBandwidth + rolloffBandwidth
}

// ToneBandwidth returns the bandwidth occupied by just the tones (without rolloff).
func (c Config) ToneBandwidth() float32 {
	if c.toneCount() <= 1 {
		return 0
	}
	return c.ToneSpacingHz * float32(c.toneCount()-1)
}

// SymbolSeparationMetric returns the frequency separation between adjacent tones in Hz.
// For orthogonal M-FSK, this should equal the symbol rate.
func (c Config) SymbolSeparationMetric() float32 {
	return c.ToneSpacingHz
}

// SymbolSeparationHz returns the frequency separation between adjacent tones in Hz.
// This is the same as ToneSpacingHz but with a clearer name for separation calculations.
func (c Config) SymbolSeparationHz() float32 {
	return c.ToneSpacingHz
}

// TotalBandwidth returns the total bandwidth required for all tones.
// This is the frequency range from the lowest to highest tone.
func (c Config) TotalBandwidth() float32 {
	if c.toneCount() <= 1 {
		return 0
	}
	return c.ToneSpacingHz * float32(c.toneCount()-1)
}

// BandwidthInfo returns detailed bandwidth information for debugging/analysis.
func (c Config) BandwidthInfo() (toneBandwidth, totalBandwidth, symbolSeparation, lowestFreq, highestFreq float32) {
	toneBandwidth = c.ToneBandwidth()
	totalBandwidth = c.RequiredBandwidth()
	symbolSeparation = c.SymbolSeparationHz()
	lowestFreq = c.CarrierHz
	highestFreq = c.CarrierHz + toneBandwidth
	return
}

// ToneFrequency returns the absolute frequency (Hz) for the provided tone index.
func (c Config) ToneFrequency(idx int) float32 {
	return c.CarrierHz + float32(idx)*c.ToneSpacingHz
}

// MinimumEbN0 returns the theoretical minimum Eb/N0 (dB) required for coherent M-FSK
// at a given BER, without implementation margin.
// Based on: P_e ≈ (M-1) * Q(sqrt(2*E_b/N_0 * log2(M)))
// For BER = 1e-5, we approximate the required Eb/N0.
func (c Config) MinimumEbN0(targetBER float32) float32 {
	bitsPerSymbol := float32(c.BitsPerSymbol)

	// Approximate Eb/N0 for coherent M-FSK at BER = 1e-5
	// For M=2: ~12.5 dB, M=4: ~14.5 dB, M=8: ~16.5 dB, M=16: ~18.5 dB
	// Formula approximation: Eb/N0 ≈ 10*log10(2*log2(M)) + baseEbN0
	baseEbN0 := float32(9.5) // Base for 2-FSK at BER=1e-5
	ebN0Linear := baseEbN0 + 10*math32.Log10(2*bitsPerSymbol)

	// Adjust for higher BER targets (less strict)
	if targetBER > 1e-5 {
		ebN0Linear -= 10 * math32.Log10(targetBER/1e-5) * 0.5
	} else if targetBER < 1e-5 {
		ebN0Linear += 10 * math32.Log10(1e-5/targetBER) * 0.5
	}

	return ebN0Linear
}

// MinimumSNR returns the theoretical minimum SNR (dB) required for coherent M-FSK
// at a given BER, including a safety margin for implementation losses.
// For our test setup, SNR is measured as signal power / noise power in the same bandwidth.
// marginDb: safety margin in dB (typically 2-3 dB)
func (c Config) MinimumSNR(targetBER float32, marginDb float32) float32 {
	if marginDb < 0 {
		marginDb = 2.5 // Default margin
	}
	ebN0Db := c.MinimumEbN0(targetBER)

	// For M-FSK with our test setup (power-based SNR in same bandwidth):
	// SNR (dB) ≈ Eb/N0 (dB) - 10*log10(BitsPerSymbol) + processing gain
	// The processing gain from symbol integration is already accounted in Eb/N0 calculation
	// Simplified: SNR ≈ Eb/N0 - 10*log10(BitsPerSymbol) for power-based SNR
	bitsPerSymbol := float32(c.BitsPerSymbol)
	snrDb := ebN0Db - 10*math32.Log10(bitsPerSymbol) + marginDb

	return snrDb
}

// IsSNRSufficient checks if the provided SNR (dB) is sufficient for reliable operation
// at the given target BER, including implementation margin.
func (c Config) IsSNRSufficient(snrDb float32, targetBER float32, marginDb float32) bool {
	minSNR := c.MinimumSNR(targetBER, marginDb)
	return snrDb >= minSNR
}

// ProcessingGain returns the processing gain in dB for coherent M-FSK.
// This is the improvement in SNR due to symbol-level integration.
func (c Config) ProcessingGain() float32 {
	// Processing gain ≈ 10*log10(SamplesPerSymbol)
	return 10 * math32.Log10(float32(c.SamplesPerSymbol))
}

// EnergyThresholdForSNR calculates the minimum energy threshold to distinguish signal+noise from noise-only.
// At SNR dB, signal power is 10^(SNR/10) times noise power.
// For M-FSK, signal power is based on the tone amplitude.
// Returns threshold in energy units (average power over symbol period).
func (c Config) EnergyThresholdForSNR(snrDb float32) float32 {
	// Expected signal power (for M-FSK, power is amplitude squared)
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

func (c *Config) ensureSpacingWithinNyquist() error {
	// Calculate required bandwidth for all tones
	toneCount := c.toneCount()
	if toneCount <= 1 {
		return errors.New("mfsk: must have at least 2 tones")
	}

	// Start with current or default spacing
	if c.ToneSpacingHz <= 0 {
		c.ToneSpacingHz = c.SymbolRate()
	}

	// Calculate total bandwidth needed
	totalBandwidth := c.ToneSpacingHz * float32(toneCount-1)

	// Check if we can fit within Nyquist
	nyquistFreq := c.SampleRate / 2
	maxAvailableBandwidth := nyquistFreq - c.CarrierHz

	if maxAvailableBandwidth < totalBandwidth {
		// Need to either:
		// 1. Increase SamplesPerSymbol to reduce symbol rate (and thus tone spacing)
		// 2. Adjust carrier frequency lower
		// 3. Both

		// Calculate minimum SamplesPerSymbol needed
		minSamplesPerSymbol := int(math32.Ceil(float32(toneCount-1) * c.SampleRate / maxAvailableBandwidth))
		if minSamplesPerSymbol < 2 {
			minSamplesPerSymbol = 2
		}
		// Round up to even number for better FFT alignment
		if minSamplesPerSymbol&1 == 1 {
			minSamplesPerSymbol++
		}

		// Update SamplesPerSymbol if needed
		if c.SamplesPerSymbol < minSamplesPerSymbol {
			c.SamplesPerSymbol = minSamplesPerSymbol
		}

		// Recalculate spacing based on symbol rate
		c.ToneSpacingHz = c.SymbolRate()
		totalBandwidth = c.ToneSpacingHz * float32(toneCount-1)

		// If still too wide, adjust carrier frequency
		if c.CarrierHz+totalBandwidth > nyquistFreq {
			// Move carrier lower to fit
			maxCarrier := nyquistFreq - totalBandwidth
			if maxCarrier < 0 {
				return fmt.Errorf("mfsk: bandwidth %.2fHz too wide for sample rate %.2fHz (need %.2fHz, have %.2fHz)",
					totalBandwidth, c.SampleRate, totalBandwidth, nyquistFreq)
			}
			c.CarrierHz = maxCarrier
		}
	}

	// Align carrier to nearest bin for better frequency resolution
	baseBin := int(math32.Round(float32(c.SamplesPerSymbol) * c.CarrierHz / c.SampleRate))
	if baseBin < 0 {
		baseBin = 0
	}
	c.CarrierHz = float32(baseBin) * c.SampleRate / float32(c.SamplesPerSymbol)

	// Final check
	highest := c.CarrierHz + c.ToneSpacingHz*float32(toneCount-1)
	if highest > nyquistFreq {
		return fmt.Errorf("mfsk: highest tone %.2fHz exceeds Nyquist %.2fHz", highest, nyquistFreq)
	}

	return nil
}
