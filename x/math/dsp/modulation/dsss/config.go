package dsss

import (
	"github.com/chewxy/math32"
)

// Config holds configuration parameters for DSSS modulation/demodulation
type Config struct {
	// Modulation parameters
	BitsPerSymbol   int     // k: 1, 2, 3, etc. (log2(M))
	ModulationOrder int     // M: 2, 4, 8, etc. (M = 2^k)

	// Spreading codes
	Codes          [][]int8 // M codes, each of length L (bipolar: -1, +1)
	SpreadingFactor int     // L: code length (8 for tests)

	// Timing parameters
	SampleRate    float32 // f_s: 96000 Hz
	SymbolRate    float32 // R_s: symbols per second
	CarrierFreq   float32 // f_c: 20000 Hz
	SamplesPerChip int     // N_sc = ⌊f_s / (R_s × L)⌋

	// Pulse shaping
	PulseShape     string  // "rectangular", "raised_cosine", "root_raised_cosine"
	Rolloff        float32 // For raised cosine (0.0-1.0), typically 0.35-0.5
	PulseFilterSpan int    // Number of chip periods for pulse shape (typically 4-8)

	// Preamble
	PreambleLength int    // Number of preamble symbols
	PreamblePattern []int // Symbol sequence for preamble (alternating for sync)

	// Transmission detection
	EnergyThreshold   float32 // Energy threshold for transmission detection
	DetectionWindow   int     // Samples for energy calculation
	TransmissionTimeout int   // Symbols without signal before declaring transmission end

	// Amplitude
	Amplitude float32 // Signal amplitude (typically 1.0)
}

// CalculateRequiredBandwidth calculates the minimum bandwidth required for transmission
func (c *Config) CalculateRequiredBandwidth() float32 {
	chipRate := c.SymbolRate * float32(c.SpreadingFactor)

	switch c.PulseShape {
	case "rectangular":
		// Null bandwidth: B = R_c / 2 (minimum)
		return chipRate / 2.0
	case "raised_cosine", "root_raised_cosine":
		// Null bandwidth: B = (1 + α) × R_c / 2
		return (1.0 + c.Rolloff) * chipRate / 2.0
	default:
		return chipRate / 2.0 // Conservative estimate
	}
}

// CalculateMinimumSampleRate calculates minimum sample rate for given carrier
func (c *Config) CalculateMinimumSampleRate() float32 {
	bandwidth := c.CalculateRequiredBandwidth()
	// Nyquist: f_s ≥ 2(f_c + B/2)
	// For safety: f_s ≥ 4f_c (if carrier >> bandwidth)
	// Or: f_s ≥ 2(f_c + B/2)
	return max(4.0*c.CarrierFreq, 2.0*(c.CarrierFreq+bandwidth/2.0))
}

// CalculateSamplesPerChip calculates samples per chip for given sample rate
func (c *Config) CalculateSamplesPerChip(sampleRate float32) int {
	chipRate := c.SymbolRate * float32(c.SpreadingFactor)
	return int(sampleRate / chipRate)
}

// ValidateSampleRate validates that sample rate meets requirements
func (c *Config) ValidateSampleRate(sampleRate float32) error {
	minRate := c.CalculateMinimumSampleRate()
	if sampleRate < minRate {
		return &ConfigError{Message: "sample rate is below minimum"}
	}
	return nil
}

// CalculateProcessingGain calculates the processing gain in linear and dB scale
func (c *Config) CalculateProcessingGain() (linear float32, dB float32) {
	linear = float32(c.SpreadingFactor)
	dB = 10.0 * math32.Log10(linear)
	return linear, dB
}

// CalculateOutputSNR calculates output SNR after despreading from input SNR
func (c *Config) CalculateOutputSNR(inputSNRdB float32) float32 {
	// SNR_out[dB] = SNR_in[dB] + 10*log10(L)
	_, processingGainDB := c.CalculateProcessingGain()
	return inputSNRdB + processingGainDB
}

// CalculateInputSNR calculates required input SNR to achieve target output SNR
func (c *Config) CalculateInputSNR(targetOutputSNRdB float32) float32 {
	// SNR_in[dB] = SNR_out[dB] - 10*log10(L)
	_, processingGainDB := c.CalculateProcessingGain()
	return targetOutputSNRdB - processingGainDB
}

// CalculateMinimumSNR calculates minimum input SNR required for reliable demodulation
func (c *Config) CalculateMinimumSNR(targetBER float32) float32 {
	bitsPerSymbol := float32(c.BitsPerSymbol)

	// Base E_b/N_0 requirement (in dB)
	// Binary (k=1): ~10dB for BER 10^-5
	// Higher order: add margin based on modulation order
	baseEbN0 := 10.0 + 3.0*(bitsPerSymbol-1.0) // Add 3dB per extra bit

	// Convert to output SNR
	outputSNR := baseEbN0 + 10.0*math32.Log10(bitsPerSymbol)

	// Convert to input SNR
	return c.CalculateInputSNR(outputSNR)
}

// CalculateMaximumAchievableSNR calculates maximum achievable output SNR
func (c *Config) CalculateMaximumAchievableSNR(inputSNRdB float32) float32 {
	return c.CalculateOutputSNR(inputSNRdB)
}

// CalculateEbN0 calculates bit energy to noise ratio from output SNR
func (c *Config) CalculateEbN0(outputSNRdB float32, symbolRate float32) float32 {
	bitsPerSymbol := float32(c.BitsPerSymbol)
	symbolPeriod := 1.0 / symbolRate

	ebN0Linear := math32.Pow(10.0, outputSNRdB/10.0) * symbolPeriod / bitsPerSymbol
	return 10.0 * math32.Log10(ebN0Linear)
}

// EstimateBER estimates bit error rate from output SNR
func (c *Config) EstimateBER(outputSNRdB float32, symbolRate float32) float32 {
	ebN0dB := c.CalculateEbN0(outputSNRdB, symbolRate)
	ebN0Linear := math32.Pow(10.0, ebN0dB/10.0)

	// Approximate Q function for BPSK
	sqrtEbN0 := math32.Sqrt(2.0 * ebN0Linear)

	// Use erfc approximation: erfc(x) ≈ exp(-x²) / (x*√π) for large x
	if sqrtEbN0 > 3.0 {
		// Large argument approximation
		return 0.5 * math32.Exp(-sqrtEbN0*sqrtEbN0/2.0) / (sqrtEbN0 * 1.772) // 1.772 ≈ √π
	}
	// Small argument: use exponential approximation
	return 0.5 * math32.Exp(-ebN0Linear)
}

// CalculateSNRLimit calculates the SNR limit for reliable communication
func (c *Config) CalculateSNRLimit(targetBER float32) float32 {
	return c.CalculateMinimumSNR(targetBER)
}

// IsSNRSufficient checks if given input SNR is sufficient for reliable demodulation
func (c *Config) IsSNRSufficient(inputSNRdB float32, targetBER float32) bool {
	minSNR := c.CalculateSNRLimit(targetBER)
	return inputSNRdB >= minSNR
}

// CalculateCodeCorrelation calculates normalized correlation between two codes
func (c *Config) CalculateCodeCorrelation(codeI, codeJ []int8) float32 {
	if len(codeI) != len(codeJ) || len(codeI) != c.SpreadingFactor {
		return 0.0 // Invalid codes
	}

	var sum float32
	L := float32(c.SpreadingFactor)

	for k := range c.SpreadingFactor {
		sum += float32(codeI[k] * codeJ[k])
	}

	return sum / L
}

// CalculateMaximumCrossCorrelation calculates maximum cross-correlation between any two different codes
func (c *Config) CalculateMaximumCrossCorrelation() float32 {
	if len(c.Codes) < 2 {
		return 1.0 // Single code or no codes
	}

	maxCorr := float32(0.0)

	// Check all pairs of different codes
	for i := range len(c.Codes) {
		for j := i + 1; j < len(c.Codes); j++ {
			corr := math32.Abs(c.CalculateCodeCorrelation(c.Codes[i], c.Codes[j]))
			if corr > maxCorr {
				maxCorr = corr
			}
		}
	}

	return maxCorr
}

// CalculateMinimumCodeSeparation calculates minimum separation between codes
func (c *Config) CalculateMinimumCodeSeparation() float32 {
	if len(c.Codes) < 2 {
		return 2.0 // Binary case: separation is 2 (between -1 and +1)
	}

	minSep := float32(2.0) // Maximum possible separation

	// For each pair of codes, calculate separation
	for i := range len(c.Codes) {
		for j := i + 1; j < len(c.Codes); j++ {
			// Separation = 1 - |correlation| (normalized)
			corr := math32.Abs(c.CalculateCodeCorrelation(c.Codes[i], c.Codes[j]))
			sep := 1.0 - corr
			if sep < minSep {
				minSep = sep
			}
		}
	}

	return minSep
}

// CalculateCodeOrthogonality calculates orthogonality metric for code set
func (c *Config) CalculateCodeOrthogonality() float32 {
	if len(c.Codes) < 2 {
		return 1.0 // Single code is trivially orthogonal
	}

	// Orthogonality = 1 - max_cross_correlation
	maxCrossCorr := c.CalculateMaximumCrossCorrelation()
	return 1.0 - maxCrossCorr
}

// CalculateSymbolSeparationDistance calculates Euclidean distance between symbols in signal space
func (c *Config) CalculateSymbolSeparationDistance(symbolI, symbolJ int) float32 {
	if symbolI == symbolJ {
		return 0.0 // Same symbol
	}

	if symbolI < 0 || symbolI >= len(c.Codes) || symbolJ < 0 || symbolJ >= len(c.Codes) {
		return 0.0 // Invalid symbols
	}

	// For binary (M=2): separation is 2 (between -code and +code)
	if c.ModulationOrder == 2 {
		return 2.0 * c.Amplitude
	}

	// For M-ary: separation = √(2L × (1 - correlation))
	corr := c.CalculateCodeCorrelation(c.Codes[symbolI], c.Codes[symbolJ])
	L := float32(c.SpreadingFactor)
	separation := math32.Sqrt(2.0*L*(1.0-corr)) * c.Amplitude

	return separation
}

// CalculateMinimumSymbolSeparation calculates minimum separation between any two different symbols
func (c *Config) CalculateMinimumSymbolSeparation() float32 {
	if c.ModulationOrder == 2 {
		// Binary: separation is always 2 (between -code and +code)
		return 2.0 * c.Amplitude
	}

	if len(c.Codes) < 2 {
		return 0.0 // Need at least 2 codes
	}

	minSep := float32(math32.MaxFloat32)

	// Find minimum separation between any two different symbols
	for i := range len(c.Codes) {
		for j := i + 1; j < len(c.Codes); j++ {
			sep := c.CalculateSymbolSeparationDistance(i, j)
			if sep < minSep {
				minSep = sep
			}
		}
	}

	return minSep
}

// CalculateMinimumDetectableSeparation calculates minimum separation required for reliable detection
func (c *Config) CalculateMinimumDetectableSeparation(inputSNRdB float32) float32 {
	inputSNRLinear := math32.Pow(10.0, inputSNRdB/10.0)
	noiseAmplitude := c.Amplitude / math32.Sqrt(inputSNRLinear)

	// Detection margin: typically 2-3 standard deviations for reliable detection
	detectionMargin := float32(2.5)

	minDetectable := 2.0 * noiseAmplitude * detectionMargin

	return minDetectable
}

// IsSymbolSeparationSufficient checks if symbol separation is sufficient for reliable demodulation
func (c *Config) IsSymbolSeparationSufficient(inputSNRdB float32) bool {
	minSep := c.CalculateMinimumSymbolSeparation()
	minDetectable := c.CalculateMinimumDetectableSeparation(inputSNRdB)

	// Separation should be at least 2× the minimum detectable for reliable operation
	return minSep >= 2.0*minDetectable
}

// CalculateSymbolSeparationMargin calculates margin between actual and required separation
func (c *Config) CalculateSymbolSeparationMargin(inputSNRdB float32) float32 {
	minSep := c.CalculateMinimumSymbolSeparation()
	minDetectable := c.CalculateMinimumDetectableSeparation(inputSNRdB)

	// Margin = actual_separation - 2×required_separation
	return minSep - 2.0*minDetectable
}

// ConfigError represents a configuration error
type ConfigError struct {
	Message string
}

func (e *ConfigError) Error() string {
	return "dsss.Config: " + e.Message
}

