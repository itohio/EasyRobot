package dsss

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/primitive/generics"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// MinimumSamplesPerChip is the minimum number of samples per chip recommended for proper DSSS operation
// Lower values may work but are not recommended for reliable performance
const MinimumSamplesPerChip = 4

// RecommendedSamplesPerChip is the recommended number of samples per chip for good DSSS performance
const RecommendedSamplesPerChip = 8

// CalculateMinimumSampleRate calculates the minimum sample rate required for DSSS modulation/demodulation
//
// This function determines the minimum sample rate based on two constraints:
// 1. Nyquist criterion for carrier frequency (with safety margin)
// 2. Chip sampling rate (baud rate * chip length * samples per chip)
//
// Parameters:
//   - symbolRate: symbol rate in symbols/sec (baud rate)
//   - spreadingFactor: number of chips per symbol (code length / chip length)
//   - carrierFreq: carrier frequency in Hz
//   - minSamplesPerChip: minimum samples per chip (default: MinimumSamplesPerChip = 4)
//
// Returns:
//   - minimum sample rate in Hz
//
// Constraints:
//  1. Nyquist criterion: sampleRate >= 4 * carrierFreq (4x safety margin to prevent aliasing)
//  2. Chip sampling: sampleRate >= chipRate * minSamplesPerChip
//     where chipRate = symbolRate * spreadingFactor (baud rate * chips per symbol)
//
// The function returns the maximum of these two constraints to satisfy both requirements.
//
// Calculation:
//
//	chipRate = symbolRate * spreadingFactor
//	nyquistRate = 4 * carrierFreq
//	chipSamplingRate = chipRate * minSamplesPerChip
//	minSampleRate = max(nyquistRate, chipSamplingRate)
//
// Example:
//
//	symbolRate (baud) = 1000 symbols/sec
//	spreadingFactor (chip length) = 7 chips/symbol (m-sequence with n=3)
//	carrierFreq = 1000 Hz
//	chipRate = 1000 * 7 = 7000 chips/sec
//	nyquistRate = 4 * 1000 = 4000 Hz
//	chipSamplingRate (4 samples/chip) = 7000 * 4 = 28000 Hz
//	minSampleRate = max(4000, 28000) = 28000 Hz
//	recommendedSampleRate (8 samples/chip) = max(4000, 7000*8) = max(4000, 56000) = 56000 Hz
func CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq float32, minSamplesPerChip int) float32 {
	if minSamplesPerChip <= 0 {
		minSamplesPerChip = MinimumSamplesPerChip
	}

	// Chip rate = symbol rate * spreading factor
	chipRate := symbolRate * spreadingFactor

	// Constraint 1: Nyquist for carrier frequency (with safety margin)
	// Use 4x carrier frequency for safety margin (allows processing without aliasing)
	nyquistRate := 4 * carrierFreq

	// Constraint 2: Sufficient samples per chip
	chipSamplingRate := chipRate * float32(minSamplesPerChip)

	// Return maximum of both constraints
	if nyquistRate > chipSamplingRate {
		return nyquistRate
	}
	return chipSamplingRate
}

// CalculateRecommendedSampleRate calculates the recommended sample rate for DSSS
//
// This function uses RecommendedSamplesPerChip (8 samples per chip) for optimal performance.
// The recommended rate provides better signal quality, timing recovery, and noise immunity
// compared to the minimum rate (4 samples per chip).
//
// Parameters:
//   - symbolRate: symbol rate in symbols/sec (baud rate)
//   - spreadingFactor: number of chips per symbol (code length / chip length)
//   - carrierFreq: carrier frequency in Hz
//
// Returns:
//   - recommended sample rate in Hz (uses 8 samples per chip)
//
// See CalculateMinimumSampleRate for detailed calculation methodology.
func CalculateRecommendedSampleRate(symbolRate, spreadingFactor, carrierFreq float32) float32 {
	return CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq, RecommendedSamplesPerChip)
}

// Modulator handles Direct Sequence Spread Spectrum modulation
// Implements modulation.Modulator interface
type Modulator struct {
	sampleRate        float32
	carrierFreq       float32
	symbolRate        float32
	spreadingFactor   int // L: chips per symbol
	chipsPerSymbol    int // Same as spreadingFactor
	samplesPerChip    int // N_sc: samples per chip
	samplesPerSymbol  int // L * N_sc
	amplitude         float32
	code              Code      // Spreading code for binary DSSS
	codeTable         CodeTable // Code table for M-ary DSSS
	modulationOrder   int       // 2 for binary, M for M-ary
	chipIndex         int       // Current chip index within symbol
	sampleIndexInChip int       // Current sample index within chip
	carrierPhase      float32
}

// NewModulator creates a new DSSS modulator with unified API
//
// Parameters:
//   - sampleRate: sampling frequency in Hz. Must meet minimum requirements (see CalculateMinimumSampleRate)
//   - carrierFreq: carrier frequency in Hz
//   - symbolRate: symbol rate in symbols/sec (baud rate)
//   - codes: spreading codes (varargs)
//   - 1 code: Binary DSSS (modulation order = 2, 1 bit per symbol)
//   - M codes: M-ary DSSS (modulation order = M, log2(M) bits per symbol)
//     M must be a power of 2 (2, 4, 8, 16, 32, 64, etc.)
//
// Sample Rate Requirements:
//
//	The sample rate must satisfy:
//	1. Nyquist criterion: sampleRate >= 4 * carrierFreq (safety margin)
//	2. Chip sampling: sampleRate >= chipRate * MinimumSamplesPerChip
//	   where chipRate = symbolRate * spreadingFactor
//
//	Use CalculateMinimumSampleRate() or CalculateRecommendedSampleRate() to determine
//	the required sample rate for your parameters.
//
//	Example:
//	  For symbolRate=1000, spreadingFactor=7, carrierFreq=1000:
//	  - Minimum: 28000 Hz (4 samples/chip)
//	  - Recommended: 56000 Hz (8 samples/chip)
//
// Code Requirements:
//   - Binary DSSS: 1 code (reused with ±1 based on bit value)
//   - 4-ary DSSS: 4 codes (2 bits per symbol, codes for symbols 00, 01, 10, 11)
//   - 8-ary DSSS: 8 codes (3 bits per symbol)
//   - 16-ary DSSS: 16 codes (4 bits per symbol)
//   - M-ary DSSS: M codes where M = 2^k (k bits per symbol)
//
// All codes must have the same length (spreading factor).
// Codes should be orthogonal or near-orthogonal for best performance.
//
// Example - Binary DSSS:
//
//	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
//	minRate := CalculateRecommendedSampleRate(1000, code.Length(), 1000)
//	modulator := NewModulator(minRate, 1000, 1000, code)
//
// Example - 4-ary DSSS:
//
//	codeTable := GenerateWalshCodeTable(4) // 4 orthogonal codes
//	modulator := NewModulator(48000, 1000, 1000, codeTable[0], codeTable[1], codeTable[2], codeTable[3])
func NewModulator(sampleRate, carrierFreq, symbolRate float32, codes ...Code) *Modulator {
	if len(codes) == 0 {
		panic("DSSS modulator: at least one code must be provided")
	}

	// Validate codes
	firstCodeLength := codes[0].length
	if firstCodeLength == 0 {
		panic("DSSS modulator: codes must not be empty")
	}

	// Verify all codes have same length
	for i, code := range codes {
		if code.length == 0 {
			panic(fmt.Sprintf("DSSS modulator: code %d must not be empty", i))
		}
		if code.length != firstCodeLength {
			panic(fmt.Sprintf("DSSS modulator: code %d has length %d, expected %d", i, code.length, firstCodeLength))
		}
	}

	// Determine modulation order
	// 1 code = binary DSSS (modulationOrder = 2)
	// M codes = M-ary DSSS (modulationOrder = M)
	numCodes := len(codes)
	var modulationOrder int
	if numCodes == 1 {
		modulationOrder = 2 // Binary DSSS
	} else {
		modulationOrder = numCodes
		// For M-ary, verify M is a power of 2
		if (modulationOrder & (modulationOrder - 1)) != 0 {
			panic(fmt.Sprintf("DSSS modulator: M-ary requires M to be a power of 2, got %d codes", modulationOrder))
		}
	}

	spreadingFactor := firstCodeLength
	chipRate := symbolRate * float32(spreadingFactor)

	// Validate sample rate meets minimum requirements
	minSampleRate := CalculateMinimumSampleRate(symbolRate, float32(spreadingFactor), carrierFreq, MinimumSamplesPerChip)
	if sampleRate < minSampleRate {
		panic(fmt.Sprintf("DSSS modulator: sample rate %f Hz is below minimum %f Hz (symbolRate=%f, spreadingFactor=%d, carrierFreq=%f). Use CalculateMinimumSampleRate() to determine required rate.",
			sampleRate, minSampleRate, symbolRate, spreadingFactor, carrierFreq))
	}

	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS modulator: samplesPerChip must be positive, got %d (sampleRate=%f, chipRate=%f)", samplesPerChip, sampleRate, chipRate))
	}

	// Warn if below recommended (but don't fail)
	if samplesPerChip < RecommendedSamplesPerChip {
		recommendedRate := CalculateRecommendedSampleRate(symbolRate, float32(spreadingFactor), carrierFreq)
		// Note: Using fmt.Printf for warning would be ideal, but to keep it clean, we just validate minimum
		_ = recommendedRate // Suppress unused warning
	}

	mod := &Modulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		spreadingFactor:  spreadingFactor,
		chipsPerSymbol:   spreadingFactor,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: spreadingFactor * samplesPerChip,
		amplitude:        1.0,
		modulationOrder:  modulationOrder,
		carrierPhase:     0,
	}

	// Set code or codeTable based on number of codes
	if numCodes == 1 {
		mod.code = codes[0] // Binary DSSS: single code
	} else {
		mod.codeTable = make(CodeTable, modulationOrder)
		copy(mod.codeTable, codes) // M-ary DSSS: code table
		// For binary DSSS (modulationOrder=2), also set code field for Encode method
		if modulationOrder == 2 {
			mod.code = codes[0] // Use first code, Encode will multiply by +1/-1 based on symbol
		}
	}

	return mod
}

// SetAmplitude sets the signal amplitude for modulation
//
// Deprecated: This method is not part of the Modulator interface.
// Use the constructor parameters or configure amplitude during initialization.
func (dsss *Modulator) SetAmplitude(amplitude float32) {
	dsss.amplitude = amplitude
}

// Reset resets the modulator state
func (dsss *Modulator) Reset() {
	dsss.chipIndex = 0
	dsss.sampleIndexInChip = 0
	dsss.carrierPhase = 0
}

// Encode encodes bytes into amplitude vector
// Implements modulation.Modulator interface
// Returns the number of amplitude samples written
// Returns expected amplitudes length if amplitudes is nil
func (dsss *Modulator) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if amplitudes == nil {
		// Calculate expected length
		bitsPerSymbol := int(math.Log2(float64(dsss.modulationOrder)))
		bitsNeeded := len(data) * 8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * dsss.samplesPerSymbol
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	samplesWritten := 0

	// Use BitBuffer to read symbols from input data
	bitsPerSymbol := int(math.Log2(float64(dsss.modulationOrder)))
	bitBuffer := generics.NewBitBuffer[byte](bitsPerSymbol)
	bitBuffer.ReadFromBytes(data)
	bitBuffer.Reset()

	// Process symbols and generate signal
	for samplesWritten < len(ampData) {
		// Read symbol value using BitBuffer
		symbolValue, ok := bitBuffer.Read()
		if !ok {
			// No more symbols available
			break
		}
		symbolIndex := int(symbolValue)

		// Get spreading code for this symbol
		var code Code
		if dsss.modulationOrder == 2 {
			// Binary DSSS: use single code, multiply by symbol value
			code = dsss.code
		} else {
			// M-ary DSSS: select code from table
			code = dsss.codeTable[symbolIndex]
		}

		// Generate symbol: spread each chip (work directly with bits)
		codeLength := code.length
		for chipIdx := 0; chipIdx < codeLength && samplesWritten < len(ampData); chipIdx++ {
			// Get bipolar value directly from bit (no allocation)
			chipValue := code.GetBipolar(chipIdx)

			// For binary DSSS, multiply chip by symbol value (+1 or -1)
			if dsss.modulationOrder == 2 {
				if symbolIndex == 0 {
					chipValue = -chipValue // Bit 0 -> -1
				} else {
					chipValue = +chipValue // Bit 1 -> +1
				}
			}

			// Generate samples for this chip
			for s := 0; s < dsss.samplesPerChip && samplesWritten < len(ampData); s++ {
				// Carrier modulation
				phase := 2 * math.Pi * dsss.carrierFreq * dsss.carrierPhase / dsss.sampleRate
				carrier := float32(math.Cos(float64(phase)))

				// DSSS modulation: chip * carrier * amplitude
				ampData[samplesWritten] = dsss.amplitude * chipValue * carrier

				dsss.carrierPhase += 1.0
				samplesWritten++
			}
		}
	}

	return samplesWritten
}

// ModulationOrder returns the modulation order
//
// Deprecated: This method is not part of the Modulator interface.
func (dsss *Modulator) ModulationOrder() int {
	return dsss.modulationOrder
}

// SpreadingFactor returns the spreading factor (chips per symbol)
//
// Deprecated: This method is not part of the Modulator interface.
func (dsss *Modulator) SpreadingFactor() int {
	return dsss.spreadingFactor
}
