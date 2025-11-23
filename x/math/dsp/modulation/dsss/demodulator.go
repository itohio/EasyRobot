package dsss

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	agcFilter "github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/math/filter/pll"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Demodulator handles Direct Sequence Spread Spectrum demodulation
// Implements modulation.Demodulator interface
type Demodulator struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	spreadingFactor  int
	samplesPerChip   int
	samplesPerSymbol int
	code             Code
	codeTable        CodeTable
	modulationOrder  int

	// Signal detection and AGC
	signalDetector modulation.SignalDetector
	agc            agcFilter.Filter

	// Testable components (SOLID principle - single responsibility)
	acquisition Acquisition
	despreader  Despreader

	// Acquisition state
	acquired          bool
	codePhase         int     // Current code phase offset (in samples)
	carrierPhase      float32 // Current carrier phase (for modulator compatibility)
	carrierFreqOffset float32 // Estimated carrier frequency offset

	// Tracking loops
	pllFilter   pll.Filter // PLL for carrier phase/frequency tracking
	dllGain     float32    // DLL gain for timing tracking
	timingError float32    // Current timing error estimate
	phaseError  float32    // Current phase error estimate

	// Buffers
	sampleBuffer []float32 // Ring buffer for incoming samples
	bufferSize   int
	bufferIndex  int
	totalSamples int // Total number of samples processed (for carrier phase tracking)

	// Symbol decision and bit conversion
	symbolBuffer   []int             // Recovered symbol decisions
	bitBuffer      buffer.Bits[byte] // BitBuffer for symbol-to-byte conversion
	bytesExtracted int               // Number of complete bytes already extracted from bitBuffer

	// State
	bitsPerSymbol int
	locked        bool
}

// NewDemodulator creates a new DSSS demodulator with unified API
//
// Parameters:
//   - sampleRate: sampling frequency in Hz. Must meet minimum requirements (see CalculateMinimumSampleRate)
//   - carrierFreq: carrier frequency in Hz
//   - symbolRate: symbol rate in symbols/sec (baud rate)
//   - codes: spreading codes (varargs, must match encoder)
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
//   - Binary DSSS: 1 code (same as encoder)
//   - 4-ary DSSS: 4 codes (same order as encoder: codes for symbols 00, 01, 10, 11)
//   - 8-ary DSSS: 8 codes (same order as encoder)
//   - M-ary DSSS: M codes in the same order as encoder
//
// All codes must have the same length and match the encoder exactly.
//
// Example - Binary DSSS:
//
//	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
//	minRate := CalculateRecommendedSampleRate(1000, code.Length(), 1000)
//	demodulator := NewDemodulator(minRate, 1000, 1000, code)
//
// Example - 4-ary DSSS:
//
//	codeTable := GenerateWalshCodeTable(4)
//	recommendedRate := CalculateRecommendedSampleRate(1000, float32(codeTable[0].Length()), 1000)
//	demodulator := NewDemodulator(recommendedRate, 1000, 1000, codeTable[0], codeTable[1], codeTable[2], codeTable[3])
func NewDemodulator(sampleRate, carrierFreq, symbolRate float32, codes ...Code) *Demodulator {
	if len(codes) == 0 {
		panic("DSSS demodulator: at least one code must be provided")
	}

	// Validate codes
	firstCodeLength := codes[0].length
	if firstCodeLength == 0 {
		panic("DSSS demodulator: codes must not be empty")
	}

	// Verify all codes have same length
	for i, code := range codes {
		if code.length == 0 {
			panic(fmt.Sprintf("DSSS demodulator: code %d must not be empty", i))
		}
		if code.length != firstCodeLength {
			panic(fmt.Sprintf("DSSS demodulator: code %d has length %d, expected %d", i, code.length, firstCodeLength))
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
			panic(fmt.Sprintf("DSSS demodulator: M-ary requires M to be a power of 2, got %d codes", modulationOrder))
		}
	}

	spreadingFactor := firstCodeLength
	chipRate := symbolRate * float32(spreadingFactor)

	// Validate sample rate meets minimum requirements
	minSampleRate := CalculateMinimumSampleRate(symbolRate, float32(spreadingFactor), carrierFreq, MinimumSamplesPerChip)
	if sampleRate < minSampleRate {
		panic(fmt.Sprintf("DSSS demodulator: sample rate %f Hz is below minimum %f Hz (symbolRate=%f, spreadingFactor=%d, carrierFreq=%f). Use CalculateMinimumSampleRate() to determine required rate.",
			sampleRate, minSampleRate, symbolRate, spreadingFactor, carrierFreq))
	}

	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS demodulator: samplesPerChip must be positive, got %d", samplesPerChip))
	}

	samplesPerSymbol := spreadingFactor * samplesPerChip
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))

	// Initialize signal detector
	detectorWindow := samplesPerSymbol * 4
	signalDetector := *modulation.NewSignalDetector(detectorWindow, modulation.MinSignalEnergy*10)
	signalDetector.SetMinSNR(3.0) // Require 3dB SNR

	// Initialize AGC
	agc := *agcFilter.New(0.5, 0.01)
	agc.SetSampleRate(sampleRate)

	// Initialize symbol buffer and BitBuffer
	symbolBuffer := make([]int, 0, 64)
	bitBuffer := buffer.NewBits[byte](bitsPerSymbol)

	// Initialize PLL for carrier phase tracking
	// Use loop bandwidth of 5% of symbol rate for good tracking vs noise trade-off
	loopBandwidth := symbolRate * 0.05
	pllFilter := *pll.NewFromLoopBandwidth(loopBandwidth, symbolRate, 0.707)

	// Buffer size: enough for acquisition window + margin
	bufferSize := samplesPerSymbol * 8

	// Create testable components (SOLID principle)
	var code Code
	var codeTable CodeTable
	if numCodes == 1 {
		code = codes[0] // Binary DSSS: single code
	} else {
		codeTable = make(CodeTable, modulationOrder)
		copy(codeTable, codes) // M-ary DSSS: code table
	}

	acquisition := NewAcquisition(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize,
		code, codeTable, carrierFreq, sampleRate)
	despreader := NewDespreader(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize,
		code, codeTable, carrierFreq, sampleRate)

	demod := &Demodulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		spreadingFactor:  spreadingFactor,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: samplesPerSymbol,
		modulationOrder:  modulationOrder,
		bitsPerSymbol:    bitsPerSymbol,
		signalDetector:   signalDetector,
		agc:              agc,
		acquisition:      acquisition,
		despreader:       despreader,
		symbolBuffer:     symbolBuffer,
		bitBuffer:        bitBuffer,
		bytesExtracted:   0,
		pllFilter:        pllFilter,
		dllGain:          0.1,
		sampleBuffer:     make([]float32, bufferSize),
		bufferSize:       bufferSize,
		code:             code,
		codeTable:        codeTable,
	}

	return demod
}

// Reset resets the demodulator state
func (dsss *Demodulator) Reset() {
	dsss.acquired = false
	dsss.locked = false
	dsss.codePhase = 0
	dsss.carrierPhase = 0
	dsss.carrierFreqOffset = 0
	dsss.timingError = 0
	dsss.phaseError = 0
	dsss.bufferIndex = 0
	dsss.totalSamples = 0
	dsss.symbolBuffer = dsss.symbolBuffer[:0]
	dsss.bitBuffer.Clear()
	dsss.bytesExtracted = 0
	dsss.signalDetector.Reset()
	dsss.agc.Reset()
	dsss.pllFilter.Reset()
}

// IsLocked returns whether the demodulator is locked
//
// Deprecated: This method is not part of the Demodulator interface.
func (dsss *Demodulator) IsLocked() bool {
	return dsss.locked
}

// Decode decodes amplitude vector into bytes
// Implements modulation.Demodulator interface
// Returns the number of bytes written
// Returns expected data length if data is nil
//
// Flow: 1. Try to get lock (acquisition)
//  2. Try to detect preamble (future: can add preamble detection here)
//  3. Decode symbols and add them to buffer
//  4. Convert symbols to bytes using BitWriter-like pattern
func (dsss *Demodulator) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		// Assume we can decode all symbols in the buffer
		maxSymbols := dsss.bufferSize / dsss.samplesPerSymbol
		return (maxSymbols * dsss.bitsPerSymbol) / 8
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	// Add samples to buffer
	for _, sample := range ampData {
		dsss.sampleBuffer[dsss.bufferIndex] = sample
		dsss.bufferIndex = (dsss.bufferIndex + 1) % dsss.bufferSize
		dsss.totalSamples++
	}

	// Update signal detector
	for _, sample := range ampData {
		dsss.signalDetector.Update(sample)
	}

	// Step 1: Try to get lock (acquisition)
	if !dsss.acquired {
		// Only attempt acquisition if we have enough samples
		// Need at least samplesPerSymbol samples for correlation
		if dsss.totalSamples >= dsss.samplesPerSymbol {
			dsss.attemptAcquisition()
		}
		if !dsss.acquired {
			return 0
		}
	}

	// Step 2: Try to detect preamble (future enhancement)
	// For now, we proceed directly to symbol decoding after acquisition

	// Step 3: Decode symbols and add them to buffer
	bytesDecoded := 0
	samplesProcessed := 0

	// Process in symbol-sized chunks
	for samplesProcessed < len(ampData) && bytesDecoded < len(data) {
		// Check if we have enough samples for a symbol
		if dsss.bufferIndex < dsss.samplesPerSymbol {
			break
		}

		// Decode symbol
		symbol := dsss.despreadSymbol()

		if symbol >= 0 {
			// Write symbol to BitBuffer
			dsss.bitBuffer.Write(byte(symbol))

			// Convert bits to bytes and extract new complete bytes
			allBytes := dsss.bitBuffer.ToBytes()
			if len(allBytes) > dsss.bytesExtracted {
				// Copy new bytes to output
				newBytes := allBytes[dsss.bytesExtracted:]
				remaining := data[bytesDecoded:]
				copied := copy(remaining, newBytes)
				bytesDecoded += copied
				dsss.bytesExtracted += copied

				// If we extracted all complete bytes, compact the buffer
				bitsWritten := dsss.bitBuffer.GetWrittenBits()
				completeBytes := (bitsWritten / 8) * 8
				if dsss.bytesExtracted*8 >= completeBytes && completeBytes > 0 {
					// Move remaining bits to beginning
					dsss.bitBuffer.SetPosition(completeBytes)
					dsss.bitBuffer.Compact()
					dsss.bytesExtracted = 0
				}
			}

			// Advance code phase for next symbol
			dsss.codePhase = (dsss.codePhase + dsss.samplesPerSymbol) % dsss.bufferSize
		} else {
			// If we can't despread, we might have lost lock
			// Advance buffer position anyway to avoid getting stuck
			if dsss.acquired {
				dsss.codePhase = (dsss.codePhase + dsss.samplesPerSymbol) % dsss.bufferSize
			}
		}

		samplesProcessed += dsss.samplesPerSymbol
	}

	return bytesDecoded
}

// attemptAcquisition attempts to acquire the spreading code
// Uses the testable Acquisition component
func (dsss *Demodulator) attemptAcquisition() {
	acquired, codePhase, _ := dsss.acquisition.AttemptAcquisition(
		dsss.sampleBuffer, dsss.bufferIndex, dsss.totalSamples)

	if acquired {
		dsss.codePhase = codePhase
		dsss.acquired = true
		dsss.locked = true
		// Initialize PLL - it will track phase from here
		dsss.pllFilter.Reset()
	}
}

// computeSimpleCorrelation computes simple correlation with carrier phase = 0
// Used for initial acquisition search
func (dsss *Demodulator) computeSimpleCorrelation(phase int) float32 {
	var correlation float32

	// Get code to correlate with
	var code Code
	if dsss.modulationOrder == 2 {
		code = dsss.code
	} else {
		code = dsss.codeTable[0]
	}

	codeLength := code.length
	// Calculate buffer start: we want to look back from current write position
	// bufferIndex points to the next write position, so we need to look back
	// by samplesPerSymbol + phase to find the start of the symbol we're correlating
	lookback := dsss.samplesPerSymbol + phase
	if lookback > dsss.bufferSize {
		lookback = dsss.bufferSize
	}
	bufferStart := (dsss.bufferIndex - lookback + dsss.bufferSize) % dsss.bufferSize

	// Calculate absolute sample index for carrier phase calculation
	// This should match the modulator's carrierPhase counter
	correlationStartSample := dsss.totalSamples - lookback
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}

	// Correlate: multiply samples by code and accumulate
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * dsss.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		for s := 0; s < dsss.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s
			bufferIdx := (bufferStart + sampleOffset) % dsss.bufferSize
			sample := dsss.sampleBuffer[bufferIdx]

			// Compute carrier phase and mix down
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * dsss.carrierFreq * float32(absoluteSampleIdx) / dsss.sampleRate
			carrierMix := float32(math.Cos(float64(carrierPhase)))

			// Correlate: sample * code * carrier
			correlation += sample * chipValue * carrierMix
		}
	}

	// Return absolute value for magnitude
	if correlation < 0 {
		return -correlation
	}
	return correlation
}

// computeIQCorrelationWithPhaseOffset computes I/Q correlation with a carrier phase offset
// This is used during acquisition when we don't know the exact carrier phase
func (dsss *Demodulator) computeIQCorrelationWithPhaseOffset(phase int, carrierPhaseOffset float32) (float32, float32) {
	var iCorr, qCorr float32

	// Get code to correlate with
	var code Code
	if dsss.modulationOrder == 2 {
		code = dsss.code
	} else {
		code = dsss.codeTable[0]
	}

	codeLength := code.length
	correlationStartSample := dsss.totalSamples - dsss.samplesPerSymbol - phase
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}
	bufferStart := (dsss.bufferIndex - dsss.samplesPerSymbol - phase + dsss.bufferSize) % dsss.bufferSize

	// Integrate over all samples
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * dsss.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		for s := 0; s < dsss.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s
			bufferIdx := (bufferStart + sampleOffset) % dsss.bufferSize
			sample := dsss.sampleBuffer[bufferIdx]

			// Compute carrier phase with offset
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * dsss.carrierFreq * float32(absoluteSampleIdx) / dsss.sampleRate
			correctedPhase := carrierPhase - carrierPhaseOffset

			iCorrLocal := float32(math.Cos(float64(correctedPhase)))
			qCorrLocal := float32(math.Sin(float64(correctedPhase)))

			iCorr += sample * chipValue * iCorrLocal
			qCorr += sample * chipValue * qCorrLocal
		}
	}

	return iCorr, qCorr
}

// computeIQCorrelation computes I/Q correlation at given code phase (for acquisition)
// Returns I and Q correlation values
func (dsss *Demodulator) computeIQCorrelation(phase int) (float32, float32) {
	var iCorr, qCorr float32

	// Get code to correlate with
	var code Code
	if dsss.modulationOrder == 2 {
		code = dsss.code
	} else {
		// For M-ary, try first code (will be refined during tracking)
		code = dsss.codeTable[0]
	}

	// Correlate over one symbol period (work directly with bits)
	codeLength := code.length

	// Calculate absolute sample index where correlation starts
	// Ensure we don't go negative
	correlationStartSample := dsss.totalSamples - dsss.samplesPerSymbol - phase
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}

	// Get buffer position for the start of correlation
	// Look back from bufferIndex (which points to next write position)
	bufferStart := (dsss.bufferIndex - dsss.samplesPerSymbol - phase + dsss.bufferSize) % dsss.bufferSize

	// Get current PLL phase estimate for carrier correction (only if acquired)
	var pllPhase float32
	if dsss.acquired {
		pllPhase = dsss.pllFilter.Phase()
	}

	// Integrate over all samples in the symbol period
	// For each chip, integrate over all samples in that chip period
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * dsss.samplesPerChip

		// Get bipolar chip value (constant for this chip)
		chipValue := code.GetBipolar(chipIdx)

		// Integrate over all samples in this chip period
		for s := 0; s < dsss.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s

			// Get buffer index for this sample
			bufferIdx := (bufferStart + sampleOffset) % dsss.bufferSize
			sample := dsss.sampleBuffer[bufferIdx]

			// Compute carrier phase using absolute sample index
			// Modulator uses: phase = 2π * carrierFreq * sampleIndex / sampleRate
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * dsss.carrierFreq * float32(absoluteSampleIdx) / dsss.sampleRate

			// Apply PLL phase correction only if acquired
			correctedPhase := carrierPhase - pllPhase

			// I/Q correlation: in-phase and quadrature components
			iCorrLocal := float32(math.Cos(float64(correctedPhase)))
			qCorrLocal := float32(math.Sin(float64(correctedPhase)))

			// Accumulate correlation: sample * code_chip * carrier_correction
			iCorr += sample * chipValue * iCorrLocal
			qCorr += sample * chipValue * qCorrLocal
		}
	}

	// Don't update PLL here - this is for acquisition, PLL update happens in computeCorrelation
	return iCorr, qCorr
}

// computeCorrelation computes I/Q correlation at given code phase and updates PLL
// Returns in-phase (I) correlation value
func (dsss *Demodulator) computeCorrelation(phase int) float32 {
	iCorr, qCorr := dsss.computeIQCorrelation(phase)

	// Update PLL with phase error after correlation
	// Phase error = atan2(Q, I) - desired_phase (which is 0 for binary DSSS)
	if iCorr != 0 || qCorr != 0 {
		phaseError := float32(math.Atan2(float64(qCorr), float64(iCorr)))
		if dsss.acquired {
			// Only update PLL after acquisition to avoid disrupting acquisition
			dsss.pllFilter.Update(phaseError)
		}
	}

	// Return in-phase correlation (this is what we use for binary DSSS decision)
	return iCorr
}

// despreadSymbol despreads one symbol from the received signal
// Returns symbol index, or -1 if not enough samples
// Uses the testable Despreader component
func (dsss *Demodulator) despreadSymbol() int {
	if !dsss.acquired {
		return -1
	}

	pllPhase := dsss.pllFilter.Phase()
	return dsss.despreader.DespreadSymbol(
		dsss.sampleBuffer, dsss.bufferIndex, dsss.totalSamples, dsss.codePhase, pllPhase)
}

// computeCorrelationWithCode computes I/Q correlation with a specific code
// Returns magnitude of complex correlation
func (dsss *Demodulator) computeCorrelationWithCode(phase int, code Code) float32 {
	var iCorr, qCorr float32

	// Work directly with bits (no allocation)
	codeLength := code.length

	// Calculate absolute sample index where correlation starts
	correlationStartSample := dsss.totalSamples - dsss.samplesPerSymbol - phase
	bufferStart := (dsss.bufferIndex - dsss.samplesPerSymbol - phase + dsss.bufferSize) % dsss.bufferSize

	// Get current PLL phase estimate (only if acquired)
	var pllPhase float32
	if dsss.acquired {
		pllPhase = dsss.pllFilter.Phase()
	}

	// Integrate over all samples in the symbol period
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * dsss.samplesPerChip

		// Get bipolar chip value (constant for this chip)
		chipValue := code.GetBipolar(chipIdx)

		// Integrate over all samples in this chip period
		for s := 0; s < dsss.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s

			// Get buffer index for this sample
			bufferIdx := (bufferStart + sampleOffset) % dsss.bufferSize
			sample := dsss.sampleBuffer[bufferIdx]

			// Compute carrier phase using absolute sample index
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * dsss.carrierFreq * float32(absoluteSampleIdx) / dsss.sampleRate

			// Apply PLL phase correction only if acquired
			correctedPhase := carrierPhase - pllPhase

			// I/Q correlation
			iCorrLocal := float32(math.Cos(float64(correctedPhase)))
			qCorrLocal := float32(math.Sin(float64(correctedPhase)))

			// Accumulate correlation
			iCorr += sample * chipValue * iCorrLocal
			qCorr += sample * chipValue * qCorrLocal
		}
	}

	// Return magnitude for M-ary DSSS
	return float32(math.Sqrt(float64(iCorr*iCorr + qCorr*qCorr)))
}
