package dsss

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	agcFilter "github.com/itohio/EasyRobot/x/math/filter/agc"
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
	signalDetector *modulation.SignalDetector
	agc            *agcFilter.Filter

	// Acquisition state
	acquired          bool
	codePhase         int     // Current code phase offset (in samples)
	carrierPhase      float32 // Current carrier phase
	carrierFreqOffset float32 // Estimated carrier frequency offset

	// Tracking loops
	dllGain     float32 // DLL gain for timing tracking
	pllGain     float32 // PLL gain for carrier tracking
	timingError float32 // Current timing error estimate
	phaseError  float32 // Current phase error estimate

	// Buffers
	sampleBuffer []float32 // Ring buffer for incoming samples
	bufferSize   int
	bufferIndex  int

	// Symbol decision
	symbolBuffer []int // Recovered symbol decisions
	bitExtractor *modulation.BitExtractor

	// State
	bitsPerSymbol int
	locked        bool

	// Callback for packet end detection
	isEndOfPacket func([]byte) bool
}

// NewDemodulator creates a new DSSS demodulator with unified API
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in symbols/sec
// codes: spreading codes (varargs, must match encoder)
//   - 1 code: Binary DSSS (modulation order = 2, 1 bit per symbol)
//   - M codes: M-ary DSSS (modulation order = M, log2(M) bits per symbol)
//     M must be a power of 2 (2, 4, 8, 16, 32, 64, etc.)
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
//	demodulator := NewDemodulator(48000, 1000, 1000, code)
//
// Example - 4-ary DSSS:
//
//	codeTable := GenerateWalshCodeTable(4)
//	demodulator := NewDemodulator(48000, 1000, 1000, codeTable[0], codeTable[1], codeTable[2], codeTable[3])
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
	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS demodulator: samplesPerChip must be positive, got %d", samplesPerChip))
	}

	samplesPerSymbol := spreadingFactor * samplesPerChip
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))

	// Initialize signal detector
	detectorWindow := samplesPerSymbol * 4
	signalDetector := modulation.NewSignalDetector(detectorWindow, modulation.MinSignalEnergy*10)
	signalDetector.SetMinSNR(3.0) // Require 3dB SNR

	// Initialize AGC
	agc := agcFilter.New(0.5, 0.01)
	agc.SetSampleRate(sampleRate)

	// Initialize bit extractor
	bitExtractor := modulation.NewBitExtractor()

	// Buffer size: enough for acquisition window + margin
	bufferSize := samplesPerSymbol * 8

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
		bitExtractor:     bitExtractor,
		dllGain:          0.1,
		pllGain:          0.1,
		sampleBuffer:     make([]float32, bufferSize),
		bufferSize:       bufferSize,
	}

	// Set code or codeTable based on number of codes
	if numCodes == 1 {
		demod.code = codes[0] // Binary DSSS: single code
	} else {
		demod.codeTable = make(CodeTable, modulationOrder)
		copy(demod.codeTable, codes) // M-ary DSSS: code table
	}

	return demod
}

// NewMDemodulator creates a new DSSS demodulator for M-ary DSSS
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in symbols/sec
// codeTable: table of spreading codes (must match encoder)
func NewMDemodulator(sampleRate, carrierFreq, symbolRate float32, codeTable CodeTable) *Demodulator {
	if len(codeTable) == 0 {
		panic("DSSS demodulator: codeTable must not be empty")
	}

	spreadingFactor := codeTable[0].length
	chipRate := symbolRate * float32(spreadingFactor)
	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS demodulator: samplesPerChip must be positive, got %d", samplesPerChip))
	}

	samplesPerSymbol := spreadingFactor * samplesPerChip
	modulationOrder := len(codeTable)
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))

	// Initialize signal detector
	detectorWindow := samplesPerSymbol * 4
	signalDetector := modulation.NewSignalDetector(detectorWindow, modulation.MinSignalEnergy*10)
	signalDetector.SetMinSNR(3.0)

	// Initialize AGC
	agc := agcFilter.New(0.5, 0.01)
	agc.SetSampleRate(sampleRate)

	// Initialize bit extractor
	bitExtractor := modulation.NewBitExtractor()

	// Buffer size
	bufferSize := samplesPerSymbol * 8

	return &Demodulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		spreadingFactor:  spreadingFactor,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: samplesPerSymbol,
		codeTable:        codeTable,
		modulationOrder:  modulationOrder,
		bitsPerSymbol:    bitsPerSymbol,
		signalDetector:   signalDetector,
		agc:              agc,
		bitExtractor:     bitExtractor,
		dllGain:          0.1,
		pllGain:          0.1,
		sampleBuffer:     make([]float32, bufferSize),
		bufferSize:       bufferSize,
	}
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
	dsss.symbolBuffer = dsss.symbolBuffer[:0]
	if dsss.bitExtractor != nil {
		dsss.bitExtractor.Reset()
	}
	if dsss.signalDetector != nil {
		dsss.signalDetector.Reset()
	}
}

// OnIsEndOfPacket sets a callback to check if packet is complete
func (dsss *Demodulator) OnIsEndOfPacket(callback func([]byte) bool) {
	dsss.isEndOfPacket = callback
}

// IsLocked returns whether the demodulator is locked
func (dsss *Demodulator) IsLocked() bool {
	return dsss.locked
}

// Decode decodes amplitude vector into bytes
// Implements modulation.Demodulator interface
// Returns the number of bytes written
// Returns expected data length if data is nil
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
	}

	// Update signal detector
	for _, sample := range ampData {
		dsss.signalDetector.Update(sample)
	}

	// Check if signal is detected
	if !dsss.acquired && dsss.signalDetector.IsLocked() {
		// Attempt acquisition
		dsss.attemptAcquisition()
	}

	if !dsss.acquired {
		return 0
	}

	// Process samples for despreading
	bytesDecoded := 0
	samplesProcessed := 0

	// Process in symbol-sized chunks
	for samplesProcessed < len(ampData) && bytesDecoded < len(data) {
		// Check if we have enough samples for a symbol
		if dsss.bufferIndex < dsss.samplesPerSymbol {
			break
		}

		// Extract symbol
		symbol := dsss.despreadSymbol()

		if symbol >= 0 {
			// Add symbol to bit extractor
			dsss.bitExtractor.AddSymbol(symbol, dsss.bitsPerSymbol)

			// Extract bytes
			remaining := data[bytesDecoded:]
			extracted := dsss.bitExtractor.ExtractBytes(remaining)
			bytesDecoded += extracted

			// Check if packet is complete
			if dsss.isEndOfPacket != nil && bytesDecoded > 0 && dsss.isEndOfPacket(data[:bytesDecoded]) {
				return bytesDecoded
			}
		}

		samplesProcessed += dsss.samplesPerSymbol
	}

	return bytesDecoded
}

// attemptAcquisition attempts to acquire the spreading code
func (dsss *Demodulator) attemptAcquisition() {
	// Simple acquisition: correlate with code at different phases
	maxCorrelation := float32(0)
	bestPhase := 0

	// Search over possible code phases
	searchRange := dsss.samplesPerSymbol
	for phase := 0; phase < searchRange; phase++ {
		correlation := dsss.computeCorrelation(phase)
		magnitude := float32(math.Abs(float64(correlation)))
		if magnitude > maxCorrelation {
			maxCorrelation = magnitude
			bestPhase = phase
		}
	}

	// Check if correlation is strong enough
	threshold := float32(0.5) * float32(dsss.spreadingFactor) // Threshold based on spreading factor
	if maxCorrelation > threshold {
		dsss.codePhase = bestPhase
		dsss.acquired = true
		dsss.locked = true
	}
}

// computeCorrelation computes correlation at given code phase
func (dsss *Demodulator) computeCorrelation(phase int) float32 {
	var correlation float32

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
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStart := phase + chipIdx*dsss.samplesPerChip
		if chipStart >= dsss.bufferSize {
			break
		}

		// Sample at chip center (middle of chip period)
		sampleIdx := chipStart + dsss.samplesPerChip/2
		if sampleIdx >= dsss.bufferSize {
			break
		}

		// Get sample (accounting for ring buffer)
		sample := dsss.sampleBuffer[sampleIdx%dsss.bufferSize]

		// Apply carrier correction (simplified - just use real part)
		carrierPhase := 2 * math.Pi * dsss.carrierFreq * float32(sampleIdx) / dsss.sampleRate
		carrierCorrection := float32(math.Cos(float64(carrierPhase)))

		// Correlate: sample * code_chip * carrier_correction (get bipolar directly)
		chipValue := code.GetBipolar(chipIdx)
		correlation += sample * chipValue * carrierCorrection
	}

	return correlation
}

// despreadSymbol despreads one symbol from the received signal
// Returns symbol index, or -1 if not enough samples
func (dsss *Demodulator) despreadSymbol() int {
	if !dsss.acquired {
		return -1
	}

	if dsss.modulationOrder == 2 {
		// Binary DSSS: correlate with single code
		correlation := dsss.computeCorrelation(dsss.codePhase)

		// Decision: positive -> 1, negative -> 0
		if correlation > 0 {
			return 1
		}
		return 0
	} else {
		// M-ary DSSS: correlate with all codes, choose maximum
		maxCorrelation := float32(0)
		bestSymbol := 0

		for symbolIdx, code := range dsss.codeTable {
			// Compute correlation with this code
			correlation := dsss.computeCorrelationWithCode(dsss.codePhase, code)
			magnitude := float32(math.Abs(float64(correlation)))

			if magnitude > maxCorrelation {
				maxCorrelation = magnitude
				bestSymbol = symbolIdx
			}
		}

		return bestSymbol
	}
}

// computeCorrelationWithCode computes correlation with a specific code
func (dsss *Demodulator) computeCorrelationWithCode(phase int, code Code) float32 {
	var correlation float32

	// Work directly with bits (no allocation)
	codeLength := code.length
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStart := phase + chipIdx*dsss.samplesPerChip
		if chipStart >= dsss.bufferSize {
			break
		}

		sampleIdx := chipStart + dsss.samplesPerChip/2
		if sampleIdx >= dsss.bufferSize {
			break
		}

		sample := dsss.sampleBuffer[sampleIdx%dsss.bufferSize]

		carrierPhase := 2 * math.Pi * dsss.carrierFreq * float32(sampleIdx) / dsss.sampleRate
		carrierCorrection := float32(math.Cos(float64(carrierPhase)))

		// Get bipolar value directly from bit
		chipValue := code.GetBipolar(chipIdx)
		correlation += sample * chipValue * carrierCorrection
	}

	return correlation
}
