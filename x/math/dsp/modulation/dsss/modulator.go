package dsss

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

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
	bitBuffer         []byte
	bitIndex          int
	chipIndex         int // Current chip index within symbol
	sampleIndexInChip int // Current sample index within chip
	carrierPhase      float32
}

// NewModulator creates a new DSSS modulator with unified API
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in symbols/sec
// codes: spreading codes (varargs)
//   - 1 code: Binary DSSS (modulation order = 2, 1 bit per symbol)
//   - M codes: M-ary DSSS (modulation order = M, log2(M) bits per symbol)
//     M must be a power of 2 (2, 4, 8, 16, 32, 64, etc.)
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
//	modulator := NewModulator(48000, 1000, 1000, code)
//
// Example - 4-ary DSSS:
//
//	codeTable := GenerateWalshCodeTable(4) // 4 orthogonal codes
//	modulator := NewModulator(48000, 1000, 1000, codeTable[0], codeTable[1], codeTable[2], codeTable[3])
//
// Example - 8-ary DSSS with unique codes:
//
//	codeTable := GetUniqueSymbolCodeTable("m-sequence", 8, 15, true) // 8 unique codes
//	modulator := NewModulator(48000, 1000, 1000, codeTable...)
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
	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS modulator: samplesPerChip must be positive, got %d (sampleRate=%f, chipRate=%f)", samplesPerChip, sampleRate, chipRate))
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
		bitBuffer:        make([]byte, 0, 1024),
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

// NewMModulator creates a new DSSS modulator for M-ary DSSS
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in symbols/sec
// codeTable: table of spreading codes (one per symbol)
func NewMModulator(sampleRate, carrierFreq, symbolRate float32, codeTable CodeTable) *Modulator {
	if len(codeTable) == 0 {
		panic("DSSS modulator: codeTable must not be empty")
	}
	if codeTable[0].length == 0 {
		panic("DSSS modulator: codes in codeTable must not be empty")
	}

	// Verify all codes have same length
	codeLength := codeTable[0].length
	for i, code := range codeTable {
		if code.length != codeLength {
			panic(fmt.Sprintf("DSSS modulator: code %d has length %d, expected %d", i, code.length, codeLength))
		}
	}

	spreadingFactor := codeLength
	chipRate := symbolRate * float32(spreadingFactor)
	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS modulator: samplesPerChip must be positive, got %d", samplesPerChip))
	}

	return &Modulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		spreadingFactor:  spreadingFactor,
		chipsPerSymbol:   spreadingFactor,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: spreadingFactor * samplesPerChip,
		amplitude:        1.0,
		codeTable:        codeTable,
		modulationOrder:  len(codeTable),
		bitBuffer:        make([]byte, 0, 1024),
		carrierPhase:     0,
	}
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
	dsss.bitBuffer = dsss.bitBuffer[:0]
	dsss.bitIndex = 0
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
		bitsNeeded := len(dsss.bitBuffer) + len(data)*8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * dsss.samplesPerSymbol
	}

	// Convert bytes to bits
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			dsss.bitBuffer = append(dsss.bitBuffer, byte(bit))
		}
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	samplesWritten := 0

	// Process bits into symbols and generate signal
	bitsPerSymbol := int(math.Log2(float64(dsss.modulationOrder)))
	for len(dsss.bitBuffer) >= bitsPerSymbol && samplesWritten < len(ampData) {
		// Extract bits for symbol
		bits := dsss.bitBuffer[:bitsPerSymbol]
		dsss.bitBuffer = dsss.bitBuffer[bitsPerSymbol:]

		// Convert bits to symbol index
		symbolIndex := 0
		for i, bit := range bits {
			if bit == 1 {
				symbolIndex |= 1 << uint(bitsPerSymbol-1-i)
			}
		}

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
