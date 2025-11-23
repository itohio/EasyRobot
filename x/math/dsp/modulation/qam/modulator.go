package qam

import (
	"math"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Encoder handles 16-QAM modulation (4 bits per symbol)
// Implements modulation.Modulator interface.
type Encoder struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	modulationOrder  int // 16 for 16-QAM
	samplesPerSymbol int
	amplitude        float32
	currentPhase     float32
	bitBuffer        buffer.Bits[byte]
}

// NewEncoder creates a new 16-QAM encoder.
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in baud
// amplitude: maximum amplitude for modulation
func NewEncoder(sampleRate, carrierFreq, symbolRate, amplitude float32) *Encoder {
	samplesPerSymbol := int(sampleRate / symbolRate)
	return &Encoder{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		modulationOrder:  16, // 16-QAM
		samplesPerSymbol: samplesPerSymbol,
		amplitude:        amplitude,
		bitBuffer:        buffer.NewBits[byte](4), // 4 bits per symbol for 16-QAM
	}
}

// Reset resets the encoder state
func (qam *Encoder) Reset() {
	qam.bitBuffer.Clear()
	qam.currentPhase = 0
}

// Encode encodes bytes into amplitude vector.
// Implements modulation.Modulator interface.
// Returns the number of amplitude samples written.
// Returns expected amplitudes length if amplitudes is nil.
func (qam *Encoder) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if amplitudes == nil {
		// Calculate expected length
		bitsPerSymbol := 4 // 16-QAM: 4 bits per symbol
		bitsNeeded := qam.bitBuffer.GetWrittenBits() + len(data)*8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * qam.samplesPerSymbol
	}

	// Add bytes to bit buffer
	qam.bitBuffer.ReadFromBytes(data)

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	samplesWritten := 0

	// Process symbols (4 bits per symbol for 16-QAM)
	for qam.bitBuffer.HasValue() && samplesWritten < len(ampData) {
		// Read one symbol (4 bits)
		symbol, ok := qam.bitBuffer.Read()
		if !ok {
			break
		}

		// Convert symbol to 4 bits for I/Q mapping
		bits := []byte{
			byte((symbol >> 3) & 1),
			byte((symbol >> 2) & 1),
			byte((symbol >> 1) & 1),
			byte(symbol & 1),
		}

		// Map 4 bits to I/Q values (-3, -1, 1, 3)
		iVal, qVal := qam.bitsToIQ(bits)
		// Scale by amplitude
		iVal *= qam.amplitude / 3.0
		qVal *= qam.amplitude / 3.0

		// Generate symbol samples
		for s := 0; s < qam.samplesPerSymbol && samplesWritten < len(ampData); s++ {
			phase := 2 * math.Pi * qam.carrierFreq * qam.currentPhase / qam.sampleRate
			carrierI := float32(math.Cos(float64(phase)))
			carrierQ := float32(math.Sin(float64(phase)))

			// Modulate: I*cos - Q*sin (standard QAM modulation)
			ampData[samplesWritten] = iVal*carrierI - qVal*carrierQ

			qam.currentPhase += 1.0
			samplesWritten++
		}
	}

	return samplesWritten
}

// bitsToIQ maps 4 bits to I/Q values for 16-QAM.
// 16-QAM constellation: 4 levels per dimension
// Map bits [b3 b2 b1 b0] to I/Q coordinates
func (qam *Encoder) bitsToIQ(bits []byte) (float32, float32) {
	iBits := (bits[0] << 1) | bits[1] // bits 0,1 for I
	qBits := (bits[2] << 1) | bits[3] // bits 2,3 for Q

	// Map 2-bit values to amplitude levels: 00->-3, 01->-1, 10->1, 11->3
	iVal := float32(iBits*2 - 3)
	qVal := float32(qBits*2 - 3)

	return iVal, qVal
}

// ModulationOrder returns the modulation order.
// ModulationOrder returns the modulation order
//
// Deprecated: This method is not part of the Modulator interface.
func (qam *Encoder) ModulationOrder() int {
	return qam.modulationOrder
}

// GenerateSignal generates a QAM modulated signal from data bytes.
// This is a convenience method for generating a complete signal.
//
// Deprecated: This method is not part of the Modulator interface.
func (qam *Encoder) GenerateSignal(data []byte, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	// Use bit buffer to read symbols
	bitBuffer := buffer.NewBits[byte](4) // 4 bits per symbol for 16-QAM
	bitBuffer.ReadFromBytes(data)
	bitBuffer.Reset()

	sampleIndex := 0
	qam.currentPhase = 0

	// Process symbols
	for bitBuffer.HasValue() && sampleIndex < length {
		// Read one symbol (4 bits)
		symbol, ok := bitBuffer.Read()
		if !ok {
			break
		}

		// Convert symbol to 4 bits for I/Q mapping
		bits := []byte{
			byte((symbol >> 3) & 1),
			byte((symbol >> 2) & 1),
			byte((symbol >> 1) & 1),
			byte(symbol & 1),
		}

		// Map 4 bits to I/Q values
		iVal, qVal := qam.bitsToIQ(bits[:])
		iVal *= qam.amplitude / 3.0 // Normalize to amplitude
		qVal *= qam.amplitude / 3.0

		// Generate symbol samples
		for s := 0; s < qam.samplesPerSymbol && sampleIndex < length; s++ {
			phase := 2 * math.Pi * qam.carrierFreq * qam.currentPhase / qam.sampleRate
			carrierI := float32(math.Cos(float64(phase)))
			carrierQ := float32(math.Sin(float64(phase)))

			// Modulate: I*cos - Q*sin (standard QAM modulation)
			resultData[sampleIndex] = iVal*carrierI - qVal*carrierQ

			qam.currentPhase += 1.0
			sampleIndex++
		}
	}

	return result
}
