package qam

import (
	"math"

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
	bitBuffer        []byte
	bitIndex         int
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
		bitBuffer:        make([]byte, 0, 1024),
	}
}

// Reset resets the encoder state
func (qam *Encoder) Reset() {
	qam.bitBuffer = qam.bitBuffer[:0]
	qam.bitIndex = 0
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
		bitsNeeded := len(qam.bitBuffer) + len(data)*8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * qam.samplesPerSymbol
	}

	// Convert bytes to bits
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			qam.bitBuffer = append(qam.bitBuffer, byte(bit))
		}
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	samplesWritten := 0

	// Process 4 bits at a time (16-QAM symbols)
	bitsPerSymbol := 4
	for len(qam.bitBuffer) >= bitsPerSymbol && samplesWritten < len(ampData) {
		// Extract 4 bits
		bits := qam.bitBuffer[:bitsPerSymbol]
		qam.bitBuffer = qam.bitBuffer[bitsPerSymbol:]

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
func (qam *Encoder) ModulationOrder() int {
	return qam.modulationOrder
}

// GenerateSignal generates a QAM modulated signal from data bytes.
// This is a convenience method for generating a complete signal.
func (qam *Encoder) GenerateSignal(data []byte, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	// Convert bytes to bits
	bitBuffer := make([]byte, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	sampleIndex := 0
	qam.currentPhase = 0

	// Process 4 bits at a time (16-QAM symbols)
	bitsPerSymbol := 4
	for i := 0; i < len(bitBuffer)-bitsPerSymbol+1 && sampleIndex < length; i += bitsPerSymbol {
		bits := bitBuffer[i : i+bitsPerSymbol]

		// Map 4 bits to I/Q values
		iVal, qVal := qam.bitsToIQ(bits)
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

