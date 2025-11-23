package mask

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Modulator handles Amplitude Shift Keying modulation for digital communication.
// Supports various ASK constellations (2-ASK, 4-ASK, 8-ASK, etc.).
type Modulator struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	modulationOrder  int // 2 for 2-ASK, 4 for 4-ASK, 8 for 8-ASK, etc.
	samplesPerSymbol int
	amplitude        float32

	// State for streaming encoding
	bitBuffer    []byte
	bitIndex     int
	currentPhase float32
}

// NewModulator creates a new ASK modulator.
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in baud
// modulationOrder: number of amplitude levels (2, 4, 8, 16, etc.)
// amplitude: maximum amplitude for modulation
func NewModulator(sampleRate, carrierFreq, symbolRate float32, modulationOrder int, amplitude float32) *Modulator {
	samplesPerSymbol := int(sampleRate / symbolRate)
	return &Modulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		modulationOrder:  modulationOrder,
		samplesPerSymbol: samplesPerSymbol,
		amplitude:        amplitude,
		bitBuffer:        make([]byte, 0, 1024),
	}
}

// Reset resets the modulator state.
func (mask *Modulator) Reset() {
	mask.bitBuffer = mask.bitBuffer[:0]
	mask.bitIndex = 0
	mask.currentPhase = 0
}

// Encode encodes bytes into amplitude vector.
// Implements modulation.Modulator interface.
// Returns the number of amplitude samples written.
// Returns expected amplitudes length if amplitudes is nil.
func (mask *Modulator) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if amplitudes == nil {
		// Calculate expected length
		bitsPerSymbol := int(math.Log2(float64(mask.modulationOrder)))
		bitsNeeded := len(mask.bitBuffer) + len(data)*8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * mask.samplesPerSymbol
	}

	// Convert bytes to bits
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			mask.bitBuffer = append(mask.bitBuffer, byte(bit))
		}
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)
	samplesWritten := 0

	bitsPerSymbol := int(math.Log2(float64(mask.modulationOrder)))

	// Process bits in symbol groups
	for len(mask.bitBuffer) >= bitsPerSymbol && samplesWritten < len(ampData) {
		// Extract bits for one symbol
		symbol := 0
		for i := 0; i < bitsPerSymbol; i++ {
			symbol = (symbol << 1) | int(mask.bitBuffer[i])
		}
		mask.bitBuffer = mask.bitBuffer[bitsPerSymbol:]

		// Map symbol to amplitude level
		amplitudeLevel := mask.symbolToAmplitude(symbol)

		// Generate symbol samples
		for s := 0; s < mask.samplesPerSymbol && samplesWritten < len(ampData); s++ {
			phase := 2 * math.Pi * mask.carrierFreq * mask.currentPhase / mask.sampleRate
			ampData[samplesWritten] = amplitudeLevel * float32(math.Cos(float64(phase)))
			mask.currentPhase += 1.0
			samplesWritten++
		}
	}

	return samplesWritten
}

// symbolToAmplitude maps a symbol value to an amplitude level.
// For M-ASK, maps symbol to one of M amplitude levels.
func (mask *Modulator) symbolToAmplitude(symbol int) float32 {
	// Map symbol (0 to modulationOrder-1) to amplitude levels
	// For 2-ASK: 0 -> 0, 1 -> amplitude
	// For 4-ASK: 0 -> 0, 1 -> amplitude/3, 2 -> 2*amplitude/3, 3 -> amplitude
	// For 8-ASK: 0 -> 0, 1 -> amplitude/7, ..., 7 -> amplitude
	if mask.modulationOrder == 2 {
		// On-Off Keying (OOK)
		if symbol == 0 {
			return 0
		}
		return mask.amplitude
	}

	// Multi-level ASK: evenly spaced amplitude levels
	if symbol == 0 {
		return 0
	}
	return mask.amplitude * float32(symbol) / float32(mask.modulationOrder-1)
}

// ModulationOrder returns the modulation order.
// ModulationOrder returns the modulation order
//
// Deprecated: This method is not part of the Modulator interface.
func (mask *Modulator) ModulationOrder() int {
	return mask.modulationOrder
}

// GenerateSignal generates an ASK modulated signal from data bytes.
// This is a convenience method for generating a complete signal.
//
// Deprecated: This method is not part of the Modulator interface.
func (mask *Modulator) GenerateSignal(data []byte, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	bitsPerSymbol := int(math.Log2(float64(mask.modulationOrder)))
	bitBuffer := make([]byte, 0, len(data)*8)

	// Convert bytes to bits
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	sampleIndex := 0
	mask.currentPhase = 0

	// Process bits in symbol groups
	for i := 0; i < len(bitBuffer)-bitsPerSymbol+1 && sampleIndex < length; i += bitsPerSymbol {
		// Collect bits for one symbol
		symbol := 0
		for j := 0; j < bitsPerSymbol && i+j < len(bitBuffer); j++ {
			symbol = (symbol << 1) | int(bitBuffer[i+j])
		}

		// Map symbol to amplitude
		amplitudeLevel := mask.symbolToAmplitude(symbol)

		// Generate symbol samples
		for s := 0; s < mask.samplesPerSymbol && sampleIndex < length; s++ {
			phase := 2 * math.Pi * mask.carrierFreq * mask.currentPhase / mask.sampleRate
			resultData[sampleIndex] = amplitudeLevel * float32(math.Cos(float64(phase)))
			mask.currentPhase += 1.0
			sampleIndex++
		}
	}

	return result
}

