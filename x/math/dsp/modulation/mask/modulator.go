package mask

import (
	"math"

	"github.com/itohio/EasyRobot/x/buffer"
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
	bitBuffer    buffer.Bits[byte]
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
		bitBuffer:        buffer.NewBits[byte](int(math.Log2(float64(modulationOrder)))), // bits per symbol
	}
}

// Reset resets the modulator state.
func (mask *Modulator) Reset() {
	mask.bitBuffer.Clear()
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
		bitsNeeded := mask.bitBuffer.GetWrittenBits() + len(data)*8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * mask.samplesPerSymbol
	}

	// Add bytes to bit buffer
	mask.bitBuffer.ReadFromBytes(data)

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)
	samplesWritten := 0

	// Process symbols
	for mask.bitBuffer.HasValue() && samplesWritten < len(ampData) {
		// Read one symbol
		symbolVal, ok := mask.bitBuffer.Read()
		if !ok {
			break
		}
		symbol := int(symbolVal)

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
	bitBuffer := buffer.NewBits[byte](bitsPerSymbol)
	bitBuffer.ReadFromBytes(data)
	bitBuffer.Reset()

	sampleIndex := 0
	mask.currentPhase = 0

	// Process symbols
	for bitBuffer.HasValue() && sampleIndex < length {
		// Read one symbol
		symbolVal, ok := bitBuffer.Read()
		if !ok {
			break
		}
		symbol := int(symbolVal)

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
