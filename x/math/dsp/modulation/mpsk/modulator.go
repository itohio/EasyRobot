package mpsk

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Modulator handles Phase Shift Keying modulation
type Modulator struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	modulationOrder  int // 2 for BPSK, 4 for QPSK, 8 for 8-PSK, etc.
	samplesPerSymbol int
	amplitude        float32
}

// NewModulator creates a new PSK modulator
func NewModulator(sampleRate, carrierFreq, symbolRate float32, modulationOrder int) *Modulator {
	samplesPerSymbol := int(sampleRate / symbolRate)
	return &Modulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		modulationOrder:  modulationOrder,
		samplesPerSymbol: samplesPerSymbol,
		amplitude:       1.0, // Default amplitude
	}
}

// SetAmplitude sets the signal amplitude for modulation
// SetAmplitude sets the signal amplitude for modulation
//
// Deprecated: This method is not part of the Modulator interface.
func (psk *Modulator) SetAmplitude(amplitude float32) {
	psk.amplitude = amplitude
}

// GenerateSignal generates a PSK modulated signal from data bytes
//
// Deprecated: This method is not part of the Modulator interface.
func (psk *Modulator) GenerateSignal(data []byte, amplitude float32, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	bitsPerSymbol := int(math.Log2(float64(psk.modulationOrder)))
	phaseStep := 2 * math.Pi / float32(psk.modulationOrder)

	bitIndex := 0
	sampleIndex := 0

	// Convert bytes to bits
	bitBuffer := make([]byte, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	// Process bits in symbol groups
	for i := 0; i < len(bitBuffer)-bitsPerSymbol+1 && sampleIndex < length; i += bitsPerSymbol {
		// Collect bits for one symbol
		symbol := 0
		for j := 0; j < bitsPerSymbol && i+j < len(bitBuffer); j++ {
			symbol = (symbol << 1) | int(bitBuffer[i+j])
		}

		// Generate symbol samples
		phaseOffset := float32(symbol) * phaseStep
		for s := 0; s < psk.samplesPerSymbol && sampleIndex < length; s++ {
			phase := 2*math.Pi*psk.carrierFreq*float32(sampleIndex)/psk.sampleRate + phaseOffset
			resultData[sampleIndex] = amplitude * float32(math.Cos(float64(phase)))
			sampleIndex++
		}
		bitIndex++
	}

	return result
}

// ModulationOrder returns the modulation order
//
// Deprecated: This method is not part of the Modulator interface.
func (psk *Modulator) ModulationOrder() int {
	return psk.modulationOrder
}

// Encode encodes bytes into amplitude vector
// Implements modulation.Modulator interface
// Returns the number of amplitude samples written
// Returns expected amplitudes length if amplitudes is nil
func (psk *Modulator) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if amplitudes == nil {
		// Calculate expected length (no padding - encoder doesn't add padding)
		bitsPerSymbol := int(math.Log2(float64(psk.modulationOrder)))
		bitsNeeded := len(data) * 8
		symbolsNeeded := (bitsNeeded + bitsPerSymbol - 1) / bitsPerSymbol
		return symbolsNeeded * psk.samplesPerSymbol
	}

	if len(data) == 0 {
		return 0
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	// Generate signal (no padding - encoder doesn't add padding)
	signal := psk.GenerateSignal(data, psk.amplitude, len(ampData))
	signalData := []float32(signal)

	// Copy signal to output
	signalCopyLen := len(signalData)
	if signalCopyLen > len(ampData) {
		signalCopyLen = len(ampData)
	}
	copy(ampData, signalData[:signalCopyLen])

	return signalCopyLen
}

// Reset resets the modulator state
// Implements modulation.Modulator interface
func (psk *Modulator) Reset() {
	// Modulator is stateless, nothing to reset
}

