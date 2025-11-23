package mfsk

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Preamble is a known byte sequence used for synchronization
// Pattern: 0xAA, 0x55 (alternating bits: 10101010, 01010101)
// This creates an alternating pattern that's easy to detect
var Preamble = []byte{0xAA, 0x55, 0xAA, 0x55} // 4 bytes = 32 bits

// Modulator handles Frequency Shift Keying modulation
type Modulator struct {
	sampleRate       float32
	frequencies      []float32 // Frequencies for each symbol (2, 4, 8, or 16 frequencies)
	symbolRate       float32
	samplesPerSymbol int
	modulationOrder  int     // Number of frequencies (2, 4, 8, 16)
	amplitude        float32 // Signal amplitude for encoding
}

// NewModulator creates a new FSK modulator with 2 frequencies (binary FSK)
func NewModulator(sampleRate, freqLow, freqHigh, symbolRate float32) *Modulator {
	return NewMModulator(sampleRate, symbolRate, freqLow, freqHigh)
}

// NewMModulator creates a new M-FSK modulator with multiple frequencies
// modulationOrder is determined by the number of frequencies provided
func NewMModulator(sampleRate, symbolRate float32, frequencies ...float32) *Modulator {
	if len(frequencies) < 2 {
		panic("FSK encoder requires at least 2 frequencies")
	}
	// Validate modulation order is power of 2
	modOrder := len(frequencies)
	if modOrder != 2 && modOrder != 4 && modOrder != 8 && modOrder != 16 {
		panic(fmt.Sprintf("FSK modulation order must be 2, 4, 8, or 16, got %d", modOrder))
	}

	// For 8-FSK and 16-FSK, use 4x sample rate to improve frequency resolution
	// This gives 4x more samples per symbol, improving frequency discrimination
	effectiveSampleRate := sampleRate
	if modOrder >= 8 {
		effectiveSampleRate = sampleRate * 4.0
	}
	
	samplesPerSymbol := int(effectiveSampleRate / symbolRate)
	return &Modulator{
		sampleRate:       effectiveSampleRate, // Store effective sample rate
		frequencies:      frequencies,
		symbolRate:       symbolRate,
		samplesPerSymbol: samplesPerSymbol,
		modulationOrder:  modOrder,
		amplitude:        1.0, // Default amplitude
	}
}

// SetAmplitude sets the signal amplitude for modulation
// SetAmplitude sets the signal amplitude for modulation
//
// Deprecated: This method is not part of the Modulator interface.
func (fsk *Modulator) SetAmplitude(amplitude float32) {
	fsk.amplitude = amplitude
}

// GenerateSignal generates an FSK modulated signal from data bytes
//
// Deprecated: This method is not part of the Modulator interface.
// Adds preamble before data to allow signal detector to lock
func (fsk *Modulator) GenerateSignal(data []byte, amplitude float32, length int) vec.Vector {
	result := vec.New(length)
	resultData := []float32(result)

	bitsPerSymbol := int(math.Log2(float64(fsk.modulationOrder)))

	sampleIndex := 0
	phase := float32(0)

	// Convert data bytes to bits
	bitBuffer := make([]byte, 0, len(data)*8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	// Process data bits in symbol groups
	for i := 0; i < len(bitBuffer)-bitsPerSymbol+1 && sampleIndex < length; i += bitsPerSymbol {
		// Collect bits for one symbol
		symbol := 0
		for j := 0; j < bitsPerSymbol && i+j < len(bitBuffer); j++ {
			symbol = (symbol << 1) | int(bitBuffer[i+j])
		}

		// Select frequency based on symbol value
		if symbol >= len(fsk.frequencies) {
			symbol = len(fsk.frequencies) - 1
		}
		freq := fsk.frequencies[symbol]

		// Generate symbol samples
		for s := 0; s < fsk.samplesPerSymbol && sampleIndex < length; s++ {
			phase += 2 * math.Pi * freq / fsk.sampleRate
			resultData[sampleIndex] = amplitude * float32(math.Cos(float64(phase)))
			sampleIndex++
		}
	}

	// Handle leftover bits that don't form a complete symbol
	// Pad with zeros to form a complete symbol (same as mask modulator)
	remainingBits := len(bitBuffer) % bitsPerSymbol
	if remainingBits > 0 && sampleIndex < length {
		// Get the last remainingBits from bitBuffer
		lastBitsStart := len(bitBuffer) - remainingBits
		symbol := 0
		for j := 0; j < remainingBits; j++ {
			symbol = (symbol << 1) | int(bitBuffer[lastBitsStart+j])
		}
		// Pad remaining bits with zeros
		for j := remainingBits; j < bitsPerSymbol; j++ {
			symbol = symbol << 1
		}

		// Select frequency based on symbol value
		if symbol >= len(fsk.frequencies) {
			symbol = len(fsk.frequencies) - 1
		}
		freq := fsk.frequencies[symbol]

		// Generate symbol samples
		for s := 0; s < fsk.samplesPerSymbol && sampleIndex < length; s++ {
			phase += 2 * math.Pi * freq / fsk.sampleRate
			resultData[sampleIndex] = amplitude * float32(math.Cos(float64(phase)))
			sampleIndex++
		}
	}

	return result
}

// ModulationOrder returns the modulation order (number of frequencies)
//
// Deprecated: This method is not part of the Modulator interface.
func (fsk *Modulator) ModulationOrder() int {
	return fsk.modulationOrder
}

// Frequencies returns the frequencies used by the modulator
func (fsk *Modulator) Frequencies() []float32 {
	return fsk.frequencies
}

// Encode encodes bytes into amplitude vector
// Implements modulation.Modulator interface
// Returns the number of amplitude samples written
// Returns expected amplitudes length if amplitudes is nil
func (fsk *Modulator) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if amplitudes == nil {
		// Calculate expected length: preamble + data
		bitsPerSymbol := int(math.Log2(float64(fsk.modulationOrder)))
		preambleBits := len(Preamble) * 8
		dataBits := len(data) * 8
		totalBits := preambleBits + dataBits
		totalSymbols := (totalBits + bitsPerSymbol - 1) / bitsPerSymbol
		return totalSymbols * fsk.samplesPerSymbol
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	// Prepend preamble to data
	fullData := make([]byte, 0, len(Preamble)+len(data))
	fullData = append(fullData, Preamble...)
	fullData = append(fullData, data...)

	// Generate signal with preamble + data
	signal := fsk.GenerateSignal(fullData, fsk.amplitude, len(ampData))
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
func (fsk *Modulator) Reset() {
	// Modulator is stateless, nothing to reset
}
