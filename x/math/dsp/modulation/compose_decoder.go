package modulation

import (
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// ComposeDecoder chains a Demodulator before a ByteDecoder.
// Decoding flow: amplitudes -> Demodulator -> bytes -> ByteDecoder -> bytes
// This allows combining signal demodulation (e.g., BFSK) with line decoding (e.g., Manchester).
type ComposeDecoder struct {
	demodulator Demodulator
	byteDecoder ByteDecoder
	buffer      []byte
}

// NewComposeDecoder creates a composed decoder that applies
// a Demodulator (e.g., BFSK) before a ByteDecoder (e.g., Manchester).
func NewComposeDecoder(demodulator Demodulator, byteDecoder ByteDecoder) *ComposeDecoder {
	return &ComposeDecoder{
		demodulator: demodulator,
		byteDecoder: byteDecoder,
		buffer:      make([]byte, 0, 4096),
	}
}

// Decode decodes amplitudes through the demodulator, then through the byte decoder.
// Decode implements Demodulator interface
// Returns the number of bytes written to data.
// Returns expected data length if data is nil.
func (c *ComposeDecoder) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		// First estimate demodulator output size
		demodulatorOutputSize := c.demodulator.Decode(nil, amplitudes)
		if demodulatorOutputSize == 0 {
			return 0
		}
		// Then estimate byte decoder output size
		// Use dummy data of the right size (fill with pattern to avoid all zeros)
		tempBuffer := make([]byte, demodulatorOutputSize)
		for i := range tempBuffer {
			tempBuffer[i] = byte(i % 256)
		}
		return c.byteDecoder.Decode(nil, tempBuffer)
	}

	if amplitudes == nil {
		return 0
	}

	// First, decode through demodulator to bytes
	// We need an intermediate buffer to hold the demodulator-decoded bytes before byte decoding
	// Don't rely on size estimation - demodulator might not be locked yet
	// Allocate a reasonable buffer size (demodulator output is typically smaller than input samples)
	ampVec := amplitudes.(vecTypes.Vector)
	estimatedSize := ampVec.Len() / 8 // Conservative estimate: 1 byte per 8 samples
	if estimatedSize < 1024 {
		estimatedSize = 1024 // Minimum buffer size
	}
	if len(c.buffer) < estimatedSize {
		c.buffer = make([]byte, estimatedSize)
	}

	// Decode amplitudes to bytes using demodulator
	// The demodulator will lock onto the signal during this call
	bytesFromDemodulator := c.demodulator.Decode(c.buffer, amplitudes)
	if bytesFromDemodulator == 0 {
		return 0
	}

	// Then decode through byte decoder
	return c.byteDecoder.Decode(data, c.buffer[:bytesFromDemodulator])
}

// Reset resets both decoders.
// Reset implements Demodulator interface
func (c *ComposeDecoder) Reset() {
	c.demodulator.Reset()
	c.byteDecoder.Reset()
}
