package modulation

import (
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// ComposeEncoder chains a ByteEncoder before a Modulator.
// Encoding flow: bytes -> ByteEncoder -> bytes -> Modulator -> amplitudes
// This allows combining line coding (e.g., Manchester) with signal modulation (e.g., BFSK).
type ComposeEncoder struct {
	byteEncoder ByteEncoder
	modulator   Modulator
	buffer      []byte
}

// NewComposeEncoder creates a composed encoder that applies
// a ByteEncoder (e.g., Manchester) before a Modulator (e.g., BFSK).
func NewComposeEncoder(byteEncoder ByteEncoder, modulator Modulator) *ComposeEncoder {
	return &ComposeEncoder{
		byteEncoder: byteEncoder,
		modulator:   modulator,
		buffer:      make([]byte, 0, 4096),
	}
}

// Encode encodes bytes through the byte encoder, then through the modulator.
// Encode implements Modulator interface
// Returns the number of amplitude samples written.
// Returns expected amplitudes length if amplitudes is nil.
func (c *ComposeEncoder) Encode(amplitudes vecTypes.Vector, data []byte) int {
	if len(data) == 0 {
		return 0
	}

	// First, encode through byte encoder
	// Estimate byte-encoded size (conservative: assume 2x for Manchester-like encoders)
	estimatedByteSize := len(data) * 2
	if len(c.buffer) < estimatedByteSize {
		c.buffer = make([]byte, estimatedByteSize)
	}

	bytesWritten := c.byteEncoder.Encode(c.buffer, data)
	if bytesWritten == 0 {
		return 0
	}

	// Then encode through modulator
	if amplitudes == nil {
		// Calculate expected length for modulator
		return c.modulator.Encode(nil, c.buffer[:bytesWritten])
	}

	return c.modulator.Encode(amplitudes, c.buffer[:bytesWritten])
}

// Reset resets both encoders.
// Reset implements Modulator interface
func (c *ComposeEncoder) Reset() {
	c.byteEncoder.Reset()
	c.modulator.Reset()
}

