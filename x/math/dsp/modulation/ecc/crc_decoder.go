package ecc

import (
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

// CRCDecoder implements CRC (Cyclic Redundancy Check) decoding at the byte level.
// It implements the modulation.ByteDecoder interface for composability.
// CRC provides error detection but not error correction.
type CRCDecoder struct {
	polynomial uint32
	initial    uint32
	xorOut     uint32
	width      int // CRC width in bits (8, 16, or 32)
	table      []uint32
}

// NewCRC8Decoder creates a new CRC-8 decoder.
func NewCRC8Decoder() *CRCDecoder {
	return NewCRCDecoder(CRC8Polynomial, CRC8Initial, CRC8XOROut, CRC8Width)
}

// NewCRC16Decoder creates a new CRC-16 decoder.
func NewCRC16Decoder() *CRCDecoder {
	return NewCRCDecoder(CRC16Polynomial, CRC16Initial, CRC16XOROut, CRC16Width)
}

// NewCRC32Decoder creates a new CRC-32 decoder.
func NewCRC32Decoder() *CRCDecoder {
	return NewCRCDecoder(CRC32Polynomial, CRC32Initial, CRC32XOROut, CRC32Width)
}

// NewCRCDecoder creates a new CRC decoder with custom parameters.
func NewCRCDecoder(polynomial, initial, xorOut uint32, width int) *CRCDecoder {
	if width != 8 && width != 16 && width != 32 {
		panic("CRC width must be 8, 16, or 32")
	}

	decoder := &CRCDecoder{
		polynomial: polynomial,
		initial:    initial,
		xorOut:     xorOut,
		width:      width,
	}

	// Generate lookup table
	decoder.table = decoder.generateTable()

	return decoder
}

// generateTable generates the CRC lookup table for fast computation.
func (c *CRCDecoder) generateTable() []uint32 {
	table := make([]uint32, CRC32TableSize)
	mask := uint32((1 << c.width) - 1)
	if c.width == 32 {
		mask = 0xFFFFFFFF
	}

	// Use BCE-friendly reslice
	table = table[:CRC32TableSize]
	for i := range table {
		crc := uint32(i)
		// Unroll inner loop for better performance (8 iterations known at compile time)
		// Bit 0
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 1
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 2
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 3
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 4
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 5
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 6
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		// Bit 7
		if (crc & 1) != 0 {
			crc = (crc >> 1) ^ c.polynomial
		} else {
			crc >>= 1
		}
		table[i] = crc & mask
	}

	return table
}

// Decode decodes CRC-encoded bytes to original data bytes (data only, CRC removed).
// Returns the number of bytes written to dst.
// Returns expected dst length if dst is nil.
// If CRC validation fails, returns 0 bytes written.
// Decode implements modulation.ByteDecoder interface.
func (c *CRCDecoder) Decode(dst, src []byte) int {
	if dst == nil {
		// Calculate expected output length: data only (CRC removed)
		crcBytes := c.width / 8
		if len(src) < crcBytes {
			return 0
		}
		return len(src) - crcBytes
	}

	crcBytes := c.width / 8
	if len(src) < crcBytes {
		return 0
	}

	dataLen := len(src) - crcBytes
	if len(dst) < dataLen {
		return 0
	}

	// Extract data and CRC
	data := src[:dataLen]
	receivedCRC := c.extractCRC(src[dataLen:])

	// Calculate expected CRC
	expectedCRC := c.calculateCRC(data)

	// Validate CRC
	if receivedCRC != expectedCRC {
		// CRC mismatch - data is corrupted
		return 0
	}

	// Copy data to dst
	copy(dst, data)
	return dataLen
}

// extractCRC extracts the CRC value from the CRC bytes (little-endian).
func (c *CRCDecoder) extractCRC(crcBytes []byte) uint32 {
	var crc uint32
	for i, b := range crcBytes {
		crc |= uint32(b) << (i * 8)
	}
	mask := uint32((1 << c.width) - 1)
	if c.width == 32 {
		mask = 0xFFFFFFFF
	}
	return crc & mask
}

// calculateCRC calculates the CRC value for the given data.
func (c *CRCDecoder) calculateCRC(data []byte) uint32 {
	crc := c.initial
	mask := uint32((1 << c.width) - 1)
	if c.width == 32 {
		mask = 0xFFFFFFFF
	}

	for _, b := range data {
		idx := byte((crc ^ uint32(b)) & 0xFF)
		crc = (crc >> 8) ^ c.table[idx]
	}

	return (crc ^ c.xorOut) & mask
}

// Reset resets the decoder state.
// Reset implements modulation.ByteDecoder interface.
func (c *CRCDecoder) Reset() {
	// No state to reset (stateless)
}

// Width returns the CRC width in bits.
func (c *CRCDecoder) Width() int {
	return c.width
}

// Verify interface compliance
var _ modulation.ByteDecoder = (*CRCDecoder)(nil)
