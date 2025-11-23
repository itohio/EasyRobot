package ecc

import (
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

// CRCEncoder implements CRC (Cyclic Redundancy Check) encoding at the byte level.
// It implements the modulation.ByteEncoder interface for composability.
// CRC provides error detection but not error correction.
type CRCEncoder struct {
	polynomial uint32
	initial    uint32
	xorOut     uint32
	width      int // CRC width in bits (8, 16, or 32)
	table      []uint32
}

// CRC polynomial constants
const (
	// CRC-8
	CRC8Polynomial = 0x07
	CRC8Initial    = 0x00
	CRC8XOROut     = 0x00
	CRC8Width      = 8
	CRC8TableSize  = 256

	// CRC-16 (CCITT)
	CRC16Polynomial = 0x1021
	CRC16Initial    = 0xFFFF
	CRC16XOROut     = 0x0000
	CRC16Width      = 16
	CRC16TableSize  = 256

	// CRC-32 (IEEE 802.3)
	CRC32Polynomial = 0x04C11DB7
	CRC32Initial    = 0xFFFFFFFF
	CRC32XOROut     = 0xFFFFFFFF
	CRC32Width      = 32
	CRC32TableSize  = 256
)

// NewCRC8Encoder creates a new CRC-8 encoder.
func NewCRC8Encoder() *CRCEncoder {
	return NewCRCEncoder(CRC8Polynomial, CRC8Initial, CRC8XOROut, CRC8Width)
}

// NewCRC16Encoder creates a new CRC-16 encoder.
func NewCRC16Encoder() *CRCEncoder {
	return NewCRCEncoder(CRC16Polynomial, CRC16Initial, CRC16XOROut, CRC16Width)
}

// NewCRC32Encoder creates a new CRC-32 encoder.
func NewCRC32Encoder() *CRCEncoder {
	return NewCRCEncoder(CRC32Polynomial, CRC32Initial, CRC32XOROut, CRC32Width)
}

// NewCRCEncoder creates a new CRC encoder with custom parameters.
func NewCRCEncoder(polynomial, initial, xorOut uint32, width int) *CRCEncoder {
	if width != 8 && width != 16 && width != 32 {
		panic("CRC width must be 8, 16, or 32")
	}

	encoder := &CRCEncoder{
		polynomial: polynomial,
		initial:    initial,
		xorOut:     xorOut,
		width:      width,
	}

	// Generate lookup table
	encoder.table = encoder.generateTable()

	return encoder
}

// generateTable generates the CRC lookup table for fast computation.
func (c *CRCEncoder) generateTable() []uint32 {
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

// Encode encodes input bytes to CRC-encoded bytes (data + CRC).
// Returns the number of bytes written to dst.
// Returns expected dst length if dst is nil.
// Encode implements modulation.ByteEncoder interface.
func (c *CRCEncoder) Encode(dst, src []byte) int {
	if dst == nil {
		// Calculate expected output length: data + CRC bytes
		crcBytes := c.width / 8
		return len(src) + crcBytes
	}

	// Allow encoding empty data (will produce only CRC bytes)

	crcBytes := c.width / 8
	requiredLen := len(src) + crcBytes
	if len(dst) < requiredLen {
		return 0
	}

	// Copy data
	copy(dst, src)

	// Calculate CRC
	crc := c.calculateCRC(src)

	// Append CRC bytes (little-endian)
	for i := 0; i < crcBytes; i++ {
		dst[len(src)+i] = byte(crc >> (i * 8))
	}

	return requiredLen
}

// calculateCRC calculates the CRC value for the given data.
func (c *CRCEncoder) calculateCRC(data []byte) uint32 {
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

// Reset resets the encoder state.
// Reset implements modulation.ByteEncoder interface.
func (c *CRCEncoder) Reset() {
	// No state to reset (stateless)
}

// Width returns the CRC width in bits.
func (c *CRCEncoder) Width() int {
	return c.width
}

// Verify interface compliance
var _ modulation.ByteEncoder = (*CRCEncoder)(nil)
