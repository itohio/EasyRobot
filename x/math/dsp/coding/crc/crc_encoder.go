package crc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// CRCEncoder implements CRC (Cyclic Redundancy Check) encoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
// CRC provides error detection but not error correction.
// Encodes data by appending CRC bytes (data + CRC).
type Encoder struct {
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

// NewEncoder creates a new CRC-32 encoder (default, most common).
func NewEncoder() *Encoder {
	return NewEncoderWith(CRC32Polynomial, CRC32Initial, CRC32XOROut, CRC32Width)
}

// NewEncoderCRC8 creates a new CRC-8 encoder.
func NewEncoderCRC8() *Encoder {
	return NewEncoderWith(CRC8Polynomial, CRC8Initial, CRC8XOROut, CRC8Width)
}

// NewEncoderCRC16 creates a new CRC-16 encoder.
func NewEncoderCRC16() *Encoder {
	return NewEncoderWith(CRC16Polynomial, CRC16Initial, CRC16XOROut, CRC16Width)
}

// NewEncoderCRC32 creates a new CRC-32 encoder.
func NewEncoderCRC32() *Encoder {
	return NewEncoderWith(CRC32Polynomial, CRC32Initial, CRC32XOROut, CRC32Width)
}

// NewEncoderWith creates a new CRC encoder with custom parameters.
func NewEncoderWith(polynomial, initial, xorOut uint32, width int) *Encoder {
	if width != 8 && width != 16 && width != 32 {
		panic("CRC width must be 8, 16, or 32")
	}

	encoder := &Encoder{
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
func (c *Encoder) generateTable() []uint32 {
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

// Process processes input bytes to CRC-encoded bytes (data + CRC).
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (c *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	var dataBuf []byte // Buffer to accumulate all data for CRC calculation
	totalRead := 0
	var readErr error
	crcBytes := c.width / 8

	// Read all available data from source
	for readErr == nil {
		n, err := src.Read(srcBuf[:])
		if n == 0 {
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
			continue
		}

		totalRead += n

		// If dst is nil, calculate and return size
		if dst == nil {
			if err == io.EOF {
				break
			}
			continue
		}

		// Buffer data for CRC calculation
		dataBuf = append(dataBuf, srcBuf[:n]...)

		// Write data to destination
		_, writeErr := dst.Write(srcBuf[:n])
		if writeErr != nil {
			// Return bytes written so far (data only, CRC not appended yet)
			return totalRead
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}

	// If dst was nil, return calculated size
	if dst == nil {
		return totalRead + crcBytes
	}

	// Calculate and append CRC if we read any data
	if len(dataBuf) > 0 {
		// Calculate CRC
		crc := c.calculateCRC(dataBuf)

		// Append CRC bytes (little-endian)
		crcData := make([]byte, crcBytes)
		for i := 0; i < crcBytes; i++ {
			crcData[i] = byte(crc >> (i * 8))
		}
		_, _ = dst.Write(crcData)
	}

	return totalRead + crcBytes
}

// calculateCRC calculates the CRC value for the given data.
func (c *Encoder) calculateCRC(data []byte) uint32 {
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
// Reset implements chain.Processor interface.
func (c *Encoder) Reset() {
	// No state to reset (stateless)
}

// Width returns the CRC width in bits.
func (c *Encoder) Width() int {
	return c.width
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)
