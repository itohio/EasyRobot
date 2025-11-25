package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// CRCDecoder implements CRC (Cyclic Redundancy Check) decoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
// CRC provides error detection but not error correction.
// Decodes data by validating CRC and returning data only (CRC removed).
type CRCDecoder struct {
	polynomial uint32
	initial    uint32
	xorOut     uint32
	width      int // CRC width in bits (8, 16, or 32)
	table      []uint32
	// Buffers for incremental processing
	crcBuf    []byte // Buffer for CRC bytes (last crcBytes)
	dataBuf   []byte // Buffer for data bytes (for CRC calculation)
	combinedBuf []byte // Reusable buffer for combining crcBuf + new chunk
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

// Process processes CRC-encoded bytes to original data bytes (data only, CRC removed).
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Writes data bytes as soon as they are available, but validates CRC at the end.
// If dst is nil, calculates and returns the theoretical maximum output size.
// If CRC validation fails, returns 0 bytes written (but data may have already been written).
// Process implements chain.Processor[byte, byte] interface.
func (c *CRCDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	// Small buffer for reading chunks - enough for data bytes
	// CRC bytes are small (1, 2, or 4 bytes), so we buffer them separately
	var srcBuf [256]byte
	// Clear buffers at start (they're reused from previous calls)
	c.dataBuf = c.dataBuf[:0]
	c.crcBuf = c.crcBuf[:0]
	totalRead := 0
	totalWritten := 0
	var readErr error
	crcBytes := c.width / 8

	// If dst is nil, calculate and return size
	if dst == nil {
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
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
		}
		if totalRead < crcBytes {
			return 0
		}
		return totalRead - crcBytes
	}

	// Process incrementally - write data bytes as we read them
	// We need to buffer the last crcBytes to extract and validate CRC
	// Strategy: always keep the last crcBytes buffered, write everything else as data
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

		// Process the chunk
		chunk := srcBuf[:n]
		
		// Combine buffered CRC bytes with new chunk
		// The buffered bytes might be data (if more data is coming) or CRC (if this is the end)
		combinedSize := len(c.crcBuf) + len(chunk)
		if cap(c.combinedBuf) < combinedSize {
			c.combinedBuf = make([]byte, combinedSize, combinedSize*2)
		}
		combined := c.combinedBuf[:combinedSize]
		copy(combined, c.crcBuf)
		copy(combined[len(c.crcBuf):], chunk)
		c.crcBuf = c.crcBuf[:0]

		// If we have enough bytes, the first (len(combined) - crcBytes) are definitely data
		// The last crcBytes might be CRC (if EOF) or data (if more data is coming)
		if len(combined) > crcBytes {
			// We can write some bytes as data
			dataPart := combined[:len(combined)-crcBytes]
			
			// Write data part immediately
			_, writeErr := dst.Write(dataPart)
			if writeErr != nil {
				return totalWritten
			}
			totalWritten += len(dataPart)
			c.dataBuf = append(c.dataBuf, dataPart...)

			// Buffer the last crcBytes (might be CRC or might be data if more chunks come)
			c.crcBuf = append(c.crcBuf, combined[len(combined)-crcBytes:]...)
		} else {
			// Not enough bytes yet - buffer everything (might all be CRC or might be partial)
			c.crcBuf = append(c.crcBuf, combined...)
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}

	// After reading all data, validate CRC
	if totalRead < crcBytes {
		return 0
	}

	// Extract CRC from buffered CRC bytes
	if len(c.crcBuf) != crcBytes {
		// Something went wrong - we should have exactly crcBytes
		return 0
	}

	receivedCRC := c.extractCRC(c.crcBuf)

	// Calculate expected CRC from all data bytes we wrote
	expectedCRC := c.calculateCRC(c.dataBuf)

	// Validate CRC
	if receivedCRC != expectedCRC {
		// CRC mismatch - data is corrupted
		// Note: We've already written the data, but return 0 to indicate failure
		return 0
	}

	return totalWritten
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
// Reset implements chain.Processor interface.
func (c *CRCDecoder) Reset() {
	c.crcBuf = c.crcBuf[:0]
	c.dataBuf = c.dataBuf[:0]
	c.combinedBuf = c.combinedBuf[:0]
}

// Width returns the CRC width in bits.
func (c *CRCDecoder) Width() int {
	return c.width
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*CRCDecoder)(nil)
