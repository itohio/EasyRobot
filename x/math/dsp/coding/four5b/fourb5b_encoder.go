package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// FourB5BEncoder implements 4b/5b encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// 4b/5b maps 4-bit nibbles to 5-bit symbols for DC balance and clock recovery.
// Each input byte (2 nibbles) produces 2 output bytes (2 symbols, 10 bits total).
// Output size: input_size * 1.25 (5 bits per 4 bits = 1.25x)
type FourB5BEncoder struct {
	encodeTable [16]byte // Lookup table for 4-bit to 5-bit mapping
}

// 4b/5b encoding table (standard Fast Ethernet mapping)
// Maps 4-bit value (0-15) to 5-bit symbol
// Ensures no more than 3 consecutive zeros for clock recovery
var fourB5BTable = [16]byte{
	0b11110, // 0x0 -> 0x1E (0000 -> 11110)
	0b01001, // 0x1 -> 0x09 (0001 -> 01001)
	0b10100, // 0x2 -> 0x14 (0010 -> 10100)
	0b10101, // 0x3 -> 0x15 (0011 -> 10101)
	0b01010, // 0x4 -> 0x0A (0100 -> 01010)
	0b01011, // 0x5 -> 0x0B (0101 -> 01011)
	0b01110, // 0x6 -> 0x0E (0110 -> 01110)
	0b01111, // 0x7 -> 0x0F (0111 -> 01111)
	0b10010, // 0x8 -> 0x12 (1000 -> 10010)
	0b10011, // 0x9 -> 0x13 (1001 -> 10011)
	0b10110, // 0xA -> 0x16 (1010 -> 10110)
	0b10111, // 0xB -> 0x17 (1011 -> 10111)
	0b11010, // 0xC -> 0x1A (1100 -> 11010)
	0b11011, // 0xD -> 0x1B (1101 -> 11011)
	0b11100, // 0xE -> 0x1C (1110 -> 11100)
	0b11101, // 0xF -> 0x1D (1111 -> 11101)
}

// NewFourB5BEncoder creates a new 4b/5b encoder.
func NewFourB5BEncoder() *FourB5BEncoder {
	return &FourB5BEncoder{
		encodeTable: fourB5BTable,
	}
}

// Process processes input bytes to 4b/5b-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each input byte (2 nibbles) produces 2 output bytes (10 bits total, stored in 2 bytes).
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (f *FourB5BEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := 0
		for readErr == nil {
			read, err := src.Read(srcBuf[:])
			if read == 0 {
				if err == io.EOF {
					break
				}
				if err != nil {
					break
				}
				continue
			}
			totalRead += read
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
		}
		// Each input byte produces 2 output bytes (10 bits in 2 bytes)
		return totalRead * 2
	}

	// Process bytes
	for readErr == nil {
		read, err := src.Read(srcBuf[:])
		if read == 0 {
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
			continue
		}

		// Each input byte produces 2 output bytes
		encodedBuf := make([]byte, read*2)
		encodedIdx := 0

		for i := range read {
			byteVal := srcBuf[i]

			// Process high nibble (bits 4-7)
			highNibble := (byteVal >> 4) & 0x0F
			high5b := f.encodeTable[highNibble]

			// Process low nibble (bits 0-3)
			lowNibble := byteVal & 0x0F
			low5b := f.encodeTable[lowNibble]

			// Pack 10 bits into 2 bytes
			// First byte: high 5 bits of high5b (3 bits) + low 2 bits of low5b (2 bits) = 5 bits
			// Actually, we need to pack: high5b (5 bits) + low5b (5 bits) = 10 bits
			// Byte 0: bits 7-3 = high5b[4:0], bits 2-0 = low5b[4:2]
			// Byte 1: bits 7-5 = low5b[1:0] + padding, but this is messy
			// Better: store each 5-bit symbol in a byte (bit 7 unused, bits 6-5 unused)
			encodedBuf[encodedIdx] = high5b
			encodedBuf[encodedIdx+1] = low5b
			encodedIdx += 2
		}

		// Write encoded bytes
		_, writeErr := dst.Write(encodedBuf[:encodedIdx])
		if writeErr != nil {
			return totalWritten
		}
		totalWritten += encodedIdx

		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}

	return totalWritten
}

// Reset resets the encoder state.
// Reset implements chain.Processor interface.
func (f *FourB5BEncoder) Reset() {
	// No state to reset (stateless)
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*FourB5BEncoder)(nil)

