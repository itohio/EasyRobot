package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// FourB5BDecoder implements 4b/5b decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// 4b/5b decodes 5-bit symbols back to 4-bit nibbles.
// Each 2 input bytes (2 symbols, 10 bits) produce 1 output byte (2 nibbles).
// Input size: output_size * 2 (2 bytes per output byte)
type FourB5BDecoder struct {
	decodeTable [32]byte // Lookup table for 5-bit to 4-bit mapping
	buffer      []byte   // Buffer for partial symbols
}

// Build 4b/5b reverse lookup table (5-bit -> 4-bit)
func buildFourB5BDecodeTable() [32]byte {
	var table [32]byte
	// Initialize with invalid marker (0xFF)
	for i := range table {
		table[i] = 0xFF
	}
	// Fill valid mappings
	for nibble := byte(0); nibble < 16; nibble++ {
		symbol := fourB5BTable[nibble]
		table[symbol] = nibble
	}
	return table
}

var fourB5BDecodeTable = buildFourB5BDecodeTable()

// NewFourB5BDecoder creates a new 4b/5b decoder.
func NewFourB5BDecoder() *FourB5BDecoder {
	return &FourB5BDecoder{
		decodeTable: fourB5BDecodeTable,
	}
}

// Process processes 4b/5b-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each 2 input bytes (2 symbols) produce 1 output byte (2 nibbles).
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (f *FourB5BDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := len(f.buffer)
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
		// Each 2 input bytes produce 1 output byte
		return totalRead / 2
	}

	// Process bytes incrementally
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

		// Combine buffered data with new chunk
		combinedSize := len(f.buffer) + read
		if cap(f.buffer) < combinedSize {
			combined := make([]byte, combinedSize, combinedSize*2)
			copy(combined, f.buffer)
			copy(combined[len(f.buffer):], srcBuf[:read])
			f.buffer = combined
		} else {
			// Reuse buffer capacity
			oldLen := len(f.buffer)
			f.buffer = f.buffer[:combinedSize]
			copy(f.buffer[oldLen:], srcBuf[:read])
		}

		// Process complete pairs (2 bytes = 2 symbols -> 1 output byte)
		processedIdx := 0
		decodedBuf := make([]byte, len(f.buffer)/2)
		decodedIdx := 0

		for processedIdx+2 <= len(f.buffer) {
			// Get two 5-bit symbols
			symbol1 := f.buffer[processedIdx] & 0x1F // Lower 5 bits
			symbol2 := f.buffer[processedIdx+1] & 0x1F

			// Decode symbols to nibbles
			nibble1 := f.decodeTable[symbol1]
			nibble2 := f.decodeTable[symbol2]

			// Check for invalid symbols
			if nibble1 == 0xFF || nibble2 == 0xFF {
				// Invalid symbol - skip this pair
				processedIdx += 2
				continue
			}

			// Combine nibbles into byte
			outputByte := nibble1 | (nibble2 << 4)

			if decodedIdx < len(decodedBuf) {
				decodedBuf[decodedIdx] = outputByte
				decodedIdx++
			}
			processedIdx += 2
		}

		// Write decoded bytes
		if decodedIdx > 0 {
			_, writeErr := dst.Write(decodedBuf[:decodedIdx])
			if writeErr != nil {
				// Save remaining data
				f.buffer = f.buffer[processedIdx:]
				return totalWritten
			}
			totalWritten += decodedIdx
		}

		// Save any remaining partial pair
		if processedIdx < len(f.buffer) {
			f.buffer = f.buffer[processedIdx:]
		} else {
			f.buffer = f.buffer[:0]
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}

	return totalWritten
}

// Reset resets the decoder state.
// Reset implements chain.Processor interface.
func (f *FourB5BDecoder) Reset() {
	f.buffer = f.buffer[:0]
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*FourB5BDecoder)(nil)

