package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// EightB10BDecoder implements 8b/10b decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// 8b/10b decodes 10-bit symbols back to 8-bit bytes.
// Each 2 input bytes (10 bits) produce 1 output byte.
// Input size: output_size * 2 (2 bytes per output byte)
type EightB10BDecoder struct {
	decodeTable [1024]byte // Lookup table for 10-bit to 8-bit mapping
	buffer      []byte     // Buffer for partial symbols
}

// NewEightB10BDecoder creates a new 8b/10b decoder.
func NewEightB10BDecoder() *EightB10BDecoder {
	return &EightB10BDecoder{
		decodeTable: eightB10BDecodeTable,
	}
}

// Process processes 8b/10b-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each 2 input bytes (10 bits) produce 1 output byte.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (e *EightB10BDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := len(e.buffer)
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
		combinedSize := len(e.buffer) + read
		if cap(e.buffer) < combinedSize {
			combined := make([]byte, combinedSize, combinedSize*2)
			copy(combined, e.buffer)
			copy(combined[len(e.buffer):], srcBuf[:read])
			e.buffer = combined
		} else {
			oldLen := len(e.buffer)
			e.buffer = e.buffer[:combinedSize]
			copy(e.buffer[oldLen:], srcBuf[:read])
		}

		// Process complete pairs (2 bytes = 10-bit symbol -> 1 output byte)
		processedIdx := 0
		decodedBuf := make([]byte, len(e.buffer)/2)
		decodedIdx := 0

		for processedIdx+2 <= len(e.buffer) {
			// Extract 10-bit symbol from 2 bytes
			// First byte: bits 7-0, Second byte: bits 9-8 (in bits 7-6)
			byte1 := e.buffer[processedIdx]
			byte2 := e.buffer[processedIdx+1]
			symbol := (uint16(byte1) << 2) | (uint16(byte2) >> 6)

			// Decode symbol to byte
			if symbol < 1024 {
				decodedByte := e.decodeTable[symbol]
				if decodedByte != 0xFF {
					if decodedIdx < len(decodedBuf) {
						decodedBuf[decodedIdx] = decodedByte
						decodedIdx++
					}
				}
			}
			processedIdx += 2
		}

		// Write decoded bytes
		if decodedIdx > 0 {
			_, writeErr := dst.Write(decodedBuf[:decodedIdx])
			if writeErr != nil {
				e.buffer = e.buffer[processedIdx:]
				return totalWritten
			}
			totalWritten += decodedIdx
		}

		// Save any remaining partial pair
		if processedIdx < len(e.buffer) {
			e.buffer = e.buffer[processedIdx:]
		} else {
			e.buffer = e.buffer[:0]
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
func (e *EightB10BDecoder) Reset() {
	e.buffer = e.buffer[:0]
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*EightB10BDecoder)(nil)

