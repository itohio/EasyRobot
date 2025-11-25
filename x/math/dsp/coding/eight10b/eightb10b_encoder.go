package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// EightB10BEncoder implements 8b/10b encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// 8b/10b maps 8-bit bytes to 10-bit symbols for DC balance and clock recovery.
// Each input byte produces 2 output bytes (10 bits stored in 2 bytes, 6 bits unused in second byte).
// Output size: input_size * 2 (2 bytes per input byte)
// Note: This is a simplified implementation without running disparity (RD) tracking.
// For full DC balance, RD tracking would be needed to alternate between positive and negative disparity symbols.
type EightB10BEncoder struct {
	encodeTable [256]uint16 // Lookup table for 8-bit to 10-bit mapping
}

// NewEightB10BEncoder creates a new 8b/10b encoder.
func NewEightB10BEncoder() *EightB10BEncoder {
	return &EightB10BEncoder{
		encodeTable: eightB10BEncodeTable,
	}
}

// Process processes input bytes to 8b/10b-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each input byte produces 2 output bytes (10 bits total, stored in 2 bytes).
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (e *EightB10BEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// Each input byte produces 2 output bytes
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
			symbol := e.encodeTable[byteVal] // 10-bit symbol

			// Pack 10 bits into 2 bytes
			// First byte: bits 9-2 (8 bits)
			// Second byte: bits 1-0 (2 bits) + 6 unused bits
			encodedBuf[encodedIdx] = byte(symbol >> 2)
			encodedBuf[encodedIdx+1] = byte((symbol & 0x03) << 6)
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
func (e *EightB10BEncoder) Reset() {
	// No state to reset (stateless, simplified version)
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*EightB10BEncoder)(nil)

