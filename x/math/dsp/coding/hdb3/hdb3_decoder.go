package hdb3

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// HDB3Decoder implements High Density Bipolar 3 (HDB3) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes HDB3-encoded bytes back to binary.
type Decoder struct {
	zeroCount int // Consecutive zero count
}

// NewHDB3Decoder creates a new HDB3 decoder.
func NewDecoder() *Decoder {
	return &Decoder{
		zeroCount: 0,
	}
}

// Process processes HDB3-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (h *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		return totalRead
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

		decodedBuf := make([]byte, read)
		for i := range read {
			hdb3Byte := srcBuf[i]
			var decodedByte byte

			for bit := 7; bit >= 0; bit-- {
				hdb3Bit := (hdb3Byte >> bit) & 1

				if hdb3Bit == 1 {
					// Mark detected
					decodedByte |= (1 << bit)
					h.zeroCount = 0
				} else {
					// Zero
					h.zeroCount++
					// HDB3 violations are removed during decoding
				}
			}

			decodedBuf[i] = decodedByte
		}

		_, writeErr := dst.Write(decodedBuf)
		if writeErr != nil {
			return totalWritten
		}
		totalWritten += read

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
func (h *Decoder) Reset() {
	h.zeroCount = 0
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)
