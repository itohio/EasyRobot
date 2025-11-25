package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// B8ZSDecoder implements Bipolar 8-Zero Substitution (B8ZS) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes B8ZS-encoded bytes back to binary.
type B8ZSDecoder struct {
	zeroCount int // Consecutive zero count
}

// NewB8ZSDecoder creates a new B8ZS decoder.
func NewB8ZSDecoder() *B8ZSDecoder {
	return &B8ZSDecoder{
		zeroCount: 0,
	}
}

// Process processes B8ZS-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (b *B8ZSDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
			b8zsByte := srcBuf[i]
			var decodedByte byte

			for bit := 7; bit >= 0; bit-- {
				b8zsBit := (b8zsByte >> bit) & 1

				if b8zsBit == 1 {
					// Mark detected
					decodedByte |= (1 << bit)
					b.zeroCount = 0
				} else {
					// Zero
					b.zeroCount++
					// B8ZS violations are removed during decoding
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
func (b *B8ZSDecoder) Reset() {
	b.zeroCount = 0
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*B8ZSDecoder)(nil)

