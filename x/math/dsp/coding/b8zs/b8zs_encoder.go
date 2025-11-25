package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// B8ZSEncoder implements Bipolar 8-Zero Substitution (B8ZS) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// B8ZS is an AMI variant that substitutes 8 consecutive zeros with a special pattern.
// Used in T1 lines (North American telecom).
// Encodes data by converting binary to B8ZS (1:1 mapping, no size change).
type B8ZSEncoder struct {
	nextPolarity bool // Next polarity for marks
	zeroCount    int  // Consecutive zero count
}

// NewB8ZSEncoder creates a new B8ZS encoder.
func NewB8ZSEncoder() *B8ZSEncoder {
	return &B8ZSEncoder{
		nextPolarity: true,
		zeroCount:    0,
	}
}

// Process processes input bytes to B8ZS-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (b *B8ZSEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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

		encodedBuf := make([]byte, read)
		for i := range read {
			byteVal := srcBuf[i]
			var encodedByte byte

			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal >> bit) & 1

				if inputBit == 1 {
					// '1' = mark with alternating polarity
					if b.nextPolarity {
						encodedByte |= (1 << bit)
					}
					b.nextPolarity = !b.nextPolarity
					b.zeroCount = 0
				} else {
					// '0' = no mark
					b.zeroCount++
					// B8ZS: if 8 consecutive zeros, substitute with 000VB0VB pattern
					if b.zeroCount == 8 {
						// Insert violation pattern (simplified)
						encodedByte |= (1 << bit) // V mark
						b.nextPolarity = !b.nextPolarity
						b.zeroCount = 0
					}
				}
			}

			encodedBuf[i] = encodedByte
		}

		_, writeErr := dst.Write(encodedBuf)
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

// Reset resets the encoder state.
// Reset implements chain.Processor interface.
func (b *B8ZSEncoder) Reset() {
	b.nextPolarity = true
	b.zeroCount = 0
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*B8ZSEncoder)(nil)

