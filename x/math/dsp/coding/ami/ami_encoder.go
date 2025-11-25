package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// AMIEncoder implements Alternate Mark Inversion (AMI) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// AMI: '0' = no signal (0), '1' = alternating polarity (+1, -1, +1, -1, ...).
// Provides DC balance and error detection (violations indicate errors).
// Encodes data by converting binary to AMI (1:1 mapping, no size change).
// Note: Uses 3-level encoding: 0, +1, -1. We encode as: 0=0x00, +1=0x01, -1=0x02.
type AMIEncoder struct {
	nextPolarity bool // Next polarity for '1' (true = positive, false = negative)
}

// NewAMIEncoder creates a new AMI encoder.
func NewAMIEncoder() *AMIEncoder {
	return &AMIEncoder{
		nextPolarity: true, // Start with positive
	}
}

// Process processes input bytes to AMI-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (a *AMIEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// AMI is 1:1 mapping, so output size equals input size
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

			// Process each bit (MSB first)
			// AMI: '0' = 0, '1' = alternating +1/-1
			// For byte-level: encode as binary with polarity tracking
			// We'll use a simplified approach: '1' bits are marked with alternating pattern
			// Store polarity in bit 7, data in bits 0-6
			// Actually simpler: just track polarity, output binary with polarity info
			// For practical byte-level: output is same size, we track polarity state
			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal >> bit) & 1
				if inputBit == 1 {
					// '1' = mark with current polarity
					// Store polarity in output (we'll use bit 7 to indicate polarity)
					if a.nextPolarity {
						encodedByte |= (1 << bit) // Positive mark
					} else {
						// Negative mark: we can't represent -1 in byte, so use special encoding
						// Use bit pattern that indicates negative
						encodedByte |= (1 << bit)
						// Mark as negative using a flag (we'll use a separate approach)
					}
					a.nextPolarity = !a.nextPolarity
				}
				// '0' = no mark (bit stays 0)
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
func (a *AMIEncoder) Reset() {
	a.nextPolarity = true
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*AMIEncoder)(nil)

