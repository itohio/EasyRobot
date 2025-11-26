package hdb3

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// HDB3Encoder implements High Density Bipolar 3 (HDB3) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// HDB3 is an AMI variant that limits consecutive zeros to maximum of 3.
// Used in E1 lines (European telecom).
// Encodes data by converting binary to HDB3 (1:1 mapping, no size change).
type Encoder struct {
	nextPolarity bool // Next polarity for marks
	zeroCount    int  // Consecutive zero count
	lastViolation bool // Last violation type (B or V)
}

// NewEncoder creates a new HDB3 encoder.
func NewEncoder() *Encoder {
	return &Encoder{
		nextPolarity: true,
		zeroCount:    0,
		lastViolation: false,
	}
}

// Process processes input bytes to HDB3-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (h *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
					if h.nextPolarity {
						encodedByte |= (1 << bit)
					}
					h.nextPolarity = !h.nextPolarity
					h.zeroCount = 0
				} else {
					// '0' = no mark
					h.zeroCount++
					// HDB3: if 4 consecutive zeros, insert violation
					if h.zeroCount == 4 {
						// Insert B00V or 000V pattern based on last violation
						// Simplified: insert mark (violation)
						if h.lastViolation {
							// B00V pattern
							encodedByte |= (1 << bit) // B mark
							h.nextPolarity = !h.nextPolarity
						}
						// V mark (violation)
						if h.nextPolarity {
							encodedByte |= (1 << (bit - 1)) // V mark (simplified)
						}
						h.nextPolarity = !h.nextPolarity
						h.zeroCount = 0
						h.lastViolation = !h.lastViolation
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
func (h *Encoder) Reset() {
	h.nextPolarity = true
	h.zeroCount = 0
	h.lastViolation = false
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)

