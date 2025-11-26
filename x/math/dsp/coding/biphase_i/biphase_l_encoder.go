package biphase

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// BiphaseLEncoder implements Biphase-L (Biphase Level) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Biphase-L: similar to Manchester but with different transition rules.
// '0' = transition at start and middle, '1' = transition only at start.
// Each input byte produces 2 output bytes (1 byte -> 2 bytes).
type Encoder struct {
	lastState bool // Last output state
}

// NewEncoder creates a new Biphase-L encoder.
func NewEncoder() *Encoder {
	return &Encoder{
		lastState: false,
	}
}

// Process processes input bytes to Biphase-L-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each input byte produces 2 output bytes.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (b *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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

		encodedBuf := make([]byte, read*2)
		encodedIdx := 0

		for i := range read {
			byteVal := srcBuf[i]

			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal >> bit) & 1

				// Always transition at start of bit period
				b.lastState = !b.lastState
				firstHalf := b.lastState

				// '0' = transition in middle, '1' = no transition in middle
				if inputBit == 0 {
					b.lastState = !b.lastState
				}
				secondHalf := b.lastState

				// Pack two bits into output
				bitPairIdx := (7 - bit) * 2
				byteIdx := bitPairIdx / 8
				bitPos := 7 - (bitPairIdx % 8)
				if firstHalf {
					encodedBuf[encodedIdx+byteIdx] |= (1 << bitPos)
				}
				bitPos--
				if bitPos < 0 {
					bitPos = 7
					byteIdx++
				}
				if secondHalf {
					encodedBuf[encodedIdx+byteIdx] |= (1 << bitPos)
				}
			}
			encodedIdx += 2
		}

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
func (b *Encoder) Reset() {
	b.lastState = false
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)
