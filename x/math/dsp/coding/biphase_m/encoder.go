package diff_manchester

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Encoder implements Differential Manchester (Biphase-M) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Differential Manchester: transition at start of bit period, '0' = transition in middle, '1' = no transition in middle.
// This combines differential encoding with Manchester clock recovery.
// Each input byte produces 2 output bytes (1 byte -> 2 bytes, same as Manchester).
type Encoder struct {
	lastState bool // Last output state (for transitions)
}

// NewEncoder creates a new Differential Manchester encoder.
func NewEncoder() *Encoder {
	return &Encoder{
		lastState: false, // Start with low state
	}
}

// Process processes input bytes to Differential Manchester-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each input byte produces 2 output bytes.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (d *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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

		// Each input byte produces 2 output bytes (16 bits for 8 bits)
		encodedBuf := make([]byte, read*2)
		encodedIdx := 0

		for i := range read {
			byteVal := srcBuf[i]

			// Process each bit (MSB first)
			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal >> bit) & 1

				// Always transition at start of bit period
				d.lastState = !d.lastState
				firstHalf := d.lastState

				// '0' = transition in middle, '1' = no transition in middle
				if inputBit == 0 {
					d.lastState = !d.lastState
				}
				secondHalf := d.lastState

				// Pack two bits into output (each bit period = 2 output bits)
				// We need to pack 8 bit periods (16 bits) into 2 bytes
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
func (d *Encoder) Reset() {
	d.lastState = false
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)
