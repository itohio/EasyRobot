package nrzi

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Decoder implements Non-Return-to-Zero Inverted (NRZI) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// NRZI decodes transitions: transition = '1', no transition = '0'.
// Decodes data by converting NRZI values back to binary (1:1 mapping, no size change).
type Decoder struct {
	lastState bool // Last input state (for detecting transitions)
}

// NewDecoder creates a new NRZI decoder.
func NewDecoder() *Decoder {
	return &Decoder{
		lastState: false, // Start with low state
	}
}

// Process processes NRZI-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (n *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// NRZI is 1:1 mapping, so output size equals input size
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

		// Process each byte bit by bit
		decodedBuf := make([]byte, read)
		for i := range read {
			nrziByte := srcBuf[i]
			var decodedByte byte

			// Process each bit (MSB first)
			for bit := 7; bit >= 0; bit-- {
				currentState := (nrziByte>>bit)&1 == 1
				if currentState != n.lastState {
					// Transition detected = '1'
					decodedByte |= (1 << bit)
				}
				// No transition = '0' (bit already 0)
				n.lastState = currentState
			}

			decodedBuf[i] = decodedByte
		}

		// Write decoded bytes
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
func (n *Decoder) Reset() {
	n.lastState = false // Reset to low state
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)

