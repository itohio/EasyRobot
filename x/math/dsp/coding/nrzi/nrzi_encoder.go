package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// NRZIEncoder implements Non-Return-to-Zero Inverted (NRZI) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// NRZI encodes transitions: '0' = no transition, '1' = transition.
// This provides robustness to polarity inversion and simple clock recovery.
// Encodes data by converting binary values to NRZI (1:1 mapping, no size change).
type NRZIEncoder struct {
	lastState bool // Last output state (true = high, false = low)
}

// NewNRZIEncoder creates a new NRZI encoder.
func NewNRZIEncoder() *NRZIEncoder {
	return &NRZIEncoder{
		lastState: false, // Start with low state
	}
}

// Process processes input bytes to NRZI-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (n *NRZIEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		encodedBuf := make([]byte, read)
		for i := range read {
			byteVal := srcBuf[i]
			var encodedByte byte

			// Process each bit (MSB first)
			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal >> bit) & 1
				if inputBit == 1 {
					// '1' = transition: flip state
					n.lastState = !n.lastState
				}
				// '0' = no transition: keep state
				// Output is the current state
				if n.lastState {
					encodedByte |= (1 << bit)
				}
			}

			encodedBuf[i] = encodedByte
		}

		// Write encoded bytes
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
func (n *NRZIEncoder) Reset() {
	n.lastState = false // Reset to low state
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*NRZIEncoder)(nil)

