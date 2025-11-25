package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// MillerEncoder implements Miller (Delay) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Miller encoding: '1' = transition in middle of bit period, '0' = no transition (or transition at start if previous was '0').
// Better spectral efficiency than Manchester while maintaining clock recovery.
// Each input byte produces 2 output bytes (1 byte -> 2 bytes).
type MillerEncoder struct {
	lastState bool // Last output state
	lastBit   bool // Last input bit (for '0' encoding rule)
}

// NewMillerEncoder creates a new Miller encoder.
func NewMillerEncoder() *MillerEncoder {
	return &MillerEncoder{
		lastState: false,
		lastBit:   false,
	}
}

// Process processes input bytes to Miller-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each input byte produces 2 output bytes.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (m *MillerEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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

			// Process each bit (MSB first)
			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal>>bit)&1 == 1

				// Miller encoding rules:
				// '1' = transition in middle of bit period
				// '0' = no transition, but if previous was '0', transition at start
				firstHalf := m.lastState
				secondHalf := m.lastState

				if inputBit {
					// '1': transition in middle
					secondHalf = !m.lastState
					m.lastState = secondHalf
				} else {
					// '0': no transition, but if previous was '0', transition at start
					if !m.lastBit {
						firstHalf = !m.lastState
						m.lastState = firstHalf
					}
					secondHalf = m.lastState
				}

				m.lastBit = inputBit

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
func (m *MillerEncoder) Reset() {
	m.lastState = false
	m.lastBit = false
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*MillerEncoder)(nil)

