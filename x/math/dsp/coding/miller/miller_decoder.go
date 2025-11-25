package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// MillerDecoder implements Miller (Delay) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes Miller-encoded bytes back to original bytes.
// Each 2 input bytes produce 1 output byte (2 bytes -> 1 byte).
type MillerDecoder struct {
	lastState bool   // Last input state
	lastBit   bool   // Last decoded bit
	buffer    []byte // Buffer for partial bytes
}

// NewMillerDecoder creates a new Miller decoder.
func NewMillerDecoder() *MillerDecoder {
	return &MillerDecoder{
		lastState: false,
		lastBit:   false,
	}
}

// Process processes Miller-encoded bytes to original bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each 2 input bytes produce 1 output byte.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (m *MillerDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := len(m.buffer)
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
		return (totalRead + 1) / 2
	}

	// Process bytes incrementally
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

		// Combine buffered data with new chunk
		combinedSize := len(m.buffer) + read
		if cap(m.buffer) < combinedSize {
			combined := make([]byte, combinedSize, combinedSize*2)
			copy(combined, m.buffer)
			copy(combined[len(m.buffer):], srcBuf[:read])
			m.buffer = combined
		} else {
			oldLen := len(m.buffer)
			m.buffer = m.buffer[:combinedSize]
			copy(m.buffer[oldLen:], srcBuf[:read])
		}

		// Process complete pairs
		processedIdx := 0
		decodedBuf := make([]byte, len(m.buffer)/2)
		decodedIdx := 0

		for processedIdx+2 <= len(m.buffer) {
			byte1 := m.buffer[processedIdx]
			byte2 := m.buffer[processedIdx+1]
			var decodedByte byte

			for period := 0; period < 8; period++ {
				bitIdx := period * 2
				byteIdx := bitIdx / 8
				bitPos := 7 - (bitIdx % 8)

				var firstHalf, secondHalf bool
				if byteIdx == 0 {
					firstHalf = (byte1>>bitPos)&1 == 1
					bitPos--
					if bitPos < 0 {
						bitPos = 7
						byteIdx = 1
					}
					secondHalf = (byte1>>bitPos)&1 == 1
				} else {
					firstHalf = (byte2>>bitPos)&1 == 1
					bitPos--
					if bitPos < 0 {
						bitPos = 7
					}
					secondHalf = (byte2>>bitPos)&1 == 1
				}

				// Detect transition in middle = '1'
				// No transition in middle = '0'
				middleTransition := secondHalf != firstHalf
				decodedBit := byte(0)
				if middleTransition {
					decodedBit = 1
				}

				decodedByte |= (decodedBit << (7 - period))
				m.lastState = secondHalf
				m.lastBit = decodedBit == 1
			}

			if decodedIdx < len(decodedBuf) {
				decodedBuf[decodedIdx] = decodedByte
				decodedIdx++
			}
			processedIdx += 2
		}

		if decodedIdx > 0 {
			_, writeErr := dst.Write(decodedBuf[:decodedIdx])
			if writeErr != nil {
				m.buffer = m.buffer[processedIdx:]
				return totalWritten
			}
			totalWritten += decodedIdx
		}

		if processedIdx < len(m.buffer) {
			m.buffer = m.buffer[processedIdx:]
		} else {
			m.buffer = m.buffer[:0]
		}

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
func (m *MillerDecoder) Reset() {
	m.lastState = false
	m.lastBit = false
	m.buffer = m.buffer[:0]
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*MillerDecoder)(nil)

