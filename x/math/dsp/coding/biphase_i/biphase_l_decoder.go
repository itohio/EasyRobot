package biphase_i

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// BiphaseLDecoder implements Biphase-L decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes Biphase-L-encoded bytes back to original bytes.
// Each 2 input bytes produce 1 output byte (2 bytes -> 1 byte).
type Decoder struct {
	lastState bool   // Last input state
	buffer    []byte // Buffer for partial bytes
}

// NewBiphaseLDecoder creates a new Biphase-L decoder.
func NewDecoder() *Decoder {
	return &Decoder{
		lastState: false,
	}
}

// Process processes Biphase-L-encoded bytes to original bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Each 2 input bytes produce 1 output byte.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (b *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := len(b.buffer)
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

	// Process bytes incrementally (similar to Differential Manchester)
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
		combinedSize := len(b.buffer) + read
		if cap(b.buffer) < combinedSize {
			combined := make([]byte, combinedSize, combinedSize*2)
			copy(combined, b.buffer)
			copy(combined[len(b.buffer):], srcBuf[:read])
			b.buffer = combined
		} else {
			oldLen := len(b.buffer)
			b.buffer = b.buffer[:combinedSize]
			copy(b.buffer[oldLen:], srcBuf[:read])
		}

		// Process complete pairs
		processedIdx := 0
		decodedBuf := make([]byte, len(b.buffer)/2)
		decodedIdx := 0

		for processedIdx+2 <= len(b.buffer) {
			byte1 := b.buffer[processedIdx]
			byte2 := b.buffer[processedIdx+1]
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

				// Detect transition in middle = '0', no transition = '1'
				middleTransition := secondHalf != firstHalf
				decodedBit := byte(1)
				if middleTransition {
					decodedBit = 0
				}

				decodedByte |= (decodedBit << (7 - period))
				b.lastState = secondHalf
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
				b.buffer = b.buffer[processedIdx:]
				return totalWritten
			}
			totalWritten += decodedIdx
		}

		if processedIdx < len(b.buffer) {
			b.buffer = b.buffer[processedIdx:]
		} else {
			b.buffer = b.buffer[:0]
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
func (b *Decoder) Reset() {
	b.lastState = false
	b.buffer = b.buffer[:0]
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)
