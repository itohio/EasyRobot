package ppm

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// PPMEncoder implements Pulse Position Modulation (PPM) encoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// PPM encodes data in pulse timing/position within a time slot.
// Common variants: 2-PPM, 4-PPM, 8-PPM (2, 4, or 8 positions per symbol).
// This implementation uses 4-PPM: 2 bits per symbol, 4 possible positions.
// Each input byte (4 symbols) produces variable output (typically 4-8 bytes depending on implementation).
// Simplified: encodes as position markers (1:1 mapping for simplicity).
type Encoder struct {
	positions int // Number of positions per symbol (2, 4, or 8)
}

// NewEncoder creates a new 4-PPM encoder.
func NewEncoder() *Encoder {
	return NewEncoderWithPositions(4)
}

// NewEncoderWithPositions creates a new PPM encoder with specified positions.
func NewEncoderWithPositions(positions int) *Encoder {
	if positions != 2 && positions != 4 && positions != 8 {
		panic("PPM positions must be 2, 4, or 8")
	}
	return &Encoder{
		positions: positions,
	}
}

// Process processes input bytes to PPM-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
// Note: Full PPM would require timing information. This is a simplified byte-level implementation.
func (p *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// Simplified: 1:1 mapping (full PPM would expand)
		return totalRead
	}

	// Simplified PPM: encode position as byte value
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

			// For 4-PPM: each 2-bit pair = position 0-3
			// Encode as position markers
			for bitPair := 0; bitPair < 4; bitPair++ {
				bitStart := 7 - (bitPair * 2)
				symbol := (byteVal >> bitStart) & 0x03
				// Position encoded in lower 2 bits
				encodedByte |= (symbol << (bitPair * 2))
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
func (p *Encoder) Reset() {
	// No state to reset
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)

