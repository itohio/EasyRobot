package ppm

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// PPMDecoder implements Pulse Position Modulation (PPM) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes PPM-encoded bytes back to original bytes.
type Decoder struct {
	positions int // Number of positions per symbol (must match encoder)
}

// NewPPMDecoder creates a new 4-PPM decoder.
func NewDecoder() *Decoder {
	return NewDecoderWithPositions(4)
}

// NewDecoderWithPositions creates a new PPM decoder with specified positions.
func NewDecoderWithPositions(positions int) *Decoder {
	if positions != 2 && positions != 4 && positions != 8 {
		panic("PPM positions must be 2, 4, or 8")
	}
	return &Decoder{
		positions: positions,
	}
}

// Process processes PPM-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (p *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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

		decodedBuf := make([]byte, read)
		for i := range read {
			ppmByte := srcBuf[i]
			var decodedByte byte

			// Decode position markers back to 2-bit symbols
			for bitPair := 0; bitPair < 4; bitPair++ {
				bitStart := 7 - (bitPair * 2)
				position := (ppmByte >> (bitPair * 2)) & 0x03
				decodedByte |= (position << bitStart)
			}

			decodedBuf[i] = decodedByte
		}

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
func (p *Decoder) Reset() {
	// No state to reset
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)

