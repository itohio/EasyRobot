package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// AMIDecoder implements Alternate Mark Inversion (AMI) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes AMI-encoded bytes back to binary.
// Note: Simplified implementation - full AMI requires 3-level detection.
type AMIDecoder struct {
	lastPolarity bool // Last detected polarity
}

// NewAMIDecoder creates a new AMI decoder.
func NewAMIDecoder() *AMIDecoder {
	return &AMIDecoder{
		lastPolarity: true,
	}
}

// Process processes AMI-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (a *AMIDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
			amiByte := srcBuf[i]
			var decodedByte byte

			// Decode: any non-zero bit = '1', zero = '0'
			// Simplified: detect marks (non-zero) as '1'
			for bit := 7; bit >= 0; bit-- {
				amiBit := (amiByte >> bit) & 1
				if amiBit == 1 {
					decodedByte |= (1 << bit)
				}
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
func (a *AMIDecoder) Reset() {
	a.lastPolarity = true
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*AMIDecoder)(nil)

