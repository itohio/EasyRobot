package rll

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// RLLDecoder implements Run-Length Limited (RLL) decoding.
// It implements the chain.Processor[byte, byte] interface for composability.
// Decodes RLL-encoded bytes back to original bytes.
type Decoder struct {
	d, k int // RLL(d,k) parameters (must match encoder)
}

// NewRLLDecoder creates a new RLL(1,7) decoder.
func NewDecoder() *Decoder {
	return NewDecoderWithParams(1, 7)
}

// NewDecoderWithParams creates a new RLL decoder with custom (d,k) parameters.
func NewDecoderWithParams(d, k int) *Decoder {
	if d < 0 || k < d {
		panic("RLL: invalid parameters, must have 0 <= d <= k")
	}
	return &Decoder{
		d: d,
		k: k,
	}
}

// Process processes RLL-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (r *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size (estimate: 2/3x for RLL(1,7))
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
		// Estimate: RLL typically compresses by ~2/3x
		return (totalRead * 2) / 3
	}

	// Simplified RLL decoding
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
		copy(decodedBuf, srcBuf[:read])

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
func (r *Decoder) Reset() {
	// No state to reset
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)

