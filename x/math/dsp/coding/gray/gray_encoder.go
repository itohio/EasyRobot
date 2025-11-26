package gray

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Encoder implements Gray code encoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
// Gray codes provide noise resilience by ensuring adjacent values differ by only one bit.
// This reduces the chance of multiple bit errors when a symbol is misdetected.
// Encodes data by converting binary values to Gray code (1:1 mapping, no size change).
type Encoder struct {
	bitsPerSymbol int // Number of bits per symbol (typically 4 for nibbles or 8 for bytes)
}

// NewEncoder creates a new Gray code encoder with default 8-bit encoding.
func NewEncoder() *Encoder {
	return NewEncoderWithBits(8)
}

// NewEncoderWithBits creates a new Gray code encoder with specified bits per symbol.
// bitsPerSymbol must be 1-8 (typically 4 for nibbles or 8 for bytes).
func NewEncoderWithBits(bitsPerSymbol int) *Encoder {
	if bitsPerSymbol < 1 || bitsPerSymbol > 8 {
		panic("Gray code bits per symbol must be 1-8")
	}
	return &Encoder{
		bitsPerSymbol: bitsPerSymbol,
	}
}

// encodeToGray converts a binary value to Gray code.
// Formula: gray = binary ^ (binary >> 1)
func (g *Encoder) encodeToGray(binary uint8) uint8 {
	mask := uint8((1 << g.bitsPerSymbol) - 1)
	binary &= mask
	return (binary ^ (binary >> 1)) & mask
}

// Process processes input bytes to Gray-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (g *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := 0
		for readErr == nil {
			n, err := src.Read(srcBuf[:])
			if n == 0 {
				if err == io.EOF {
					break
				}
				if err != nil {
					break
				}
				continue
			}
			totalRead += n
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
		}
		// Gray code is 1:1 mapping, so output size equals input size
		return totalRead
	}

	// Process bytes
	for readErr == nil {
		n, err := src.Read(srcBuf[:])
		if n == 0 {
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
			continue
		}

		// Process each byte
		encodedBuf := make([]byte, n)
		for i := range n {
			byteVal := srcBuf[i]

			if g.bitsPerSymbol == 8 {
				// Full byte encoding
				encodedBuf[i] = g.encodeToGray(byteVal)
			} else if g.bitsPerSymbol == 4 {
				// Nibble encoding: process each nibble separately
				lowNibble := byteVal & 0x0F
				highNibble := (byteVal >> 4) & 0x0F
				encodedLow := g.encodeToGray(lowNibble)
				encodedHigh := g.encodeToGray(highNibble)
				encodedBuf[i] = encodedLow | (encodedHigh << 4)
			} else {
				// For other bit widths, process the lower bits
				encodedBuf[i] = g.encodeToGray(byteVal)
			}
		}

		// Write encoded bytes
		_, writeErr := dst.Write(encodedBuf)
		if writeErr != nil {
			return totalWritten
		}
		totalWritten += n

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
func (g *Encoder) Reset() {
	// No state to reset (stateless)
}

// BitsPerSymbol returns the number of bits per symbol.
func (g *Encoder) BitsPerSymbol() int {
	return g.bitsPerSymbol
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)

