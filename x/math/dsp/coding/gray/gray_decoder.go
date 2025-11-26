package gray

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Decoder implements Gray code decoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
// Gray codes provide noise resilience by ensuring adjacent values differ by only one bit.
// Decodes data by converting Gray code values back to binary (1:1 mapping, no size change).
type Decoder struct {
	bitsPerSymbol int // Number of bits per symbol (typically 4 for nibbles or 8 for bytes)
}

// NewDecoder creates a new Gray code decoder with default 8-bit decoding.
func NewDecoder() *Decoder {
	return NewDecoderWithBits(8)
}

// NewDecoderWithBits creates a new Gray code decoder with specified bits per symbol.
// bitsPerSymbol must be 1-8 (typically 4 for nibbles or 8 for bytes).
func NewDecoderWithBits(bitsPerSymbol int) *Decoder {
	if bitsPerSymbol < 1 || bitsPerSymbol > 8 {
		panic("Gray code bits per symbol must be 1-8")
	}
	return &Decoder{
		bitsPerSymbol: bitsPerSymbol,
	}
}

// decodeFromGray converts a Gray code value back to binary.
// Formula: binary = gray ^ (gray >> 1) ^ (gray >> 2) ^ ... (for n bits)
func (g *Decoder) decodeFromGray(gray uint8) uint8 {
	mask := uint8((1 << g.bitsPerSymbol) - 1)
	gray &= mask
	binary := gray
	// For n bits, we need n-1 XOR operations
	for shift := 1; shift < g.bitsPerSymbol; shift++ {
		binary ^= (gray >> shift)
	}
	return binary & mask
}

// Process processes Gray-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (g *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		decodedBuf := make([]byte, n)
		for i := range n {
			grayVal := srcBuf[i]

			if g.bitsPerSymbol == 8 {
				// Full byte decoding
				decodedBuf[i] = g.decodeFromGray(grayVal)
			} else if g.bitsPerSymbol == 4 {
				// Nibble decoding: process each nibble separately
				lowNibble := grayVal & 0x0F
				highNibble := (grayVal >> 4) & 0x0F
				decodedLow := g.decodeFromGray(lowNibble)
				decodedHigh := g.decodeFromGray(highNibble)
				decodedBuf[i] = decodedLow | (decodedHigh << 4)
			} else {
				// For other bit widths, process the lower bits
				decodedBuf[i] = g.decodeFromGray(grayVal)
			}
		}

		// Write decoded bytes
		_, writeErr := dst.Write(decodedBuf)
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

// Reset resets the decoder state.
// Reset implements chain.Processor interface.
func (g *Decoder) Reset() {
	// No state to reset (stateless)
}

// BitsPerSymbol returns the number of bits per symbol.
func (g *Decoder) BitsPerSymbol() int {
	return g.bitsPerSymbol
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)

