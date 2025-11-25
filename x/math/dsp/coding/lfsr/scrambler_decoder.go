package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// ScramblerDecoder implements LFSR-based descrambling.
// It implements the chain.Processor[byte, byte] interface for composability.
// Descrambling XORs scrambled data with the same pseudo-random sequence to recover original.
// Decodes data by XORing with pseudo-random sequence (1:1 mapping, no size change).
// Must use the same polynomial and initial state as the encoder.
type ScramblerDecoder struct {
	polynomial uint32 // LFSR polynomial (must match encoder)
	state      uint32 // Current LFSR state (must match encoder)
	width      int    // LFSR width in bits
}

// NewScramblerDecoder creates a new scrambler decoder with default 7-bit polynomial.
func NewScramblerDecoder() *ScramblerDecoder {
	return NewScramblerDecoderWithPolynomial(Scrambler7Polynomial, Scrambler7Width)
}

// NewScramblerDecoderWithPolynomial creates a new scrambler decoder with custom polynomial.
func NewScramblerDecoderWithPolynomial(polynomial uint32, width int) *ScramblerDecoder {
	if width < 1 || width > 32 {
		panic("scrambler width must be 1-32")
	}
	// Initialize state to non-zero value (must match encoder)
	initialState := uint32((1 << width) - 1)
	return &ScramblerDecoder{
		polynomial: polynomial,
		state:      initialState,
		width:      width,
	}
}

// nextBit generates the next bit from the LFSR (same as encoder).
func (s *ScramblerDecoder) nextBit() byte {
	mask := uint32((1 << s.width) - 1)
	feedback := s.state & s.polynomial
	parity := byte(0)
	for feedback != 0 {
		parity ^= 1
		feedback &= feedback - 1
	}
	s.state = ((s.state << 1) | uint32(parity)) & mask
	return parity
}

// Process processes scrambled bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (s *ScramblerDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// Descrambling is 1:1 mapping, so output size equals input size
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
		descrambledBuf := make([]byte, read)
		for i := range read {
			scrambledByte := srcBuf[i]
			var descrambledByte byte

			// Process each bit (MSB first)
			for bit := 7; bit >= 0; bit-- {
				scrambledBit := (scrambledByte >> bit) & 1
				scramblerBit := s.nextBit()
				// XOR scrambled with scrambler sequence to recover original
				descrambledBit := scrambledBit ^ scramblerBit
				descrambledByte |= (descrambledBit << bit)
			}

			descrambledBuf[i] = descrambledByte
		}

		// Write descrambled bytes
		_, writeErr := dst.Write(descrambledBuf)
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
func (s *ScramblerDecoder) Reset() {
	// Reset to initial state (must match encoder)
	s.state = uint32((1 << s.width) - 1)
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*ScramblerDecoder)(nil)

