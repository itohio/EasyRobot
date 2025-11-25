package ecc

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// ScramblerEncoder implements LFSR-based scrambling.
// It implements the chain.Processor[byte, byte] interface for composability.
// Scrambling pseudo-randomizes data using linear feedback shift registers.
// This reduces long runs of zeros/ones, spreads spectrum, and reduces EMI.
// Encodes data by XORing with pseudo-random sequence (1:1 mapping, no size change).
// Common polynomials: x^7 + x^4 + 1 (7-bit), x^15 + x^14 + 1 (15-bit), x^23 + x^18 + 1 (23-bit)
type ScramblerEncoder struct {
	polynomial uint32 // LFSR polynomial (taps)
	state      uint32 // Current LFSR state
	width      int    // LFSR width in bits
}

// Common scrambler polynomials
const (
	// 7-bit scrambler: x^7 + x^4 + 1 (used in some protocols)
	Scrambler7Polynomial = 0x91 // 0b10010001 (bits 7 and 4)
	Scrambler7Width      = 7

	// 15-bit scrambler: x^15 + x^14 + 1 (used in some protocols)
	Scrambler15Polynomial = 0xC001 // 0b1100000000000001 (bits 15 and 14)
	Scrambler15Width      = 15

	// 23-bit scrambler: x^23 + x^18 + 1 (used in some protocols)
	Scrambler23Polynomial = 0x40041 // 0b1000000000001000001 (bits 23 and 18)
	Scrambler23Width      = 23
)

// NewScramblerEncoder creates a new scrambler encoder with default 7-bit polynomial.
func NewScramblerEncoder() *ScramblerEncoder {
	return NewScramblerEncoderWithPolynomial(Scrambler7Polynomial, Scrambler7Width)
}

// NewScramblerEncoderWithPolynomial creates a new scrambler encoder with custom polynomial.
func NewScramblerEncoderWithPolynomial(polynomial uint32, width int) *ScramblerEncoder {
	if width < 1 || width > 32 {
		panic("scrambler width must be 1-32")
	}
	// Initialize state to non-zero value (all ones is common)
	initialState := uint32((1 << width) - 1)
	return &ScramblerEncoder{
		polynomial: polynomial,
		state:      initialState,
		width:      width,
	}
}

// nextBit generates the next bit from the LFSR.
func (s *ScramblerEncoder) nextBit() byte {
	mask := uint32((1 << s.width) - 1)
	// Calculate feedback: XOR of tapped bits
	feedback := s.state & s.polynomial
	// Count set bits in feedback (parity)
	parity := byte(0)
	for feedback != 0 {
		parity ^= 1
		feedback &= feedback - 1 // Clear least significant bit
	}
	// Shift and insert feedback
	s.state = ((s.state << 1) | uint32(parity)) & mask
	return parity
}

// Process processes input bytes to scrambled bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (s *ScramblerEncoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// Scrambling is 1:1 mapping, so output size equals input size
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
		scrambledBuf := make([]byte, read)
		for i := range read {
			byteVal := srcBuf[i]
			var scrambledByte byte

			// Process each bit (MSB first)
			for bit := 7; bit >= 0; bit-- {
				inputBit := (byteVal >> bit) & 1
				scramblerBit := s.nextBit()
				// XOR input with scrambler sequence
				scrambledBit := inputBit ^ scramblerBit
				scrambledByte |= (scrambledBit << bit)
			}

			scrambledBuf[i] = scrambledByte
		}

		// Write scrambled bytes
		_, writeErr := dst.Write(scrambledBuf)
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
func (s *ScramblerEncoder) Reset() {
	// Reset to initial state (all ones)
	s.state = uint32((1 << s.width) - 1)
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*ScramblerEncoder)(nil)

