package hamming

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// HammingEncoder implements Hamming(7,4) encoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
// Hamming(7,4) encodes 4 data bits into 7 bits (4 data + 3 parity).
// It can correct single-bit errors and detect double-bit errors.
// Encodes data by processing bytes as nibbles (4-bit chunks).
// Each input byte (2 nibbles) produces 2 output bytes (each 7-bit codeword in a byte).
type Encoder struct {
}

// NewEncoder creates a new Hamming(7,4) encoder.
func NewEncoder() *Encoder {
	return &Encoder{}
}

// encodeNibble encodes a 4-bit nibble into a 7-bit Hamming codeword.
// The nibble contains data bits, and we calculate 3 parity bits.
// Hamming(7,4) uses positions 1, 2, 4 for parity (1-indexed).
// Data bits go in positions 3, 5, 6, 7 (1-indexed).
// Returns the encoded 7-bit value (bits 0-6, bit 7 is unused).
func (h *Encoder) encodeNibble(nibble uint8) uint8 {
	// Extract 4 data bits from nibble
	d1 := (nibble >> 0) & 1 // Data bit 1 (goes to position 3, 0-indexed: 2)
	d2 := (nibble >> 1) & 1 // Data bit 2 (goes to position 5, 0-indexed: 4)
	d3 := (nibble >> 2) & 1 // Data bit 3 (goes to position 6, 0-indexed: 5)
	d4 := (nibble >> 3) & 1 // Data bit 4 (goes to position 7, 0-indexed: 6)

	// Calculate parity bits (1-indexed positions):
	// P1 (position 1, 0-indexed: 0): covers positions 1, 3, 5, 7 → bits 0, 2, 4, 6
	p1 := d1 ^ d2 ^ d4
	// P2 (position 2, 0-indexed: 1): covers positions 2, 3, 6, 7 → bits 1, 2, 5, 6
	p2 := d1 ^ d3 ^ d4
	// P4 (position 4, 0-indexed: 3): covers positions 4, 5, 6, 7 → bits 3, 4, 5, 6
	p4 := d2 ^ d3 ^ d4

	// Build 7-bit codeword (0-indexed):
	// Position 0 (P1), 1 (P2), 2 (d1), 3 (P4), 4 (d2), 5 (d3), 6 (d4)
	var codeword uint8
	codeword |= (p1 << 0) // Position 0: P1
	codeword |= (p2 << 1) // Position 1: P2
	codeword |= (d1 << 2) // Position 2: d1
	codeword |= (p4 << 3) // Position 3: P4
	codeword |= (d2 << 4) // Position 4: d2
	codeword |= (d3 << 5) // Position 5: d3
	codeword |= (d4 << 6) // Position 6: d4

	return codeword
}

// Process processes input bytes to Hamming-encoded bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Processes bytes as nibbles (4-bit chunks), encoding each nibble to 7 bits.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (h *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
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
		// Each byte = 2 nibbles, each nibble = 7 bits stored in 1 byte
		// So 1 input byte → 2 output bytes
		return totalRead * 2
	}

	// Process bytes as nibbles
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
		for i := range n {
			byteVal := srcBuf[i]

			// Process two nibbles from this byte
			// Low nibble (bits 0-3)
			lowNibble := byteVal & 0x0F
			encodedLow := h.encodeNibble(lowNibble)

			// High nibble (bits 4-7)
			highNibble := (byteVal >> 4) & 0x0F
			encodedHigh := h.encodeNibble(highNibble)

			// Each 7-bit codeword is stored in a byte (bit 7 unused)
			// Write low nibble codeword
			_, writeErr := dst.Write([]byte{encodedLow})
			if writeErr != nil {
				return totalWritten
			}
			totalWritten++

			// Write high nibble codeword
			_, writeErr = dst.Write([]byte{encodedHigh})
			if writeErr != nil {
				return totalWritten
			}
			totalWritten++
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

// Reset resets the encoder state.
// Reset implements chain.Processor interface.
func (h *Encoder) Reset() {
	// No state to reset (stateless)
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)

