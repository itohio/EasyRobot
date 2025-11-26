package hamming

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// HammingDecoder implements Hamming(7,4) decoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
// Hamming(7,4) can correct single-bit errors and detect double-bit errors.
// Decodes data by processing 7-bit codewords back to 4-bit nibbles.
// Each 2 input bytes (2 codewords) produce 1 output byte (2 nibbles).
type Decoder struct {
	// Buffer for partial codewords (less than 2 bytes = 2 codewords)
	buffer []byte
	// Reusable buffer for combining buffer + src (avoids allocations in hot path)
	combinedBuf []byte
}

// NewDecoder creates a new Hamming(7,4) decoder.
func NewDecoder() *Decoder {
	return &Decoder{}
}

// decodeCodeword decodes a 7-bit Hamming codeword to a 4-bit nibble.
// Also detects and corrects single-bit errors.
// Returns the decoded nibble and a boolean indicating if an error was corrected.
func (h *Decoder) decodeCodeword(codeword uint8) (uint8, bool) {
	// Extract bits (0-indexed positions)
	// Position 0: P1, 1: P2, 2: D1, 3: P4, 4: D2, 5: D3, 6: D4
	p1 := (codeword >> 0) & 1
	p2 := (codeword >> 1) & 1
	d1 := (codeword >> 2) & 1
	p4 := (codeword >> 3) & 1
	d2 := (codeword >> 4) & 1
	d3 := (codeword >> 5) & 1
	d4 := (codeword >> 6) & 1

	// Recalculate parity bits to check for errors
	// P1 should equal: d1 ^ d2 ^ d4
	checkP1 := d1 ^ d2 ^ d4
	// P2 should equal: d1 ^ d3 ^ d4
	checkP2 := d1 ^ d3 ^ d4
	// P4 should equal: d2 ^ d3 ^ d4
	checkP4 := d2 ^ d3 ^ d4

	// Calculate syndrome (1-indexed error position)
	// Syndrome bits: S1, S2, S4 (corresponding to parity positions)
	s1 := p1 ^ checkP1
	s2 := p2 ^ checkP2
	s4 := p4 ^ checkP4

	// Syndrome value (1-indexed position)
	syndrome := (s1 << 0) | (s2 << 1) | (s4 << 2)

	// If syndrome is 0, no error
	// If syndrome is non-zero, it indicates the error position (1-indexed)
	errorCorrected := false
	if syndrome != 0 {
		// Convert 1-indexed to 0-indexed
		errorPos := syndrome - 1
		// Flip the bit at error position
		codeword ^= (1 << errorPos)
		errorCorrected = true

		// Re-extract data bits after correction
		d1 = (codeword >> 2) & 1
		d2 = (codeword >> 4) & 1
		d3 = (codeword >> 5) & 1
		d4 = (codeword >> 6) & 1
	}

	// Reconstruct nibble from data bits
	nibble := (d1 << 0) | (d2 << 1) | (d3 << 2) | (d4 << 3)

	return nibble, errorCorrected
}

// Process processes Hamming-encoded bytes to original data bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Processes pairs of bytes (2 codewords) to produce 1 output byte (2 nibbles).
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (h *Decoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	var srcBuf [256]byte
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := len(h.buffer)
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
		// Each 2 input bytes (2 codewords) → 1 output byte (2 nibbles)
		return totalRead / 2
	}

	// Process incrementally - decode and write bytes as soon as complete pairs (2 bytes) are available
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

		// Combine buffered data with new chunk
		combinedSize := len(h.buffer) + n
		if cap(h.combinedBuf) < combinedSize {
			// Grow buffer with some extra capacity
			h.combinedBuf = make([]byte, combinedSize, combinedSize*2)
		}
		combined := h.combinedBuf[:combinedSize]
		copy(combined, h.buffer)
		copy(combined[len(h.buffer):], srcBuf[:n])

		// Clear buffer (we'll rebuild it with any remaining partial pair)
		h.buffer = h.buffer[:0]

		// Process complete pairs (2 bytes = 2 codewords → 1 output byte)
		processedIdx := 0
		for processedIdx+2 <= len(combined) {
			// Get two codewords
			codeword1 := combined[processedIdx]
			codeword2 := combined[processedIdx+1]

			// Decode both codewords to nibbles
			nibble1, _ := h.decodeCodeword(codeword1)
			nibble2, _ := h.decodeCodeword(codeword2)

			// Combine nibbles into byte
			outputByte := nibble1 | (nibble2 << 4)

			// Write output byte
			_, writeErr := dst.Write([]byte{outputByte})
			if writeErr != nil {
				// Write failed, save remaining data to buffer
				h.buffer = append(h.buffer, combined[processedIdx:]...)
				return totalWritten
			}
			totalWritten++
			processedIdx += 2
		}

		// Save any remaining partial pair to buffer (at most 1 byte)
		if processedIdx < len(combined) {
			h.buffer = append(h.buffer, combined[processedIdx:]...)
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
func (h *Decoder) Reset() {
	h.buffer = h.buffer[:0]           // Clear buffer but keep capacity
	h.combinedBuf = h.combinedBuf[:0] // Clear combined buffer but keep capacity
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Decoder)(nil)

