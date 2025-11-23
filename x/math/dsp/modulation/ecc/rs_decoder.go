package ecc

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

// RSDecoder implements Reed-Solomon decoding at the byte level.
// It implements the modulation.ByteDecoder interface for composability.
// The decoder handles streaming data with unknown boundaries.
// Note: Decoder doesn't expect padding - it only decodes complete blocks (no padding).
type RSDecoder struct {
	// RS code parameters
	n int // Total codeword length (data + parity)
	k int // Data symbols per codeword
	t int // Maximum correctable errors (t = (n-k)/2)

	// Internal state
	decoder rsDecoderImpl

	// Streaming state: buffer for partial blocks
	buffer []byte // Buffered partial block data
	
	// Reusable buffer for combining buffer + src (avoids allocations in hot path)
	combinedBuf []byte
}

// rsDecoderImpl is the internal decoder implementation interface.
type rsDecoderImpl interface {
	Decode(data []byte) ([]byte, error)
	Reset()
}

// NewRSDecoder creates a new Reed-Solomon decoder.
// Parameters must match the encoder:
//   - n: Total codeword length
//   - k: Data symbols per codeword
func NewRSDecoder(n, k int) *RSDecoder {
	if n <= k {
		panic(fmt.Sprintf("RS decoder: n (%d) must be greater than k (%d)", n, k))
	}
	if k <= 0 {
		panic(fmt.Sprintf("RS decoder: k (%d) must be positive", k))
	}
	if n > 255 {
		panic(fmt.Sprintf("RS decoder: n (%d) must be <= 255 for byte-level decoding", n))
	}

	t := (n - k) / 2
	if t <= 0 {
		panic(fmt.Sprintf("RS decoder: t (%d) must be positive (n=%d, k=%d)", t, n, k))
	}

	decoder := &RSDecoder{
		n: n,
		k: k,
		t: t,
	}

	// Use library-based implementation
	decoder.decoder = newLibraryRSDecoder(n, k)

	return decoder
}

// Decode decodes Reed-Solomon encoded bytes to original data bytes.
// Handles streaming data with unknown boundaries and padding.
// Input data may contain partial blocks, padding, or start at arbitrary offsets.
// Returns the number of bytes written to dst.
// Returns expected dst length if dst is nil.
// Decode implements modulation.ByteDecoder interface.
func (rs *RSDecoder) Decode(dst, src []byte) int {
	if dst == nil {
		// Calculate expected output length (conservative estimate)
		// Include buffered data and new data
		totalBytes := len(rs.buffer) + len(src)
		blocks := totalBytes / rs.n
		return blocks * rs.k
	}

	if len(src) == 0 && len(rs.buffer) == 0 {
		return 0
	}

	// Combine buffered data with new input
	// Reuse combinedBuf if large enough, otherwise grow it
	combinedSize := len(rs.buffer) + len(src)
	if cap(rs.combinedBuf) < combinedSize {
		// Grow buffer with some extra capacity for future use
		rs.combinedBuf = make([]byte, combinedSize, combinedSize*2)
	}
	combined := rs.combinedBuf[:combinedSize]
	
	// Copy buffer and src into combined using BCE-friendly reslice
	bufferSlice := rs.buffer
	bufferSlice = bufferSlice[:len(rs.buffer)] // BCE hint
	copy(combined, bufferSlice)
	copy(combined[len(rs.buffer):], src)

	// Clear buffer (we'll rebuild it with any remaining partial block)
	rs.buffer = rs.buffer[:0]

	written := 0
	processedIdx := 0

	// Try to find valid blocks by attempting to decode at different offsets
	// This handles cases where data doesn't start at a block boundary
	maxOffset := rs.n - 1
	if maxOffset >= len(combined) {
		maxOffset = len(combined) - 1
	}
	if maxOffset < 0 {
		maxOffset = 0
	}

	// Try decoding starting at different offsets to find block boundaries
	foundAlignment := false
	for offset := 0; offset <= maxOffset && !foundAlignment; offset++ {
		currentIdx := offset
		offsetWritten := 0

		// Try to decode blocks starting at this offset
		for currentIdx+rs.n <= len(combined) {
			block := combined[currentIdx : currentIdx+rs.n]

			// Attempt to decode block
			decoded, err := rs.decoder.Decode(block)
			if err != nil {
				// Decoding failed - this might not be the right offset
				// But if we've already decoded some blocks, continue with this alignment
				if offsetWritten > 0 {
					// We've found valid blocks, skip this corrupted one
					currentIdx += rs.n
					continue
				}
				// No valid blocks yet at this offset, try next offset
				break
			}

			// Successful decode - check if we have space in dst
			if written+len(decoded) > len(dst) {
				// Save remaining data to buffer for next call
				rs.buffer = append(rs.buffer, combined[currentIdx:]...)
				return written
			}

			// Copy decoded block to dst
			copy(dst[written:], decoded)
			written += len(decoded)
			offsetWritten += len(decoded)
			currentIdx += rs.n
			processedIdx = currentIdx
			foundAlignment = true // Found at least one valid block
		}

		// If we found valid blocks at this offset, use it and don't try other offsets
		if foundAlignment {
			break
		}
	}

	// Save any remaining partial block to buffer
	if processedIdx < len(combined) {
		remaining := len(combined) - processedIdx
		// Keep at most (n-1) bytes (max partial block size)
		if remaining > rs.n-1 {
			remaining = rs.n - 1
		}
		if remaining > 0 {
			rs.buffer = append(rs.buffer, combined[len(combined)-remaining:]...)
		}
	}

	return written
}

// Reset resets the decoder state.
// Reset implements modulation.ByteDecoder interface.
func (rs *RSDecoder) Reset() {
	if rs.decoder != nil {
		rs.decoder.Reset()
	}
	rs.buffer = rs.buffer[:0] // Clear buffer but keep capacity
	rs.combinedBuf = rs.combinedBuf[:0] // Clear combined buffer but keep capacity
}

// N returns the total codeword length.
func (rs *RSDecoder) N() int {
	return rs.n
}

// K returns the data symbols per codeword.
func (rs *RSDecoder) K() int {
	return rs.k
}

// T returns the maximum correctable errors.
func (rs *RSDecoder) T() int {
	return rs.t
}

// Verify interface compliance
var _ modulation.ByteDecoder = (*RSDecoder)(nil)

