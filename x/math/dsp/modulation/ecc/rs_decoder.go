package ecc

import (
	"fmt"
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// RSDecoder implements Reed-Solomon decoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
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

// Process processes Reed-Solomon encoded bytes to original data bytes.
// Handles streaming data with unknown boundaries and padding.
// Input data may contain partial blocks, padding, or start at arbitrary offsets.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Decodes and writes bytes as soon as complete blocks (n bytes) are available.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (rs *RSDecoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	// Buffer size: enough for one complete block (n bytes) plus some extra for partial blocks
	// We only need to buffer up to n-1 bytes for partial blocks
	var srcBuf [256]byte // Small buffer for reading chunks
	totalWritten := 0
	var readErr error

	// If dst is nil, calculate and return size
	if dst == nil {
		totalRead := len(rs.buffer)
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
		blocks := totalRead / rs.n
		return blocks * rs.k
	}

	// Combine buffered data with new input for processing
	// We'll process incrementally and write as soon as we have complete blocks
	for readErr == nil {
		// Read a chunk of data
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
		combinedSize := len(rs.buffer) + n
		if cap(rs.combinedBuf) < combinedSize {
			// Grow buffer with some extra capacity
			rs.combinedBuf = make([]byte, combinedSize, combinedSize*2)
		}
		combined := rs.combinedBuf[:combinedSize]

		// Copy buffer and new data
		copy(combined, rs.buffer)
		copy(combined[len(rs.buffer):], srcBuf[:n])

		// Clear buffer (we'll rebuild it with any remaining partial block)
		rs.buffer = rs.buffer[:0]

		// Process complete blocks and write immediately
		// Try to find valid blocks by attempting to decode at different offsets
		// This handles cases where data doesn't start at a block boundary
		processedIdx := 0
		maxOffset := rs.n - 1
		if maxOffset >= len(combined) {
			maxOffset = len(combined) - 1
		}
		if maxOffset < 0 {
			maxOffset = 0
		}

		// Try decoding starting at different offsets to find block boundaries
		foundAlignment := false
		for offset := 0; offset <= maxOffset && !foundAlignment && offset+rs.n <= len(combined); offset++ {
			currentIdx := offset
			offsetWritten := 0

			// Try to decode blocks starting at this offset
			for currentIdx+rs.n <= len(combined) {
				block := combined[currentIdx : currentIdx+rs.n]

				// Attempt to decode block
				decoded, decodeErr := rs.decoder.Decode(block)
				if decodeErr != nil {
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

				// Successful decode - write to dst immediately
				_, writeErr := dst.Write(decoded)
				if writeErr != nil {
					// Write failed - save remaining data to buffer for next call
					rs.buffer = append(rs.buffer, combined[currentIdx:]...)
					return totalWritten
				}

				totalWritten += len(decoded)
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

		// Save any remaining partial block to buffer (at most n-1 bytes)
		if processedIdx < len(combined) {
			remaining := len(combined) - processedIdx
			if remaining > rs.n-1 {
				// Only keep the last n-1 bytes (max partial block size)
				rs.buffer = append(rs.buffer, combined[len(combined)-(rs.n-1):]...)
			} else if remaining > 0 {
				// Save the remaining bytes
				rs.buffer = append(rs.buffer, combined[processedIdx:]...)
			}
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
func (rs *RSDecoder) Reset() {
	if rs.decoder != nil {
		rs.decoder.Reset()
	}
	rs.buffer = rs.buffer[:0]           // Clear buffer but keep capacity
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
var _ chain.Processor[byte, byte] = (*RSDecoder)(nil)
