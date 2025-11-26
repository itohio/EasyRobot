package rs

import (
	"fmt"
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// RSEncoder implements Reed-Solomon encoding at the byte level.
// It implements the chain.Processor[byte, byte] interface for composability.
type Encoder struct {
	// RS code parameters
	n int // Total codeword length (data + parity)
	k int // Data symbols per codeword
	t int // Maximum correctable errors (t = (n-k)/2)

	// Internal state
	encoder rsEncoderImpl

	// Reusable buffer for block encoding (avoids allocations in hot path)
	blockBuf []byte

	// Buffer for partial blocks (less than k bytes)
	buffer []byte
}

// rsEncoderImpl is the internal encoder implementation interface.
// This allows swapping implementations (library-based or custom).
type rsEncoderImpl interface {
	Encode(data []byte) ([]byte, error)
	Reset()
}

// NewEncoder creates a new Reed-Solomon encoder.
// Parameters:
//   - n: Total codeword length (must be > k, typically 255 for byte-level)
//   - k: Data symbols per codeword (must be < n)
//
// The encoder can correct up to t = (n-k)/2 errors.
func NewEncoder(n, k int) *Encoder {
	if n <= k {
		panic(fmt.Sprintf("RS encoder: n (%d) must be greater than k (%d)", n, k))
	}
	if k <= 0 {
		panic(fmt.Sprintf("RS encoder: k (%d) must be positive", k))
	}
	if n > 255 {
		panic(fmt.Sprintf("RS encoder: n (%d) must be <= 255 for byte-level encoding", n))
	}

	t := (n - k) / 2
	if t <= 0 {
		panic(fmt.Sprintf("RS encoder: t (%d) must be positive (n=%d, k=%d)", t, n, k))
	}

	encoder := &Encoder{
		n:        n,
		k:        k,
		t:        t,
		blockBuf: make([]byte, k), // Pre-allocate block buffer
	}

	// Use library-based implementation
	encoder.encoder = newLibraryRSEncoder(n, k)

	return encoder
}

// Process processes input bytes to Reed-Solomon encoded bytes.
// Input data is split into blocks of k bytes, each block encoded to n bytes.
// Reads from buffer.Reader[byte] and writes to buffer.Writer[byte].
// Encodes and writes blocks as soon as complete blocks (k bytes) are available.
// If dst is nil, calculates and returns the theoretical maximum output size.
// Process implements chain.Processor[byte, byte] interface.
func (rs *Encoder) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	// Small buffer for reading chunks
	var srcBuf [256]byte
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
		if totalRead < rs.k {
			return 0
		}
		blocks := totalRead / rs.k
		return blocks * rs.n
	}

	// Process incrementally - encode and write blocks as soon as we have complete blocks (k bytes)
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
		combinedSize := len(rs.buffer) + n
		if cap(rs.blockBuf) < combinedSize {
			// Grow buffer with some extra capacity
			rs.blockBuf = make([]byte, combinedSize, combinedSize*2)
		}
		combined := rs.blockBuf[:combinedSize]
		copy(combined, rs.buffer)
		copy(combined[len(rs.buffer):], srcBuf[:n])

		// Clear buffer (we'll rebuild it with any remaining partial block)
		rs.buffer = rs.buffer[:0]

		// Process complete blocks and write immediately
		processedIdx := 0
		for processedIdx+rs.k <= len(combined) {
			// Get complete block of k bytes
			block := combined[processedIdx : processedIdx+rs.k]

			// Encode block
			encoded, encodeErr := rs.encoder.Encode(block)
			if encodeErr != nil {
				// Encoding failed, save remaining data to buffer
				rs.buffer = append(rs.buffer, combined[processedIdx:]...)
				return totalWritten
			}

			// Write encoded block to dst immediately
			_, writeErr := dst.Write(encoded)
			if writeErr != nil {
				// Write failed, save remaining data to buffer
				rs.buffer = append(rs.buffer, combined[processedIdx:]...)
				return totalWritten
			}

			totalWritten += len(encoded)
			processedIdx += rs.k
		}

		// Save any remaining partial block to buffer (less than k bytes)
		if processedIdx < len(combined) {
			rs.buffer = append(rs.buffer, combined[processedIdx:]...)
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
func (rs *Encoder) Reset() {
	if rs.encoder != nil {
		rs.encoder.Reset()
	}
	rs.buffer = rs.buffer[:0] // Clear buffer but keep capacity
}

// N returns the total codeword length.
func (rs *Encoder) N() int {
	return rs.n
}

// K returns the data symbols per codeword.
func (rs *Encoder) K() int {
	return rs.k
}

// T returns the maximum correctable errors.
func (rs *Encoder) T() int {
	return rs.t
}

// Verify interface compliance
var _ chain.Processor[byte, byte] = (*Encoder)(nil)
