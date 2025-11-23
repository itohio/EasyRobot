package ecc

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

// RSEncoder implements Reed-Solomon encoding at the byte level.
// It implements the modulation.ByteEncoder interface for composability.
type RSEncoder struct {
	// RS code parameters
	n int // Total codeword length (data + parity)
	k int // Data symbols per codeword
	t int // Maximum correctable errors (t = (n-k)/2)

	// Internal state
	encoder rsEncoderImpl
	
	// Reusable buffer for block encoding (avoids allocations in hot path)
	blockBuf []byte
}

// rsEncoderImpl is the internal encoder implementation interface.
// This allows swapping implementations (library-based or custom).
type rsEncoderImpl interface {
	Encode(data []byte) ([]byte, error)
	Reset()
}

// NewRSEncoder creates a new Reed-Solomon encoder.
// Parameters:
//   - n: Total codeword length (must be > k, typically 255 for byte-level)
//   - k: Data symbols per codeword (must be < n)
//
// The encoder can correct up to t = (n-k)/2 errors.
func NewRSEncoder(n, k int) *RSEncoder {
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

	encoder := &RSEncoder{
		n:        n,
		k:        k,
		t:        t,
		blockBuf: make([]byte, k), // Pre-allocate block buffer
	}

	// Use library-based implementation
	encoder.encoder = newLibraryRSEncoder(n, k)

	return encoder
}

// Encode encodes input bytes to Reed-Solomon encoded bytes.
// Input data is split into blocks of k bytes, each block encoded to n bytes.
// Returns the number of bytes written to dst.
// Returns expected dst length if dst is nil.
// Encode implements modulation.ByteEncoder interface.
func (rs *RSEncoder) Encode(dst, src []byte) int {
	if dst == nil {
		// Calculate expected output length
		blocks := (len(src) + rs.k - 1) / rs.k // Ceiling division
		return blocks * rs.n
	}

	if len(src) == 0 {
		return 0
	}

	// Process data in blocks of k bytes
	written := 0
	srcIdx := 0
	
	// Use encoder's pre-allocated block buffer (zero allocations in hot path)
	block := rs.blockBuf[:rs.k]

	for srcIdx < len(src) {
		// Only encode complete blocks (no padding - encoder doesn't add padding)
		if srcIdx+rs.k > len(src) {
			// Incomplete block - don't encode it (no padding)
			break
		}

		// Get complete block of k bytes
		block = block[:rs.k]
		copy(block, src[srcIdx:srcIdx+rs.k])

		// Encode block
		encoded, err := rs.encoder.Encode(block)
		if err != nil {
			// Encoding failed, return bytes written so far
			return written
		}

		// Check if we have space in dst
		if written+len(encoded) > len(dst) {
			return written
		}

		// Copy encoded block to dst
		copy(dst[written:], encoded)
		written += len(encoded)
		srcIdx += rs.k
	}

	return written
}

// Reset resets the encoder state.
// Reset implements modulation.ByteEncoder interface.
func (rs *RSEncoder) Reset() {
	if rs.encoder != nil {
		rs.encoder.Reset()
	}
}

// N returns the total codeword length.
func (rs *RSEncoder) N() int {
	return rs.n
}

// K returns the data symbols per codeword.
func (rs *RSEncoder) K() int {
	return rs.k
}

// T returns the maximum correctable errors.
func (rs *RSEncoder) T() int {
	return rs.t
}

// Verify interface compliance
var _ modulation.ByteEncoder = (*RSEncoder)(nil)

