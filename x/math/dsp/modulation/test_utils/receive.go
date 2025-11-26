package test_utils

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// ReceiveOptions configures the Receive function behavior
type ReceiveOptions struct {
	// ChunkParts is the number of chunks to split processing into for streaming simulation
	ChunkParts int
}

// DefaultReceiveOptions returns default options
func DefaultReceiveOptions() ReceiveOptions {
	return ReceiveOptions{
		ChunkParts: 10,
	}
}

// Receive processes samples through a demodulator using chunked processing to simulate streaming.
//
// Parameters:
//   - sampleRate: Sample rate in Hz (currently unused, reserved for future channel distortion)
//   - demodulator: Demodulator processor (chain.Processor[byte, float32])
//   - dst: Destination buffer writer (typically buffer.Bits)
//   - samples: Complete samples buffer to process
//   - opts: Optional configuration (nil uses defaults)
//
// Returns:
//   - totalDecoded: Total number of bytes/symbols decoded
func Receive(
	sampleRate float32,
	demodulator chain.Processor[byte, float32],
	dst buffer.Writer[byte],
	samples []float32,
	opts *ReceiveOptions,
) int {
	if opts == nil {
		def := DefaultReceiveOptions()
		opts = &def
	}

	// Simulate streaming by splitting into chunks
	// Each chunk is a slice view of the same buffer (no copying!)
	chunks := splitIntoChunksFromBuffer(samples, opts.ChunkParts)
	totalDecoded := 0
	for _, chunk := range chunks {
		// chunk is already a slice view, use it directly
		reader := buffer.New(chunk)
		decoded := demodulator.Process(dst, reader)
		totalDecoded += decoded
	}
	return totalDecoded
}

// splitIntoChunksFromBuffer splits a buffer into chunk slices (views, no copying)
// All chunks are views into the same underlying buffer
func splitIntoChunksFromBuffer(samples []float32, parts int) [][]float32 {
	if parts <= 1 {
		return [][]float32{samples}
	}
	chunks := make([][]float32, 0, parts)
	chunkSize := len(samples) / parts
	for i := range parts {
		start := i * chunkSize
		end := start + chunkSize
		if i == parts-1 {
			end = len(samples)
		}
		// Use slice view (no copying) - all chunks point to same underlying buffer
		chunks = append(chunks, samples[start:end])
	}
	return chunks
}

