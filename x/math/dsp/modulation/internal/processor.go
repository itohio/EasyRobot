package internal

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Processor processes complex64 samples.
// This is a specialization of chain.Processor for complex64 → complex64 operations.
var _ chain.Processor[complex64, complex64] = Processor(nil)

type Processor interface {
	// Process processes complex64 samples using one of two strategies:
	// 1. If dst implements buffer.Viewer: obtain the slice and operate on it in-place
	// 2. Otherwise: read from src (or dst if it's also a Reader) in small batches and write to dst
	// Returns the number of samples written to dst.
	// If dst is nil, returns 0.
	// Note: Some processors (e.g., TimingRecovery, MatchedFilter) change sample counts
	// and must read from src/dst and write to dst.
	Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int

	// Reset resets the processor state.
	Reset()
}
