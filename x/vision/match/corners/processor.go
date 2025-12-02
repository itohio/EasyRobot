package corners

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for corner matching.
// It matches corners between two images (e.g., for stereo calibration or tracking).
//
// Allocations: This processor preallocates buffers to avoid allocations in the Process() callstack.
type Processor struct {
	// Preallocated scratch space to avoid allocations in Process()
}

// New creates a new corner matching processor.
func New() *Processor {
	return &Processor{}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads two input tensors (corners from two images), matches them, and writes output tensor to dst.
//
// Input: Two corner tensors (expected to be read sequentially)
// Output: Matched corners tensor or match data
//
// If dst is nil, returns the estimated number of output tensors.
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 2 tensors in -> 1 tensor out (matched pairs)
		return src.Len() / 2
	}

	// Read two corner tensors for matching
	// TODO: Implement corner matching logic
	// For now, this is a placeholder
	return 0
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// No state to reset
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	// No resources to close
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)
