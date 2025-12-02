package format

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// DataToStereoProcessor implements chain.Processor[StereoPair, types.Tensor] for converting synchronized data to stereo pairs.
// It reads pairs of tensors (left/right) from the input buffer.
//
// Allocations: This processor does not allocate - it just pairs existing tensors.
type DataToStereoProcessor struct {
}

// NewDataToStereoProcessor creates a new data to stereo conversion processor.
func NewDataToStereoProcessor() *DataToStereoProcessor {
	return &DataToStereoProcessor{}
}

// Process implements chain.Processor[StereoPair, types.Tensor].
// Reads pairs of tensors from src (assumes left, right, left, right, ... pattern),
// creates stereo pairs, and writes to dst.
//
// Input: Sequence of tensors (left, right, left, right, ...)
// Output: StereoPair (left + right tensors)
//
// If dst is nil, returns the estimated number of output pairs (src.Len() / 2).
func (p *DataToStereoProcessor) Process(dst buffer.Writer[StereoPair], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 2 tensors in -> 1 StereoPair out
		return src.Len() / 2
	}

	// Read two tensors for a stereo pair
	tensors := make([]types.Tensor, 2)
	n, err := src.Read(tensors)
	if err != nil || n < 2 {
		return 0
	}

	// Create stereo pair
	pair := StereoPair{
		Left:  tensors[0],
		Right: tensors[1],
	}

	// Write output
	pairs := []StereoPair{pair}
	written, _ := dst.Write(pairs)
	return written
}

// Reset resets the processor state.
func (p *DataToStereoProcessor) Reset() {
	// No state to reset
}

// Close releases all resources held by the processor.
func (p *DataToStereoProcessor) Close() {
	// No resources to close
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[StereoPair, types.Tensor] = (*DataToStereoProcessor)(nil)
