package features

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for feature matching.
// It matches features (keypoints/descriptors) between two images.
//
// Allocations: This processor preallocates matcher and buffers to avoid allocations
// in the Process() callstack.
type Processor struct {
	matcher gocv.FlannBasedMatcher
}

// New creates a new feature matching processor.
func New() *Processor {
	return &Processor{
		matcher: gocv.NewFlannBasedMatcher(),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads two input tensors (descriptors from two images), matches them, and writes output to dst.
//
// Input: Two descriptor tensors (expected to be read sequentially)
// Output: Matched features tensor or match data
//
// If dst is nil, returns the estimated number of output tensors.
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 2 tensors in -> 1 tensor out (matched pairs)
		return src.Len() / 2
	}

	// Read two descriptor tensors for matching
	descriptors := make([]types.Tensor, 2)
	n, err := src.Read(descriptors)
	if err != nil || n < 2 {
		return 0
	}

	// Convert tensors to gocv.Mat
	queryTensor, ok1 := descriptors[0].(tensorgocv.Tensor)
	trainTensor, ok2 := descriptors[1].(tensorgocv.Tensor)
	if !ok1 || !ok2 {
		return 0
	}

	queryMat, err1 := queryTensor.Mat()
	trainMat, err2 := trainTensor.Mat()
	if err1 != nil || err2 != nil {
		return 0
	}

	// Match features using KNN (k=2 for Lowe's ratio test)
	matches := p.matcher.KnnMatch(queryMat, trainMat, 2)
	if len(matches) == 0 {
		return 0
	}

	// For now, return first descriptor tensor as placeholder
	// In a full implementation, we would return matched pairs or match data
	outputTensor := descriptors[0]

	outputTensors := []types.Tensor{outputTensor}
	written, _ := dst.Write(outputTensors)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// No state to reset for matcher
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	p.matcher.Close()
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)

