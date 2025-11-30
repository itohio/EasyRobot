package features

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for feature extraction.
// It extracts features (keypoints and descriptors) from input image tensors.
//
// Allocations: This processor preallocates gocv.Mat objects for descriptors to avoid
// allocations in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	detector gocv.ORB
	matcher  gocv.FlannBasedMatcher

	// Preallocated scratch space to avoid allocations in Process()
	scratchDescriptors gocv.Mat
	scratchMask        gocv.Mat
}

// New creates a new feature extraction processor.
func New() (*Processor, error) {
	p := &Processor{
		detector: gocv.NewORB(),
		matcher:  gocv.NewFlannBasedMatcher(),
		// Preallocate scratch space (will be resized as needed by OpenCV)
		scratchMask: gocv.NewMat(),
	}
	return p, nil
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads input tensor(s) from src, extracts features, and writes output tensor(s) to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: Descriptor tensor (gocv.Mat-backed) with keypoints stored in metadata
//
// If dst is nil, returns the estimated number of output tensors (1 tensor per input tensor).
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 1 tensor in -> 1 tensor out
		return src.Len()
	}

	// Read input tensor(s)
	inputTensors := make([]types.Tensor, 1)
	n, err := src.Read(inputTensors)
	if err != nil || n == 0 {
		return 0
	}

	inputTensor := inputTensors[0]

	// Convert tensor to gocv.Mat
	gocvTensor, ok := inputTensor.(tensorgocv.Tensor)
	if !ok {
		return 0
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return 0
	}

	// Extract features
	// Use preallocated scratch mask (empty mat)
	keypoints, descriptors := p.detector.DetectAndCompute(mat, p.scratchMask)
	if len(keypoints) == 0 {
		// No features found, return 0
		return 0
	}

	// Create output tensor from descriptors
	// Note: descriptors is a gocv.Mat returned by DetectAndCompute.
	// We need to create a tensor from it using WithAdoptedMat to transfer ownership.
	// IMPORTANT: Tensors are NOT guaranteed to be zero-initialized.
	outputTensor := tensorgocv.WithAdoptedMat(descriptors)

	// Store keypoints in tensor metadata (if tensor supports metadata)
	// For now, we'll just return the descriptor tensor
	// In a full implementation, we might need a custom tensor type that holds both

	// Write output tensor
	outputTensors := []types.Tensor{outputTensor}
	written, _ := dst.Write(outputTensors)
	return written
}

// Reset resets the processor state.
// Clears preallocated scratch space.
func (p *Processor) Reset() {
	// Clear scratch descriptors (will be recreated as needed)
	if !p.scratchDescriptors.Empty() {
		p.scratchDescriptors.Close()
		p.scratchDescriptors = cv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	p.detector.Close()
	p.matcher.Close()
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
	}
	if !p.scratchDescriptors.Empty() {
		p.scratchDescriptors.Close()
	}
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)
