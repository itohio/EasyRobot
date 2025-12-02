package depth

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for depth computation.
// It converts disparity maps to depth maps using the Q matrix from stereo calibration.
//
// Allocations: This processor preallocates output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	// Q matrix (4x4) for disparity-to-depth conversion
	qMatrix gocv.Mat

	// Preallocated scratch space to avoid allocations in Process()
	scratchPoints3D gocv.Mat
}

// New creates a new depth computation processor with the given Q matrix.
// The Q matrix is not copied; the processor takes a reference to it.
func New(qMatrix gocv.Mat) *Processor {
	return &Processor{
		qMatrix:         qMatrix,
		scratchPoints3D: gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads disparity tensor(s) from src, converts to depth, and writes depth tensor(s) to dst.
//
// Input: Disparity tensor (gocv.Mat-backed, typically from stereo matching)
// Output: Depth tensor (gocv.Mat-backed, 3D points or depth map)
//
// If dst is nil, returns the estimated number of output tensors (1 tensor per input tensor).
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 1 disparity tensor in -> 1 depth tensor out
		return src.Len()
	}

	// Read input disparity tensor(s)
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

	disparityMat, err := gocvTensor.Mat()
	if err != nil {
		return 0
	}

	if disparityMat.Empty() {
		return 0
	}

	// Convert disparity to 3D points using Q matrix
	// ReprojectImageTo3D converts disparity map to 3D point cloud
	gocv.ReprojectImageTo3D(disparityMat, &p.scratchPoints3D, p.qMatrix, false, gocv.MatTypeCV32FC1)

	// Extract depth (Z coordinate) from 3D points
	// The scratchPoints3D is a 3-channel Mat where each pixel is (X, Y, Z)
	// We can either return the full 3D points or extract just the Z channel (depth)
	// For now, we'll return the full 3D points tensor
	
	// Create output tensor from 3D points Mat
	outputMat := p.scratchPoints3D.Clone()
	outputTensor, err := tensorgocv.FromMat(outputMat, tensorgocv.WithAdoptedMat())
	if err != nil {
		return 0
	}

	// Write output tensor
	outputTensors := []types.Tensor{outputTensor}
	written, _ := dst.Write(outputTensors)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// Clear scratch space if needed
	if !p.scratchPoints3D.Empty() {
		p.scratchPoints3D.Close()
		p.scratchPoints3D = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchPoints3D.Empty() {
		p.scratchPoints3D.Close()
	}
	// Note: We don't close qMatrix as it is owned by the caller
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)

