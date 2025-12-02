package matching

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visionformat "github.com/itohio/EasyRobot/x/vision/transform/format"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, visionformat.StereoPair] for stereo matching.
// It computes disparity maps from rectified stereo image pairs.
//
// Allocations: This processor preallocates SGBM matcher and output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	// SGBM matcher for stereo matching
	sgbm gocv.StereoSGBM

	// Preallocated scratch space to avoid allocations in Process()
	scratchDisparity gocv.Mat
}

// New creates a new stereo matching processor with default SGBM parameters.
func New() *Processor {
	// Default SGBM parameters - can be customized via options later
	sgbm := gocv.NewStereoSGBM(
		0,      // minDisparity
		16,     // numDisparities (must be multiple of 16)
		3,      // blockSize
		8*3*3,  // P1 (penalty for small disparity changes)
		32*3*3, // P2 (penalty for large disparity changes)
		1,      // disp12MaxDiff
		0,      // preFilterCap
		0,      // uniquenessRatio
		0,      // speckleWindowSize
		0,      // speckleRange
		gocv.StereoSGBMModeSGBM3Way,
	)
	return &Processor{
		sgbm:             sgbm,
		scratchDisparity: gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, visionformat.StereoPair].
// Reads StereoPair from src, computes disparity, and writes disparity tensor to dst.
//
// Input: StereoPair (left + right rectified image tensors)
// Output: Disparity tensor (gocv.Mat-backed)
//
// If dst is nil, returns the estimated number of output tensors (1 tensor per input pair).
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[visionformat.StereoPair]) int {
	if dst == nil {
		// Size estimation: 1 StereoPair in -> 1 disparity tensor out
		return src.Len()
	}

	// Read input StereoPair
	pairs := make([]visionformat.StereoPair, 1)
	n, err := src.Read(pairs)
	if err != nil || n == 0 {
		return 0
	}

	pair := pairs[0]

	// Convert tensors to gocv.Mat
	leftTensor, ok1 := pair.Left.(tensorgocv.Tensor)
	rightTensor, ok2 := pair.Right.(tensorgocv.Tensor)
	if !ok1 || !ok2 {
		return 0
	}

	leftMat, err1 := leftTensor.Mat()
	rightMat, err2 := rightTensor.Mat()
	if err1 != nil || err2 != nil {
		return 0
	}

	// Compute disparity using SGBM
	p.sgbm.Compute(leftMat, rightMat, &p.scratchDisparity)
	if p.scratchDisparity.Empty() {
		return 0
	}

	// Create output tensor from disparity Mat
	// Clone the disparity image to create a new tensor
	outputMat := p.scratchDisparity.Clone()
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
	if !p.scratchDisparity.Empty() {
		p.scratchDisparity.Close()
		p.scratchDisparity = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	p.sgbm.Close()
	if !p.scratchDisparity.Empty() {
		p.scratchDisparity.Close()
	}
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, visionformat.StereoPair] = (*Processor)(nil)
