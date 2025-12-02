package depth

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	visionformat "github.com/itohio/EasyRobot/x/vision/transform/format"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// CombinedProcessor implements chain.Processor[types.Tensor, visionformat.StereoPair] for depth computation.
// It computes depth directly from rectified stereo image pairs using stereo matching and Q matrix.
//
// Allocations: This processor preallocates SGBM matcher and output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type CombinedProcessor struct {
	// Q matrix (4x4) for disparity-to-depth conversion
	qMatrix gocv.Mat

	// SGBM matcher for stereo matching
	sgbm gocv.StereoSGBM

	// Preallocated scratch space to avoid allocations in Process()
	scratchDisparity gocv.Mat
	scratchPoints3D  gocv.Mat
}

// NewCombined creates a new combined stereo depth processor with the given Q matrix.
// The Q matrix is not copied; the processor takes a reference to it.
func NewCombined(qMatrix gocv.Mat) *CombinedProcessor {
	// Default SGBM parameters
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
	return &CombinedProcessor{
		qMatrix:         qMatrix,
		sgbm:            sgbm,
		scratchDisparity: gocv.NewMat(),
		scratchPoints3D:  gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, visionformat.StereoPair].
// Reads StereoPair from src, computes disparity and depth, and writes depth tensor to dst.
//
// Input: StereoPair (left + right rectified image tensors)
// Output: Depth tensor (3D points tensor - X, Y, Z channels, where Z is depth)
//
// If dst is nil, returns the estimated number of output tensors (1 tensor per input pair).
func (p *CombinedProcessor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[visionformat.StereoPair]) int {
	if dst == nil {
		// Size estimation: 1 StereoPair in -> 1 depth tensor out
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

	// Step 1: Compute disparity using SGBM
	p.sgbm.Compute(leftMat, rightMat, &p.scratchDisparity)
	if p.scratchDisparity.Empty() {
		return 0
	}

	// Step 2: Convert disparity to 3D points (depth) using Q matrix
	// ReprojectImageTo3D converts disparity map to 3D point cloud
	// Output is 3-channel Mat: (X, Y, Z) where Z is the depth
	gocv.ReprojectImageTo3D(p.scratchDisparity, &p.scratchPoints3D, p.qMatrix, false, gocv.MatTypeCV32FC1)

	if p.scratchPoints3D.Empty() {
		return 0
	}

	// Create output tensor from 3D points Mat
	// The 3D points Mat contains (X, Y, Z) coordinates where Z is the depth
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
func (p *CombinedProcessor) Reset() {
	// Clear scratch space if needed
	if !p.scratchDisparity.Empty() {
		p.scratchDisparity.Close()
		p.scratchDisparity = gocv.NewMat()
	}
	if !p.scratchPoints3D.Empty() {
		p.scratchPoints3D.Close()
		p.scratchPoints3D = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *CombinedProcessor) Close() {
	p.sgbm.Close()
	if !p.scratchDisparity.Empty() {
		p.scratchDisparity.Close()
	}
	if !p.scratchPoints3D.Empty() {
		p.scratchPoints3D.Close()
	}
	// Note: We don't close qMatrix as it is owned by the caller
}

// Ensure CombinedProcessor implements chain.Processor
var _ chain.Processor[types.Tensor, visionformat.StereoPair] = (*CombinedProcessor)(nil)

