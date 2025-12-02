package undistort

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for image undistortion.
// It corrects lens distortion using camera calibration parameters.
//
// Allocations: This processor preallocates output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	// Camera calibration parameters
	cameraMatrix     gocv.Mat // Camera matrix (3x3)
	distortionCoeffs gocv.Mat // Distortion coefficients
	newCameraMatrix  gocv.Mat // Optional new camera matrix (can be empty)

	// Preallocated scratch space to avoid allocations in Process()
	scratchUndistorted gocv.Mat
}

// New creates a new undistortion processor with the given camera calibration parameters.
// The matrices are not copied; the processor takes references to them.
func New(cameraMatrix, distortionCoeffs gocv.Mat) *Processor {
	return &Processor{
		cameraMatrix:       cameraMatrix,
		distortionCoeffs:   distortionCoeffs,
		scratchUndistorted: gocv.NewMat(),
	}
}

// NewWithNewCameraMatrix creates a new undistortion processor with optional new camera matrix.
func NewWithNewCameraMatrix(cameraMatrix, distortionCoeffs, newCameraMatrix gocv.Mat) *Processor {
	return &Processor{
		cameraMatrix:       cameraMatrix,
		distortionCoeffs:   distortionCoeffs,
		newCameraMatrix:    newCameraMatrix,
		scratchUndistorted: gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads input tensor(s) from src, undistorts them, and writes output tensor(s) to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: Undistorted image tensor (gocv.Mat-backed)
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

	// Undistort the image
	var newCamMat gocv.Mat
	if p.newCameraMatrix.Empty() {
		newCamMat = gocv.NewMat() // Empty mat means use identity
	} else {
		newCamMat = p.newCameraMatrix
	}

	if err := gocv.Undistort(mat, &p.scratchUndistorted, p.cameraMatrix, p.distortionCoeffs, newCamMat); err != nil {
		return 0
	}

	// Create output tensor from undistorted Mat
	outputMat := p.scratchUndistorted.Clone()
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
	if !p.scratchUndistorted.Empty() {
		p.scratchUndistorted.Close()
		p.scratchUndistorted = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchUndistorted.Empty() {
		p.scratchUndistorted.Close()
	}
	// Note: We don't close cameraMatrix, distortionCoeffs, or newCameraMatrix
	// as they are owned by the caller
}

// Undistort corrects lens distortion in an image using camera calibration parameters.
// This is a direct helper function that can be called without using the processor.
// Returns the undistorted image Mat. The caller is responsible for closing the returned Mat.
func Undistort(src, cameraMatrix, distortionCoeffs, newCameraMatrix gocv.Mat) (gocv.Mat, error) {
	dst := gocv.NewMat()

	var newCamMat gocv.Mat
	if newCameraMatrix.Empty() {
		newCamMat = gocv.NewMat() // Empty mat means use identity
	} else {
		newCamMat = newCameraMatrix
	}

	if err := gocv.Undistort(src, &dst, cameraMatrix, distortionCoeffs, newCamMat); err != nil {
		dst.Close()
		return gocv.Mat{}, err
	}

	return dst, nil
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)
