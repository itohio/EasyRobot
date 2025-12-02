package affine

import (
	"image"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for affine transformation.
// It applies an affine transform to input images using a transformation matrix.
//
// Allocations: This processor preallocates output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	// Affine transformation matrix (2x3)
	transformMatrix gocv.Mat

	// Output image size (if zero, uses input size)
	outputSize image.Point

	// Preallocated scratch space to avoid allocations in Process()
	scratchTransformed gocv.Mat
}

// New creates a new affine transformation processor with the given transformation matrix.
// The matrix is not copied; the processor takes a reference to it.
func New(transformMatrix gocv.Mat) *Processor {
	return &Processor{
		transformMatrix:    transformMatrix,
		scratchTransformed: gocv.NewMat(),
	}
}

// NewWithSize creates a new affine transformation processor with transform matrix and output size.
func NewWithSize(transformMatrix gocv.Mat, outputSize image.Point) *Processor {
	return &Processor{
		transformMatrix:    transformMatrix,
		outputSize:         outputSize,
		scratchTransformed: gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads input tensor(s) from src, applies affine transform, and writes output tensor(s) to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: Transformed image tensor (gocv.Mat-backed)
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

	// Determine output size
	size := p.outputSize
	if size.X == 0 || size.Y == 0 {
		// Use input image size if output size not specified
		size = image.Point{X: mat.Cols(), Y: mat.Rows()}
	}

	// Apply affine transformation
	// WarpAffine signature: WarpAffine(src, dst, transformMatrix, size image.Point)
	gocv.WarpAffine(mat, &p.scratchTransformed, p.transformMatrix, size)

	// Create output tensor from transformed Mat
	outputMat := p.scratchTransformed.Clone()
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
	if !p.scratchTransformed.Empty() {
		p.scratchTransformed.Close()
		p.scratchTransformed = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchTransformed.Empty() {
		p.scratchTransformed.Close()
	}
	// Note: We don't close transformMatrix as it is owned by the caller
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)
