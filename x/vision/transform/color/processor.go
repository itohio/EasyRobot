package color

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for color space conversion.
// It converts images between different color spaces (e.g., BGR to RGB, BGR to grayscale).
//
// Allocations: This processor preallocates output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	// Color conversion code (e.g., gocv.ColorBGRToGray, gocv.ColorBGRToRGB)
	conversionCode gocv.ColorConversionCode

	// Preallocated scratch space to avoid allocations in Process()
	scratchConverted gocv.Mat
}

// New creates a new color conversion processor with the given conversion code.
func New(conversionCode gocv.ColorConversionCode) *Processor {
	return &Processor{
		conversionCode:   conversionCode,
		scratchConverted: gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads input tensor(s) from src, converts color space, and writes output tensor(s) to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: Color-converted image tensor (gocv.Mat-backed)
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

	if mat.Empty() {
		return 0
	}

	// Convert color space
	gocv.CvtColor(mat, &p.scratchConverted, p.conversionCode)

	// Create output tensor from converted Mat
	outputMat := p.scratchConverted.Clone()
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
	if !p.scratchConverted.Empty() {
		p.scratchConverted.Close()
		p.scratchConverted = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchConverted.Empty() {
		p.scratchConverted.Close()
	}
}

// CvtColor converts the color space of an image.
// This is a direct helper function that can be called without using the processor.
// Returns the color-converted image Mat. The caller is responsible for closing the returned Mat.
func CvtColor(src gocv.Mat, code gocv.ColorConversionCode) (gocv.Mat, error) {
	dst := gocv.NewMat()
	gocv.CvtColor(src, &dst, code)
	return dst, nil
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)

