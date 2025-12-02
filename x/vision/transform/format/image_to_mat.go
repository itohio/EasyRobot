package format

import (
	"image/color"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// ImageToMatProcessor implements chain.Processor[types.Tensor, ImageWithTensor] for converting image.Image to Mat tensors.
// It converts image.Image to gocv.Mat-backed tensors.
//
// Allocations: This processor creates new Mat objects from image.Image.
// The Mat is adopted by the output tensor.
type ImageToMatProcessor struct {
	// Preallocated scratch space (not used currently, but reserved for future optimizations)
}

// NewImageToMatProcessor creates a new Image to Mat conversion processor.
func NewImageToMatProcessor() *ImageToMatProcessor {
	return &ImageToMatProcessor{}
}

// Process implements chain.Processor[types.Tensor, ImageWithTensor].
// Reads ImageWithTensor from src, converts to Mat tensor, and writes to dst.
//
// Input: ImageWithTensor (image.Image + optional source tensor)
// Output: Image tensor (gocv.Mat-backed)
//
// If dst is nil, returns the estimated number of output tensors (1 tensor per input item).
func (p *ImageToMatProcessor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[ImageWithTensor]) int {
	if dst == nil {
		// Size estimation: 1 ImageWithTensor in -> 1 tensor out
		return src.Len()
	}

	// Read input ImageWithTensor(s)
	inputs := make([]ImageWithTensor, 1)
	n, err := src.Read(inputs)
	if err != nil || n == 0 {
		return 0
	}

	input := inputs[0]

	if input.Image == nil {
		return 0
	}

	// Convert image.Image to gocv.Mat
	var mat gocv.Mat
	var errMat error

	switch input.Image.ColorModel() {
	case color.RGBAModel:
		mat, errMat = gocv.ImageToMatRGB(input.Image)
	default:
		// Try RGB conversion as fallback
		mat, errMat = gocv.ImageToMatRGB(input.Image)
	}

	if errMat != nil {
		return 0
	}

	// Create output tensor from Mat
	// Note: Mat is adopted by the tensor, which takes ownership
	outputTensor, err := tensorgocv.FromMat(mat, tensorgocv.WithAdoptedMat())
	if err != nil {
		return 0
	}

	// Write output tensor
	outputTensors := []types.Tensor{outputTensor}
	written, _ := dst.Write(outputTensors)
	return written
}

// Reset resets the processor state.
func (p *ImageToMatProcessor) Reset() {
	// No state to reset
}

// Close releases all resources held by the processor.
func (p *ImageToMatProcessor) Close() {
	// No resources to close
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, ImageWithTensor] = (*ImageToMatProcessor)(nil)
