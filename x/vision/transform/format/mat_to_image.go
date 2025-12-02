package format

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// MatToImageProcessor implements chain.Processor[ImageWithTensor, types.Tensor] for converting Mat tensors to image.Image.
// It converts gocv.Mat-backed tensors to image.Image.
//
// Allocations: This processor does not preallocate image.Image as it's created from Mat.
// The image.Image is created fresh for each input tensor.
type MatToImageProcessor struct {
}

// NewMatToImageProcessor creates a new Mat to Image conversion processor.
func NewMatToImageProcessor() *MatToImageProcessor {
	return &MatToImageProcessor{}
}

// Process implements chain.Processor[ImageWithTensor, types.Tensor].
// Reads input tensor(s) from src, converts to image.Image, and writes to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: ImageWithTensor (image.Image + source tensor reference)
//
// If dst is nil, returns the estimated number of output items (1 item per input tensor).
func (p *MatToImageProcessor) Process(dst buffer.Writer[ImageWithTensor], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 1 tensor in -> 1 ImageWithTensor out
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

	// Convert Mat to image.Image
	img, err := mat.ToImage()
	if err != nil {
		return 0
	}

	// Create ImageWithTensor output
	output := ImageWithTensor{
		Image:  img,
		Source: inputTensor, // Keep reference to source tensor
	}

	// Write output
	outputs := []ImageWithTensor{output}
	written, _ := dst.Write(outputs)
	return written
}

// Reset resets the processor state.
func (p *MatToImageProcessor) Reset() {
	// No state to reset
}

// Close releases all resources held by the processor.
func (p *MatToImageProcessor) Close() {
	// No resources to close
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[ImageWithTensor, types.Tensor] = (*MatToImageProcessor)(nil)
