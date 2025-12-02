package crop

import (
	"image"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Processor implements chain.Processor[types.Tensor, CropRectangles] for image cropping.
// It crops subimages from rectangles defined in the input.
//
// Input: CropRectangles (image tensor + list of rectangles)
// Output: Multiple cropped image tensors (one per rectangle)
//
// Allocations: This processor allocates output tensors per rectangle (acceptable for cropping).
type Processor struct {
}

// CropRectangles holds an image tensor and a list of rectangles to crop.
type CropRectangles struct {
	Image      types.Tensor
	Rectangles []image.Rectangle
}

// New creates a new cropping processor.
func New() *Processor {
	return &Processor{}
}

// Process implements chain.Processor[types.Tensor, CropRectangles].
// Reads CropRectangles from src, crops subimages, and writes multiple output tensors to dst.
//
// Input: CropRectangles (image tensor + list of rectangles)
// Output: Multiple image tensors (one per rectangle)
//
// If dst is nil, returns the estimated number of output tensors (number of rectangles).
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[CropRectangles]) int {
	if dst == nil {
		// Size estimation: need to read input to count rectangles
		inputs := make([]CropRectangles, 1)
		n, err := src.Read(inputs)
		if err != nil || n == 0 {
			return 0
		}
		return len(inputs[0].Rectangles)
	}

	// Read input CropRectangles
	inputs := make([]CropRectangles, 1)
	n, err := src.Read(inputs)
	if err != nil || n == 0 {
		return 0
	}

	input := inputs[0]

	if len(input.Rectangles) == 0 {
		return 0
	}

	// Convert tensor to gocv.Mat
	gocvTensor, ok := input.Image.(tensorgocv.Tensor)
	if !ok {
		return 0
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return 0
	}

	// Crop each rectangle
	outputTensors := make([]types.Tensor, 0, len(input.Rectangles))

	for _, rect := range input.Rectangles {
		// Validate rectangle bounds
		if rect.Min.X < 0 || rect.Min.Y < 0 {
			continue
		}
		if rect.Max.X > mat.Cols() || rect.Max.Y > mat.Rows() {
			continue
		}
		if rect.Dx() <= 0 || rect.Dy() <= 0 {
			continue
		}

		// Extract region of interest (ROI)
		roi := mat.Region(rect)
		croppedMat := roi.Clone()
		roi.Close()

		// Create output tensor from cropped Mat
		outputTensor, err := tensorgocv.FromMat(croppedMat, tensorgocv.WithAdoptedMat())
		if err != nil {
			continue
		}
		outputTensors = append(outputTensors, outputTensor)
	}

	if len(outputTensors) == 0 {
		return 0
	}

	// Write all output tensors
	written, _ := dst.Write(outputTensors)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// No state to reset
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	// No resources to close
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, CropRectangles] = (*Processor)(nil)
