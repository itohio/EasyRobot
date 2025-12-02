package rectify

import (
	"fmt"
	"image/color"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// RectificationMaps holds the remapping matrices for stereo rectification.
type RectificationMaps struct {
	Map1 gocv.Mat // First map (x-coordinates or CV_16SC2)
	Map2 gocv.Mat // Second map (y-coordinates or CV_16UC1)
}

// Processor implements chain.Processor[types.Tensor, types.Tensor] for stereo image rectification.
// It rectifies stereo images using precomputed rectification maps.
//
// Allocations: This processor preallocates output Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	// Rectification maps (precomputed via InitUndistortRectifyMap)
	maps RectificationMaps

	// Preallocated scratch space to avoid allocations in Process()
	scratchRectified gocv.Mat
}

// New creates a new stereo rectification processor with the given rectification maps.
// The maps are not copied; the processor takes ownership of them.
func New(maps RectificationMaps) *Processor {
	return &Processor{
		maps:             maps,
		scratchRectified: gocv.NewMat(),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads input tensor(s) from src, rectifies them, and writes output tensor(s) to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: Rectified image tensor (gocv.Mat-backed)
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

	// Rectify the image using remapping
	// Remap will resize scratchRectified to match input size if needed
	borderValue := color.RGBA{R: 0, G: 0, B: 0, A: 0}
	gocv.Remap(mat, &p.scratchRectified, &p.maps.Map1, &p.maps.Map2, gocv.InterpolationLinear, gocv.BorderConstant, borderValue)

	// Create output tensor from rectified Mat
	// Clone the rectified image to create a new tensor
	outputMat := p.scratchRectified.Clone()
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
	if !p.scratchRectified.Empty() {
		p.scratchRectified.Close()
		p.scratchRectified = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchRectified.Empty() {
		p.scratchRectified.Close()
	}
	// Note: We don't close maps.Map1 and maps.Map2 here as they are owned by the caller
}

// Rectify rectifies an image using precomputed rectification maps.
// This is a direct helper function that can be called without using the processor.
// Returns the rectified image Mat. The caller is responsible for closing the returned Mat.
func Rectify(src gocv.Mat, maps RectificationMaps) (gocv.Mat, error) {
	if maps.Map1.Empty() || maps.Map2.Empty() {
		return gocv.Mat{}, fmt.Errorf("rectification maps not initialized")
	}

	dst := gocv.NewMat()
	borderValue := color.RGBA{R: 0, G: 0, B: 0, A: 0}
	gocv.Remap(src, &dst, &maps.Map1, &maps.Map2, gocv.InterpolationLinear, gocv.BorderConstant, borderValue)
	return dst, nil
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)
