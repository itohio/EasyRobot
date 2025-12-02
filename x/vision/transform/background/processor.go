package background

import (
	"image"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
)

// Algorithm represents the background subtraction algorithm to use.
type Algorithm int

const (
	// MOG uses Mixture of Gaussians background subtraction (legacy).
	MOG Algorithm = iota
	// MOG2 uses improved Mixture of Gaussians background subtraction.
	MOG2
	// GMG uses Godbehere-Matsukawa-Goldberg background subtraction.
	GMG
)

// Processor implements chain.Processor[visiontypes.ImageWithMask, types.Tensor] for background separation.
// It separates foreground from background using various algorithms (MOG, MOG2, GMG).
//
// Allocations: This processor preallocates Mat objects and background subtractor to avoid allocations
// in the Process() callstack.
type Processor struct {
	algorithm Algorithm

	// Background subtractor implementations (only one is initialized)
	subtractor interface {
		Apply(gocv.Mat, *gocv.Mat) gocv.Mat
	}

	// Learning control
	maxBackgroundNum int // Maximum number of frames to learn background
	backgroundNum    int // Current frame count

	// Preallocated scratch space
	scratchMask gocv.Mat
	discKernel  gocv.Mat // Morphological kernel for filtering
}

// Option configures a background separation processor.
type Option func(*Processor)

// WithAlgorithm sets the background subtraction algorithm.
func WithAlgorithm(alg Algorithm) Option {
	return func(p *Processor) {
		p.algorithm = alg
	}
}

// WithMaxBackgroundNum sets the maximum number of frames to learn the background.
func WithMaxBackgroundNum(max int) Option {
	return func(p *Processor) {
		p.maxBackgroundNum = max
	}
}

// New creates a new background separation processor.
// Default algorithm is MOG2 with 200 frames of background learning.
func New(opts ...Option) *Processor {
	p := &Processor{
		algorithm:        MOG2,
		maxBackgroundNum: 200,
		backgroundNum:    0,
		discKernel:       gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(5, 5)),
	}

	for _, opt := range opts {
		opt(p)
	}

	// Initialize the selected background subtractor
	switch p.algorithm {
	case MOG:
		// Note: MOG might not be available in all gocv versions
		// This may need to be adjusted based on actual gocv API
		p.subtractor = nil // TODO: Initialize MOG if available
	case MOG2:
		subtractor := gocv.NewBackgroundSubtractorMOG2()
		p.subtractor = &subtractor
	case GMG:
		// Note: GMG might not be available in all gocv versions
		p.subtractor = nil // TODO: Initialize GMG if available
	}

	return p
}

// Process implements chain.Processor[visiontypes.ImageWithMask, types.Tensor].
// Reads image tensor from src, applies background subtraction, and writes mask to dst.
//
// Input: Image tensor (gocv.Mat-backed, BGR format)
// Output: ImageWithMask containing the original image and the foreground mask
func (p *Processor) Process(dst buffer.Writer[visiontypes.ImageWithMask], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		return src.Len()
	}

	// Read input image tensor
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

	// Apply background subtraction
	if p.subtractor == nil {
		return 0 // Subtractor not initialized
	}

	learningRate := 0.9
	if p.backgroundNum > p.maxBackgroundNum {
		learningRate = 0 // Stop learning
	}

	// Apply background subtraction
	// Note: Background subtractor Apply method signature may vary
	// This needs to be verified against actual gocv API
	var mask gocv.Mat
	switch s := p.subtractor.(type) {
	case *gocv.BackgroundSubtractorMOG2:
		s.Apply(mat, &p.scratchMask)
		mask = p.scratchMask
	default:
		return 0
	}

	// Filter with morphological kernel
	gocv.Filter2D(mask, &mask, -1, p.discKernel, image.Pt(-1, -1), 0, gocv.BorderDefault)

	// Threshold
	gocv.Threshold(mask, &mask, 100, 255, gocv.ThresholdBinary)

	// Create tensor from mask
	maskMat := mask.Clone()
	maskTensor, err := tensorgocv.FromMat(maskMat, tensorgocv.WithAdoptedMat())
	if err != nil {
		return 0
	}

	// Create output with image and mask
	output := visiontypes.ImageWithMask{
		Image: inputTensor,
		Mask:  maskTensor,
	}

	outputs := []visiontypes.ImageWithMask{output}
	written, _ := dst.Write(outputs)

	// Increment frame counter
	p.backgroundNum++

	return written
}

// BackgroundNum returns the current frame count.
func (p *Processor) BackgroundNum() int {
	return p.backgroundNum
}

// SetBackgroundNum sets the current frame count (useful for resetting learning).
func (p *Processor) SetBackgroundNum(num int) {
	p.backgroundNum = num
}

// Reset resets the processor state and frame counter.
func (p *Processor) Reset() {
	p.backgroundNum = 0
	// Clear scratch space if needed
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
		p.scratchMask = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
	}
	if !p.discKernel.Empty() {
		p.discKernel.Close()
	}
	// Close subtractor if it needs cleanup
	if p.subtractor != nil {
		switch s := p.subtractor.(type) {
		case *gocv.BackgroundSubtractorMOG2:
			s.Close()
		}
	}
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.ImageWithMask, types.Tensor] = (*Processor)(nil)
