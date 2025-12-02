package blobs

import (
	"image"
	"image/color"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[visiontypes.KeypointsWithImage, visiontypes.ImageWithMask] for blob extraction.
// It extracts blobs from masks using SimpleBlobDetector.
//
// Allocations: This processor preallocates Mat objects and blob detector to avoid allocations
// in the Process() callstack.
type Processor struct {
	// Blob detector
	detector gocv.SimpleBlobDetector

	// Blob detection parameters
	blurSize image.Point

	// Preallocated scratch space
	scratchBlurred gocv.Mat
	scratchMask    gocv.Mat
}

// Option configures a blob extraction processor.
type Option func(*Processor)

// WithBlurSize sets the blur kernel size for preprocessing masks.
func WithBlurSize(size image.Point) Option {
	return func(p *Processor) {
		p.blurSize = size
	}
}

// WithBlobParams configures blob detection parameters.
func WithBlobParams(params gocv.SimpleBlobDetectorParams) Option {
	return func(p *Processor) {
		// Note: This would need to be applied during detector creation
		// For now, we'll use default params
	}
}

// New creates a new blob extraction processor.
// Default blur size is (30, 30).
func New(opts ...Option) *Processor {
	// Create blob detector with default parameters
	params := gocv.NewSimpleBlobDetectorParams()
	params.SetFilterByArea(true)
	params.SetMinArea(100)
	params.SetMaxArea(400 * 400)
	
	detector := gocv.NewSimpleBlobDetectorWithParams(params)

	p := &Processor{
		detector:  detector,
		blurSize:  image.Pt(30, 30),
		scratchMask: gocv.NewMat(),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// NewWithArea creates a blob extractor with custom area constraints.
func NewWithArea(minArea, maxArea int, blurSize image.Point) *Processor {
	params := gocv.NewSimpleBlobDetectorParams()
	params.SetFilterByArea(true)
	params.SetMinArea(minArea)
	params.SetMaxArea(maxArea)
	
	detector := gocv.NewSimpleBlobDetectorWithParams(params)

	return &Processor{
		detector:     detector,
		blurSize:     blurSize,
		scratchMask:  gocv.NewMat(),
	}
}

// Process implements chain.Processor[visiontypes.KeypointsWithImage, visiontypes.ImageWithMask].
// Reads ImageWithMask from src, extracts blobs, and writes KeypointsWithImage to dst.
//
// Input: ImageWithMask containing image and mask
// Output: KeypointsWithImage containing image and detected blob keypoints
func (p *Processor) Process(dst buffer.Writer[visiontypes.KeypointsWithImage], src buffer.Reader[visiontypes.ImageWithMask]) int {
	if dst == nil {
		return src.Len()
	}

	// Read input
	inputData := make([]visiontypes.ImageWithMask, 1)
	n, err := src.Read(inputData)
	if err != nil || n == 0 {
		return 0
	}

	input := inputData[0]

	// Get mask tensor
	maskTensor := input.Mask
	maskGocv, ok := maskTensor.(tensorgocv.Tensor)
	if !ok {
		return 0
	}

	maskMat, err := maskGocv.Mat()
	if err != nil {
		return 0
	}

	if maskMat.Empty() {
		return 0
	}

	// Invert mask (blobs are typically white on black)
	gocv.BitwiseNot(maskMat, &p.scratchMask)

	// Blur the mask
	gocv.Blur(p.scratchMask, &p.scratchMask, p.blurSize)

	// Threshold
	gocv.Threshold(p.scratchMask, &p.scratchMask, 50, 255, gocv.ThresholdBinary)

	// Detect blobs
	keypoints := p.detector.Detect(p.scratchMask)

	// Create output
	output := visiontypes.KeypointsWithImage{
		Image:     input.Image, // Keep reference to original image
		Keypoints: keypoints,
	}

	outputs := []visiontypes.KeypointsWithImage{output}
	written, _ := dst.Write(outputs)
	return written
}

// Draw draws the detected blobs on the image.
func (p *Processor) Draw(dst types.Tensor, keypoints []gocv.KeyPoint) error {
	if len(keypoints) == 0 {
		return nil
	}

	// Convert tensor to gocv.Mat
	gocvTensor, ok := dst.(tensorgocv.Tensor)
	if !ok {
		return nil
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return err
	}

	// Draw keypoints
	color := color.RGBA{R: 0, G: 0, B: 255, A: 255} // Red
	gocv.DrawKeyPoints(mat, keypoints, &mat, color, gocv.DrawRichKeyPoints)

	return nil
}

// Reset resets the processor state.
func (p *Processor) Reset() {
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
	p.detector.Close()
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.KeypointsWithImage, visiontypes.ImageWithMask] = (*Processor)(nil)

