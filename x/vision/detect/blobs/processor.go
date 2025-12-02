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

// Processor implements chain.Processor[visiontypes.KeypointsWithImage, types.Tensor] for blob detection.
// It detects blobs from input image tensors using SimpleBlobDetector.
//
// Allocations: This processor preallocates Mat objects and blob detector to avoid allocations
// in the Process() callstack.
type Processor struct {
	// Blob detector
	detector gocv.SimpleBlobDetector

	// Blob detection parameters
	blurSize image.Point

	// Preallocated scratch space
	scratchGray    gocv.Mat
	scratchMask    gocv.Mat
	scratchBlurred gocv.Mat
}

// Option configures a blob detection processor.
type Option func(*Processor)

// WithBlurSize sets the blur kernel size for preprocessing.
func WithBlurSize(size image.Point) Option {
	return func(p *Processor) {
		p.blurSize = size
	}
}

// WithMinArea sets the minimum blob area.
func WithMinArea(minArea int) Option {
	return func(p *Processor) {
		// Note: This requires recreating the detector with new params
		// For now, use NewWithArea for custom area constraints
	}
}

// WithMaxArea sets the maximum blob area.
func WithMaxArea(maxArea int) Option {
	return func(p *Processor) {
		// Note: This requires recreating the detector with new params
		// For now, use NewWithArea for custom area constraints
	}
}

// New creates a new blob detection processor.
// Default blur size is (30, 30).
func New(opts ...Option) *Processor {
	// Create blob detector with default parameters
	params := gocv.NewSimpleBlobDetectorParams()
	params.SetFilterByArea(true)
	params.SetMinArea(100)
	params.SetMaxArea(400 * 400)

	detector := gocv.NewSimpleBlobDetectorWithParams(params)

	p := &Processor{
		detector:       detector,
		blurSize:       image.Pt(30, 30),
		scratchGray:    gocv.NewMat(),
		scratchMask:    gocv.NewMat(),
		scratchBlurred: gocv.NewMat(),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// NewWithArea creates a blob detector with custom area constraints.
func NewWithArea(minArea, maxArea int, blurSize image.Point) *Processor {
	params := gocv.NewSimpleBlobDetectorParams()
	params.SetFilterByArea(true)
	params.SetMinArea(float64(minArea))
	params.SetMaxArea(float64(maxArea))

	detector := gocv.NewSimpleBlobDetectorWithParams(params)

	return &Processor{
		detector:       detector,
		blurSize:       blurSize,
		scratchGray:    gocv.NewMat(),
		scratchMask:    gocv.NewMat(),
		scratchBlurred: gocv.NewMat(),
	}
}

// Process implements chain.Processor[visiontypes.KeypointsWithImage, types.Tensor].
// Reads image tensor from src, detects blobs, and writes KeypointsWithImage to dst.
//
// Input: Image tensor (gocv.Mat-backed, BGR or grayscale)
// Output: KeypointsWithImage containing image and detected blob keypoints
func (p *Processor) Process(dst buffer.Writer[visiontypes.KeypointsWithImage], src buffer.Reader[types.Tensor]) int {
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

	// Convert to grayscale if needed
	var grayMat gocv.Mat
	if mat.Channels() == 1 {
		grayMat = mat
	} else {
		gocv.CvtColor(mat, &p.scratchGray, gocv.ColorBGRToGray)
		grayMat = p.scratchGray
	}

	// Invert image (blobs are typically white on black background)
	gocv.BitwiseNot(grayMat, &p.scratchMask)

	// Blur the mask
	gocv.Blur(p.scratchMask, &p.scratchBlurred, p.blurSize)

	// Threshold
	gocv.Threshold(p.scratchBlurred, &p.scratchMask, 50, 255, gocv.ThresholdBinary)

	// Detect blobs
	keypoints := p.detector.Detect(p.scratchMask)

	// Create output
	output := visiontypes.KeypointsWithImage{
		Image:     inputTensor, // Keep reference to original image
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
	color := color.RGBA{R: 0, G: 0, B: 255, A: 255} // Blue
	gocv.DrawKeyPoints(mat, keypoints, &mat, color, gocv.DrawRichKeyPoints)

	return nil
}

// DrawKeypoints draws keypoints from KeypointsWithImage on the destination tensor.
func (p *Processor) DrawKeypoints(dst types.Tensor, kp visiontypes.KeypointsWithImage) error {
	return p.Draw(dst, kp.Keypoints)
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// Clear scratch space if needed
	if !p.scratchGray.Empty() {
		p.scratchGray.Close()
		p.scratchGray = gocv.NewMat()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
		p.scratchMask = gocv.NewMat()
	}
	if !p.scratchBlurred.Empty() {
		p.scratchBlurred.Close()
		p.scratchBlurred = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchGray.Empty() {
		p.scratchGray.Close()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
	}
	if !p.scratchBlurred.Empty() {
		p.scratchBlurred.Close()
	}
	p.detector.Close()
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.KeypointsWithImage, types.Tensor] = (*Processor)(nil)
