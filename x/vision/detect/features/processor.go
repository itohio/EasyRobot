package features

import (
	"image/color"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
)

// FeatureType represents the type of feature detector to use.
type FeatureType int

const (
	ORB FeatureType = iota
	SIFT
	FAST
	GFTT
	KAZE
	AKAZE
	BRISK
)

// Processor implements chain.Processor[visiontypes.KeypointsWithImage, types.Tensor] for feature detection.
// It detects features (keypoints) from input image tensors.
//
// Allocations: This processor preallocates gocv.Mat objects to avoid allocations
// in the Process() callstack. Keypoints slice is allocated per frame (acceptable).
type Processor struct {
	featureType FeatureType

	// Detector implementations (only one is initialized based on featureType)
	orb   gocv.ORB
	sift  gocv.SIFT
	fast  gocv.FastFeatureDetector
	gftt  gocv.GFTTDetector
	kaze  gocv.KAZE
	akaze gocv.AKAZE
	brisk gocv.BRISK

	// Detector interface
	detector interface {
		Detect(gocv.Mat) []gocv.KeyPoint
	}

	// Preallocated scratch space to avoid allocations in Process()
	scratchMask gocv.Mat
}

// New creates a new feature detection processor with the specified feature type.
func New(featureType FeatureType) *Processor {
	p := &Processor{
		featureType: featureType,
		scratchMask: gocv.NewMat(),
	}

	// Initialize the selected detector
	switch featureType {
	case ORB:
		p.orb = gocv.NewORB()
		p.detector = &p.orb
	case SIFT:
		p.sift = gocv.NewSIFT()
		p.detector = &p.sift
	case FAST:
		p.fast = gocv.NewFastFeatureDetector()
		p.detector = &p.fast
	case GFTT:
		p.gftt = gocv.NewGFTTDetector()
		p.detector = &p.gftt
	case KAZE:
		p.kaze = gocv.NewKAZE()
		p.detector = &p.kaze
	case AKAZE:
		p.akaze = gocv.NewAKAZE()
		p.detector = &p.akaze
	case BRISK:
		p.brisk = gocv.NewBRISK()
		p.detector = &p.brisk
	default:
		// Default to ORB
		p.orb = gocv.NewORB()
		p.detector = &p.orb
		p.featureType = ORB
	}

	return p
}

// Process implements chain.Processor[visiontypes.KeypointsWithImage, types.Tensor].
// Reads input tensor(s) from src, detects features (keypoints), and writes KeypointsWithImage to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: KeypointsWithImage (tensor + keypoints)
//
// If dst is nil, returns the estimated number of output items (1 item per input tensor).
func (p *Processor) Process(dst buffer.Writer[visiontypes.KeypointsWithImage], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 1 tensor in -> 1 KeypointsWithImage out
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

	// Detect features (keypoints only, no descriptors)
	keypoints := p.detector.Detect(mat)
	if len(keypoints) == 0 {
		return 0
	}

	// Create KeypointsWithImage output
	// Note: We keep the original tensor reference (don't clone)
	// The caller should ensure the tensor remains valid
	output := visiontypes.KeypointsWithImage{
		Image:     inputTensor,
		Keypoints: keypoints, // Slice is allocated, but this is acceptable for detection
	}

	// Write output
	outputs := []visiontypes.KeypointsWithImage{output}
	written, _ := dst.Write(outputs)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// No state to reset for detector
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	switch p.featureType {
	case ORB:
		p.orb.Close()
	case SIFT:
		p.sift.Close()
	case FAST:
		p.fast.Close()
	case GFTT:
		p.gftt.Close()
	case KAZE:
		p.kaze.Close()
	case AKAZE:
		p.akaze.Close()
	case BRISK:
		p.brisk.Close()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
	}
}

// Draw draws keypoints on the given image tensor.
// The image tensor is modified in place.
// Returns an error if the tensor cannot be converted to gocv.Mat or if drawing fails.
func (p *Processor) Draw(dst types.Tensor, keypoints []gocv.KeyPoint) error {
	if len(keypoints) == 0 {
		return nil
	}

	// Convert tensor to gocv.Mat
	gocvTensor, ok := dst.(tensorgocv.Tensor)
	if !ok {
		return nil // Skip if not a gocv tensor
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return err
	}

	// Draw keypoints on the image
	// Default color is white (255, 255, 255) for RGB
	color := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	gocv.DrawKeyPoints(mat, keypoints, &mat, color, gocv.DrawDefault)

	return nil
}

// DrawKeypoints draws keypoints from KeypointsWithImage on the given image tensor.
// The image tensor is modified in place.
// This is a convenience method that draws keypoints from a KeypointsWithImage structure.
func (p *Processor) DrawKeypoints(dst types.Tensor, kp visiontypes.KeypointsWithImage) error {
	return p.Draw(dst, kp.Keypoints)
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.KeypointsWithImage, types.Tensor] = (*Processor)(nil)
