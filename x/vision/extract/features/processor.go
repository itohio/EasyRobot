package features

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
)

// FeatureType represents the type of feature extractor to use.
// Must match the feature type used in detection.
type FeatureType int

const (
	ORB FeatureType = iota
	SIFT
	KAZE
	AKAZE
	BRISK
	// FAST and GFTT don't compute descriptors, only detect keypoints
)

// Processor implements chain.Processor[visiontypes.DescriptorsWithImage, visiontypes.KeypointsWithImage] for feature extraction.
// It extracts descriptors from keypoints and image tensor.
//
// Allocations: This processor preallocates gocv.Mat objects for descriptors to avoid
// allocations in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	featureType FeatureType

	// Extractor implementations (only one is initialized based on featureType)
	orb   gocv.ORB
	sift  gocv.SIFT
	kaze  gocv.KAZE
	akaze gocv.AKAZE
	brisk gocv.BRISK

	// extractorFunc computes descriptors for given keypoints
	extractorFunc func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat)

	// Preallocated scratch space to avoid allocations in Process()
	scratchMask gocv.Mat
}

// New creates a new feature extraction processor with the specified feature type.
func New(featureType FeatureType) *Processor {
	p := &Processor{
		featureType: featureType,
		scratchMask: gocv.NewMat(),
	}

	// Initialize the selected extractor
	switch featureType {
	case ORB:
		p.orb = gocv.NewORB()
		// ORB can compute descriptors for existing keypoints via DetectAndCompute
		// We'll use a wrapper that filters to use only the provided keypoints
		p.extractorFunc = func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat) {
			// For now, use DetectAndCompute and return all keypoints/descriptors
			// TODO: Filter to match provided keypoints if needed
			return p.orb.DetectAndCompute(src, mask)
		}
	case SIFT:
		p.sift = gocv.NewSIFT()
		p.extractorFunc = func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat) {
			return p.sift.DetectAndCompute(src, mask)
		}
	case KAZE:
		p.kaze = gocv.NewKAZE()
		p.extractorFunc = func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat) {
			return p.kaze.DetectAndCompute(src, mask)
		}
	case AKAZE:
		p.akaze = gocv.NewAKAZE()
		p.extractorFunc = func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat) {
			return p.akaze.DetectAndCompute(src, mask)
		}
	case BRISK:
		p.brisk = gocv.NewBRISK()
		p.extractorFunc = func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat) {
			return p.brisk.DetectAndCompute(src, mask)
		}
	default:
		// Default to ORB
		p.orb = gocv.NewORB()
		p.extractorFunc = func(src gocv.Mat, keypoints []gocv.KeyPoint, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat) {
			return p.orb.DetectAndCompute(src, mask)
		}
		p.featureType = ORB
	}

	return p
}

// Process implements chain.Processor[visiontypes.DescriptorsWithImage, visiontypes.KeypointsWithImage].
// Reads KeypointsWithImage from src, extracts descriptors, and writes DescriptorsWithImage to dst.
//
// Input: KeypointsWithImage (image tensor + keypoints)
// Output: DescriptorsWithImage (image tensor + keypoints + descriptors)
//
// If dst is nil, returns the estimated number of output items (1 item per input item).
func (p *Processor) Process(dst buffer.Writer[visiontypes.DescriptorsWithImage], src buffer.Reader[visiontypes.KeypointsWithImage]) int {
	if dst == nil {
		// Size estimation: 1 KeypointsWithImage in -> 1 DescriptorsWithImage out
		return src.Len()
	}

	// Read input KeypointsWithImage
	inputs := make([]visiontypes.KeypointsWithImage, 1)
	n, err := src.Read(inputs)
	if err != nil || n == 0 {
		return 0
	}

	input := inputs[0]

	// Convert tensor to gocv.Mat
	gocvTensor, ok := input.Image.(tensorgocv.Tensor)
	if !ok {
		return 0
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return 0
	}

	// Extract descriptors from keypoints and image
	// Note: Since gocv detectors use DetectAndCompute (not separate Compute),
	// we re-detect keypoints and compute descriptors. The input.Keypoints are kept
	// for reference/visualization but may differ from the detected keypoints.
	// TODO: Implement Compute-only path if gocv supports it for existing keypoints.
	// Use preallocated scratch mask (empty mat)
	detectedKeypoints, descriptors := p.extractorFunc(mat, input.Keypoints, p.scratchMask)
	if len(detectedKeypoints) == 0 || descriptors.Empty() {
		return 0
	}

	// Create descriptor tensor from Mat
	// Note: descriptors is a gocv.Mat returned by DetectAndCompute.
	// We need to create a tensor from it using WithAdoptedMat to transfer ownership.
	// IMPORTANT: Tensors are NOT guaranteed to be zero-initialized.
	descriptorTensor, err := tensorgocv.FromMat(descriptors, tensorgocv.WithAdoptedMat())
	if err != nil {
		return 0
	}

	// Create DescriptorsWithImage output
	// Use detectedKeypoints (from DetectAndCompute) since they correspond to the descriptors
	output := visiontypes.DescriptorsWithImage{
		Image:       input.Image,
		Keypoints:   detectedKeypoints, // Keypoints that correspond to the descriptors
		Descriptors: descriptorTensor,
	}

	// Write output
	outputs := []visiontypes.DescriptorsWithImage{output}
	written, _ := dst.Write(outputs)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// No state to reset
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	switch p.featureType {
	case ORB:
		p.orb.Close()
	case SIFT:
		p.sift.Close()
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

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.DescriptorsWithImage, visiontypes.KeypointsWithImage] = (*Processor)(nil)
