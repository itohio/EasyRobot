package pose

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[visiontypes.Pose, visiontypes.MatchedFeatures] for pose estimation.
// It computes the relative camera pose (rotation and translation) from matched features between two images.
//
// Allocations: This processor preallocates Mat objects to avoid allocations in the Process() callstack.
type Processor struct {
	// Camera intrinsic matrix (3x3) - required for essential matrix computation
	cameraMatrix gocv.Mat

	// Preallocated scratch space to avoid allocations in Process()
	scratchPoints1   gocv.Mat       // Matched points from first image
	scratchPoints2   gocv.Mat       // Matched points from second image
	scratchEssential gocv.Mat       // Essential matrix (3x3)
	scratchR         gocv.Mat       // Rotation matrix (3x3)
	scratchT         gocv.Mat       // Translation vector (3x1)
	scratchMask      gocv.Mat       // Inliers mask from RANSAC
	scratchPointsBuf []gocv.Point2f // Reusable slice buffer for point extraction

	// RANSAC parameters
	ransacThreshold float64 // Threshold for RANSAC (typically 3.0)
	confidence      float64 // Confidence level (typically 0.999)
}

// New creates a new pose estimation processor with the given camera intrinsic matrix.
// The camera matrix is not copied; the processor takes a reference to it.
// Preallocates buffers for up to maxMatches matches (default 1000).
func New(cameraMatrix gocv.Mat) *Processor {
	const maxMatches = 1000 // Preallocate for up to 1000 matched points
	return &Processor{
		cameraMatrix:     cameraMatrix,
		scratchPoints1:   gocv.NewMatWithSize(maxMatches, 1, gocv.MatTypeCV32FC2),
		scratchPoints2:   gocv.NewMatWithSize(maxMatches, 1, gocv.MatTypeCV32FC2),
		scratchEssential: gocv.NewMat(),
		scratchR:         gocv.NewMat(),
		scratchT:         gocv.NewMat(),
		scratchMask:      gocv.NewMat(),
		scratchPointsBuf: make([]gocv.Point2f, 0, maxMatches*2),
		ransacThreshold:  3.0,
		confidence:       0.999,
	}
}

// Process implements chain.Processor[visiontypes.Pose, visiontypes.MatchedFeatures].
// Reads MatchedFeatures from src, computes pose, and writes Pose to dst.
//
// Input: MatchedFeatures (two images with matched keypoints)
// Output: Pose (rotation matrix + translation vector + inliers mask)
//
// If dst is nil, returns the estimated number of output poses (1 pose per input match set).
func (p *Processor) Process(dst buffer.Writer[visiontypes.Pose], src buffer.Reader[visiontypes.MatchedFeatures]) int {
	if dst == nil {
		// Size estimation: 1 MatchedFeatures in -> 1 Pose out
		return src.Len()
	}

	// Read input matched features
	matched := make([]visiontypes.MatchedFeatures, 1)
	n, err := src.Read(matched)
	if err != nil || n == 0 {
		return 0
	}

	m := matched[0]

	// Extract matched keypoints as Point2f arrays
	// Note: Small allocation for slices is acceptable here as they're just references
	points1, points2 := extractMatchedPoints(m)
	if len(points1) < 5 {
		// Need at least 5 points for 5-point algorithm
		return 0
	}

	numPoints := len(points1)
	maxRows := p.scratchPoints1.Rows()

	// Check if we need to resize buffers (should be rare if preallocated properly)
	var points1Mat, points2Mat gocv.Mat
	var needClose bool
	if numPoints > maxRows {
		// Need larger buffers - create new Mat objects of exact size (rare case)
		points1Mat = gocv.NewMatWithSize(numPoints, 1, gocv.MatTypeCV32FC2)
		points2Mat = gocv.NewMatWithSize(numPoints, 1, gocv.MatTypeCV32FC2)
		needClose = true
	} else {
		// Use preallocated buffers
		points1Mat = p.scratchPoints1
		points2Mat = p.scratchPoints2
		needClose = false
	}

	// Fill the Mat objects with point coordinates
	// For CV_32FC2 Mat, use SetFloatAt3(row, col, channel, value)
	for i := range points1 {
		if i < numPoints {
			points1Mat.SetFloatAt3(i, 0, 0, points1[i].X) // row i, col 0, channel 0 (x)
			points1Mat.SetFloatAt3(i, 0, 1, points1[i].Y) // row i, col 0, channel 1 (y)
			points2Mat.SetFloatAt3(i, 0, 0, points2[i].X) // row i, col 0, channel 0 (x)
			points2Mat.SetFloatAt3(i, 0, 1, points2[i].Y) // row i, col 0, channel 1 (y)
		}
	}

	// TODO: gocv doesn't expose FindEssentialMat or FindFundamentalMat.
	// Essential matrix computation requires manual implementation using:
	// - 5-point algorithm (Nister's method) or 8-point algorithm
	// - RANSAC for robust estimation
	// - Conversion from fundamental to essential: E = K^T * F * K
	// - RecoverPose to extract rotation and translation from essential matrix
	//
	// For now, return 0 to indicate this is not yet implemented.
	// This processor requires either:
	// 1. gocv bindings for findEssentialMat/findFundamentalMat/recoverPose, or
	// 2. Manual implementation of essential matrix estimation
	if needClose {
		points1Mat.Close()
		points2Mat.Close()
	}
	return 0
}

// extractMatchedPoints extracts matched keypoints from MatchedFeatures.
// Returns two slices of Point2f corresponding to matched keypoints.
// Note: Small slice allocations are acceptable here as they're just references to keypoint data.
func extractMatchedPoints(m visiontypes.MatchedFeatures) ([]gocv.Point2f, []gocv.Point2f) {
	if len(m.Matches) == 0 {
		return nil, nil
	}

	points1 := make([]gocv.Point2f, 0, len(m.Matches))
	points2 := make([]gocv.Point2f, 0, len(m.Matches))

	for _, match := range m.Matches {
		if match.QueryIdx >= 0 && match.QueryIdx < len(m.Source.Keypoints) &&
			match.TrainIdx >= 0 && match.TrainIdx < len(m.Target.Keypoints) {
			kp1 := m.Source.Keypoints[match.QueryIdx]
			kp2 := m.Target.Keypoints[match.TrainIdx]
			points1 = append(points1, gocv.Point2f{X: float32(kp1.X), Y: float32(kp1.Y)})
			points2 = append(points2, gocv.Point2f{X: float32(kp2.X), Y: float32(kp2.Y)})
		}
	}

	return points1, points2
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// Clear scratch space if needed
	if !p.scratchPoints1.Empty() {
		p.scratchPoints1.Close()
		p.scratchPoints1 = gocv.NewMat()
	}
	if !p.scratchPoints2.Empty() {
		p.scratchPoints2.Close()
		p.scratchPoints2 = gocv.NewMat()
	}
	if !p.scratchEssential.Empty() {
		p.scratchEssential.Close()
		p.scratchEssential = gocv.NewMat()
	}
	if !p.scratchR.Empty() {
		p.scratchR.Close()
		p.scratchR = gocv.NewMat()
	}
	if !p.scratchT.Empty() {
		p.scratchT.Close()
		p.scratchT = gocv.NewMat()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
		p.scratchMask = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchPoints1.Empty() {
		p.scratchPoints1.Close()
	}
	if !p.scratchPoints2.Empty() {
		p.scratchPoints2.Close()
	}
	if !p.scratchEssential.Empty() {
		p.scratchEssential.Close()
	}
	if !p.scratchR.Empty() {
		p.scratchR.Close()
	}
	if !p.scratchT.Empty() {
		p.scratchT.Close()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
	}
	// Note: We don't close cameraMatrix as it is owned by the caller
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.Pose, visiontypes.MatchedFeatures] = (*Processor)(nil)
