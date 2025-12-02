package types

import (
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// KeypointsWithImage holds an image tensor and detected keypoints.
// Used to pass keypoints from detection to extraction processors.
type KeypointsWithImage struct {
	Image     types.Tensor
	Keypoints []gocv.KeyPoint
}

// DescriptorsWithImage holds an image tensor, keypoints, and descriptors.
// Used to pass extracted features from extraction to matching processors.
type DescriptorsWithImage struct {
	Image       types.Tensor
	Keypoints   []gocv.KeyPoint
	Descriptors types.Tensor
}

// MatchedFeatures holds two images with their matched features.
// Used to pass matched features to pose estimation processors.
type MatchedFeatures struct {
	Source  DescriptorsWithImage // First image (previous frame)
	Target  DescriptorsWithImage // Second image (current frame)
	Matches []gocv.DMatch        // Matches between source and target
}

// Pose represents a camera pose with rotation and translation.
// Used as output from pose estimation processors.
type Pose struct {
	// Rotation matrix (3x3)
	Rotation gocv.Mat
	// Translation vector (3x1)
	Translation gocv.Mat
	// Inliers mask from RANSAC (optional, for quality assessment)
	Inliers gocv.Mat
}

// Displacement represents a 2D displacement vector (dx, dy) from phase correlation.
// Used as output from phase correlation matching processors.
type Displacement struct {
	// X displacement (pixels)
	DX float64
	// Y displacement (pixels)
	DY float64
	// Correlation peak value (0-1, higher is better match)
	PeakValue float64
}
