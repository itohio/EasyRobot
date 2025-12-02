package types

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"
)

// StereoCamera represents a stereo camera system with two cameras and stereo-specific parameters.
// Similar to Python EasyVision's StereoCamera class.
//
// Contains:
//   - Left and Right: Two Camera instances (PinholeCamera equivalents)
//   - R: 3x3 rotation matrix from right camera to left camera
//   - T: 3x1 translation vector from right camera to left camera
//   - E: 3x3 essential matrix
//   - F: 3x3 fundamental matrix
//   - Q: 4x4 disparity-to-depth matrix (Q matrix)
type StereoCamera struct {
	// Left and right cameras (PinholeCamera equivalents)
	Left  Camera
	Right Camera

	// Stereo extrinsics: rotation and translation from right camera to left camera
	R gocv.Mat // 3x3 rotation matrix
	T gocv.Mat // 3x1 translation vector

	// Essential and fundamental matrices
	E gocv.Mat // 3x3 essential matrix
	F gocv.Mat // 3x3 fundamental matrix

	// Disparity-to-depth matrix
	Q gocv.Mat // 4x4 Q matrix for disparity-to-depth conversion

}

// NewStereoCamera creates a new StereoCamera from two cameras and stereo parameters.
// Takes ownership of the provided Mat objects (they should not be closed externally).
//
// Validates that:
//   - Left and Right cameras are not nil
//   - Left and Right camera sizes match
//
// Returns an error if validation fails.
func NewStereoCamera(left, right Camera, r, t, e, f, q gocv.Mat) (*StereoCamera, error) {
	if left.Size.X != right.Size.X || left.Size.Y != right.Size.Y {
		return nil, fmt.Errorf("left and right camera sizes must match: left=%v, right=%v", left.Size, right.Size)
	}

	return &StereoCamera{
		Left:  left,
		Right: right,
		R:     r,
		T:     t,
		E:     e,
		F:     f,
		Q:     q,
	}, nil
}

// NewStereoCameraFromParameters creates a StereoCamera from individual parameters.
// Similar to Python's StereoCamera.from_parameters().
//
// Parameters:
//   - size: Frame size (width, height) - must match for both cameras
//   - M1, d1: Left camera matrix and distortion coefficients
//   - R1, P1: Left camera rectification and projection matrices (optional, use empty Mat)
//   - M2, d2: Right camera matrix and distortion coefficients
//   - R2, P2: Right camera rectification and projection matrices (optional, use empty Mat)
//   - R, T: Stereo rotation and translation
//   - E, F: Essential and fundamental matrices
//   - Q: Disparity-to-depth matrix
func NewStereoCameraFromParameters(
	size image.Point,
	m1, d1, r1, p1 gocv.Mat, // Left camera
	m2, d2, r2, p2 gocv.Mat, // Right camera
	r, t, e, f, q gocv.Mat, // Stereo parameters
) (*StereoCamera, error) {
	leftCam := NewCameraWithRectification(size, m1, d1, r1, p1)
	rightCam := NewCameraWithRectification(size, m2, d2, r2, p2)

	return NewStereoCamera(leftCam, rightCam, r, t, e, f, q)
}

// Baseline returns the baseline distance between the two cameras in the same units as calibration.
// The baseline is the magnitude of the translation vector T.
func (sc *StereoCamera) Baseline() float64 {
	if sc.T.Empty() {
		return 0
	}
	// Baseline is the magnitude of the translation vector
	tx := sc.T.GetDoubleAt(0, 0)
	ty := sc.T.GetDoubleAt(1, 0)
	tz := sc.T.GetDoubleAt(2, 0)
	return math.Sqrt(tx*tx + ty*ty + tz*tz)
}

// TriangulatePoints triangulates 3D points from matched 2D points in left and right images.
// leftPoints: Nx2 Mat of points in left image (normalized or pixel coordinates)
// rightPoints: Nx2 Mat of corresponding points in right image
// Returns: Nx3 Mat of triangulated 3D points
//
// Note: Requires projection matrices to be set in Left.Projection and Right.Projection.
// Rectification matrices (R1, R2) and projection matrices (P1, P2) are stored in the Camera objects.
func (sc *StereoCamera) TriangulatePoints(leftPoints, rightPoints gocv.Mat) (gocv.Mat, error) {
	if sc.Left == nil || sc.Right == nil {
		return gocv.NewMat(), fmt.Errorf("left or right camera is nil")
	}
	if sc.Left.Projection.Empty() || sc.Right.Projection.Empty() {
		return gocv.NewMat(), fmt.Errorf("projection matrices are not set - set Left.Projection and Right.Projection")
	}

	points3D := gocv.NewMat()

	// TODO: Check if gocv has TriangulatePoints function
	// If not, implement using least squares triangulation from projection matrices
	// Can use: sc.Left.Projection and sc.Right.Projection for triangulation
	// For now, return empty mat as placeholder
	return points3D, fmt.Errorf("TriangulatePoints not yet implemented - check gocv API availability or implement manually using projection matrices")
}

// ReprojectTo3D converts disparity map to 3D points using the Q matrix.
// disparity: Disparity map (single channel, CV_16SC1 or CV_32F)
// Returns: 3-channel Mat (X, Y, Z coordinates) or depth map
//
// This uses gocv.ReprojectImageTo3D which is already verified to exist in the codebase.
func (sc *StereoCamera) ReprojectTo3D(disparity gocv.Mat) (gocv.Mat, error) {
	if sc.Q.Empty() {
		return gocv.NewMat(), fmt.Errorf("Q matrix (disparity-to-depth map) is not set")
	}

	points3D := gocv.NewMat()
	err := gocv.ReprojectImageTo3D(disparity, &points3D, sc.Q, false, gocv.MatTypeCV32FC1)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to reproject disparity to 3D: %w", err)
	}

	return points3D, nil
}

// Rotation returns the rotation matrix R (read-only reference).
// Returns an empty Mat if not set.
func (sc *StereoCamera) Rotation() gocv.Mat {
	if sc.R.Empty() {
		return gocv.NewMat()
	}
	return sc.R
}

// Translation returns the translation vector T (read-only reference).
// Returns an empty Mat if not set.
func (sc *StereoCamera) Translation() gocv.Mat {
	if sc.T.Empty() {
		return gocv.NewMat()
	}
	return sc.T
}

// Essential returns the essential matrix E (read-only reference).
// Returns an empty Mat if not set.
func (sc *StereoCamera) Essential() gocv.Mat {
	if sc.E.Empty() {
		return gocv.NewMat()
	}
	return sc.E
}

// Fundamental returns the fundamental matrix F (read-only reference).
// Returns an empty Mat if not set.
func (sc *StereoCamera) Fundamental() gocv.Mat {
	if sc.F.Empty() {
		return gocv.NewMat()
	}
	return sc.F
}

// Q returns the disparity-to-depth Q matrix (read-only reference).
// Returns an empty Mat if not set.
func (sc *StereoCamera) Q() gocv.Mat {
	if sc.Q.Empty() {
		return gocv.NewMat()
	}
	return sc.Q
}

// Close releases all resources held by the stereo camera.
func (sc *StereoCamera) Close() {
	if sc.Left != nil {
		sc.Left.Close()
	}
	if sc.Right != nil {
		sc.Right.Close()
	}
	if !sc.R.Empty() {
		sc.R.Close()
	}
	if !sc.T.Empty() {
		sc.T.Close()
	}
	if !sc.E.Empty() {
		sc.E.Close()
	}
	if !sc.F.Empty() {
		sc.F.Close()
	}
	if !sc.Q.Empty() {
		sc.Q.Close()
	}
}
