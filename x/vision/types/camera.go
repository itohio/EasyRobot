package types

import (
	"fmt"
	"image"

	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Camera represents a calibrated pinhole camera model with intrinsic parameters.
// Similar to Python EasyVision's PinholeCamera class.
type Camera struct {
	// Image resolution (width, height) in pixels
	Size image.Point

	// Camera matrix (K) - 3x3 intrinsic matrix
	// [fx  0  cx]
	// [ 0 fy  cy]
	// [ 0  0   1]
	CameraMatrix gocv.Mat

	// Distortion coefficients (k1, k2, p1, p2, k3, ...)
	// Typically 1x5 or 1x8 depending on distortion model
	DistortionCoeffs gocv.Mat

	// Rectification matrix (R) - 3x3 rotation matrix for image rectification (optional)
	Rectification gocv.Mat

	// Projection matrix (P) - 3x3 or 3x4 projection matrix after rectification (optional)
	Projection gocv.Mat
}

// NewCamera creates a new Camera from the given parameters.
// Takes ownership of the provided Mat objects (they should not be closed externally).
func NewCamera(size image.Point, cameraMatrix, distortionCoeffs gocv.Mat) *Camera {
	return &Camera{
		Size:             size,
		CameraMatrix:     cameraMatrix,
		DistortionCoeffs: distortionCoeffs,
	}
}

// NewCameraWithRectification creates a Camera with rectification and projection matrices.
func NewCameraWithRectification(size image.Point, cameraMatrix, distortionCoeffs, rectification, projection gocv.Mat) *Camera {
	return &Camera{
		Size:             size,
		CameraMatrix:     cameraMatrix,
		DistortionCoeffs: distortionCoeffs,
		Rectification:    rectification,
		Projection:       projection,
	}
}

// NewCameraFromParameters creates a Camera from individual parameters.
// Useful for constructing a camera from calibration parameters.
func NewCameraFromParameters(size image.Point, focalPoint image.Point, center image.Point, distortion []float64, rectification, projection gocv.Mat) (*Camera, error) {
	// Create camera matrix (3x3)
	cameraMatrix := gocv.NewMatWithSize(3, 3, gocv.MatTypeCV64F)
	cameraMatrix.SetDoubleAt(0, 0, float64(focalPoint.X))
	cameraMatrix.SetDoubleAt(1, 1, float64(focalPoint.Y))
	cameraMatrix.SetDoubleAt(0, 2, float64(center.X))
	cameraMatrix.SetDoubleAt(1, 2, float64(center.Y))
	cameraMatrix.SetDoubleAt(2, 2, 1.0)

	// Create distortion coefficients (1xN where N is number of coefficients)
	if len(distortion) == 0 {
		return nil, fmt.Errorf("distortion must have at least one coefficient")
	}
	distortionMat := gocv.NewMatWithSize(1, len(distortion), gocv.MatTypeCV64F)
	for i, d := range distortion {
		distortionMat.SetDoubleAt(0, i, d)
	}

	return &Camera{
		Size:             size,
		CameraMatrix:     cameraMatrix,
		DistortionCoeffs: distortionMat,
		Rectification:    rectification,
		Projection:       projection,
	}, nil
}

// Width returns the width of the camera sensor in pixels.
func (c *Camera) Width() int {
	return c.Size.X
}

// Height returns the height of the camera sensor in pixels.
func (c *Camera) Height() int {
	return c.Size.Y
}

// FocalPoint returns the focal point (fx, fy) in pixels.
// These are the diagonal elements of the camera matrix.
func (c *Camera) FocalPoint() image.Point {
	if c.CameraMatrix.Empty() {
		return image.Point{}
	}
	return image.Point{
		X: int(c.CameraMatrix.GetDoubleAt(0, 0)),
		Y: int(c.CameraMatrix.GetDoubleAt(1, 1)),
	}
}

// Center returns the principal point (cx, cy) in pixels.
// These are the translation elements of the camera matrix.
func (c *Camera) Center() image.Point {
	if c.CameraMatrix.Empty() {
		return image.Point{}
	}
	return image.Point{
		X: int(c.CameraMatrix.GetDoubleAt(0, 2)),
		Y: int(c.CameraMatrix.GetDoubleAt(1, 2)),
	}
}

// CameraMatrixTensor returns the camera matrix as a tensor.
// Returns a new tensor that shares data with the underlying Mat.
func (c *Camera) CameraMatrixTensor() (types.Tensor, error) {
	if c.CameraMatrix.Empty() {
		return nil, fmt.Errorf("camera matrix is empty")
	}
	return tensorgocv.FromMat(c.CameraMatrix.Clone(), tensorgocv.WithAdoptedMat())
}

// DistortionCoeffsTensor returns the distortion coefficients as a tensor.
func (c *Camera) DistortionCoeffsTensor() (types.Tensor, error) {
	if c.DistortionCoeffs.Empty() {
		return nil, fmt.Errorf("distortion coefficients are empty")
	}
	return tensorgocv.FromMat(c.DistortionCoeffs.Clone(), tensorgocv.WithAdoptedMat())
}

// ProjectPoints projects 3D object points onto the 2D image plane.
// objectPoints: Nx3 or 3xN Mat of 3D object points
// rvec: 3x1 rotation vector (Rodrigues representation)
// tvec: 3x1 translation vector
// Returns: Nx2 Mat of projected 2D image points
func (c *Camera) ProjectPoints(objectPoints gocv.Mat, rvec, tvec gocv.Mat) (gocv.Mat, error) {
	imagePoints := gocv.NewMat()

	// Note: gocv may not have ProjectPoints directly exposed
	// This is a placeholder that would need to be implemented based on actual gocv API
	// or manually computed using: imagePoints = K * (R * objectPoints + t)
	// where K is camera matrix, R is rotation from rvec, t is translation

	// TODO: Implement projection based on actual gocv API availability
	// For now, return empty mat as placeholder
	return imagePoints, fmt.Errorf("ProjectPoints not yet implemented - check gocv API availability")
}

// BackprojectPoints converts 2D image points to 3D rays (direction vectors).
// imagePoints: Nx2 Mat of 2D image points
// Returns: Nx3 Mat of 3D direction vectors (not actual 3D points, but rays)
func (c *Camera) BackprojectPoints(imagePoints gocv.Mat) (gocv.Mat, error) {
	// TODO: Implement backprojection
	// This involves: (x-cx)/fx, (y-cy)/fy, 1.0 for each point
	// Then normalize to unit vectors
	return gocv.NewMat(), fmt.Errorf("BackprojectPoints not yet implemented")
}

// UndistortPoints corrects image points for lens distortion.
// distortedPoints: Nx2 Mat of distorted 2D image points
// Returns: Nx2 Mat of undistorted 2D image points
// Note: gocv.UndistortPoints may not be available in all versions.
// If unavailable, implement using manual undistortion or use Undistort with sparse maps.
func (c *Camera) UndistortPoints(distortedPoints gocv.Mat) (gocv.Mat, error) {
	if c.CameraMatrix.Empty() || c.DistortionCoeffs.Empty() {
		return gocv.NewMat(), fmt.Errorf("camera matrix or distortion coefficients are empty")
	}

	undistortedPoints := gocv.NewMat()

	// TODO: Verify if gocv.UndistortPoints exists in the version being used
	// If not available, implement manual undistortion or return error with guidance
	// For now, return empty mat as placeholder - will need to check gocv API
	return undistortedPoints, fmt.Errorf("UndistortPoints not yet implemented - verify gocv.UndistortPoints availability")
}

// Close releases all resources held by the camera.
func (c *Camera) Close() {
	if !c.CameraMatrix.Empty() {
		c.CameraMatrix.Close()
	}
	if !c.DistortionCoeffs.Empty() {
		c.DistortionCoeffs.Close()
	}
	if !c.Rectification.Empty() {
		c.Rectification.Close()
	}
	if !c.Projection.Empty() {
		c.Projection.Close()
	}
}
