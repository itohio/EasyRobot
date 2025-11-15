package corners

import (
	"fmt"
	"image"

	cv "gocv.io/x/gocv"
)

// Detector handles chessboard corner detection.
type Detector struct {
	gridSize image.Point
}

// NewDetector creates a new corner detector with the specified grid size.
func NewDetector(gridSize image.Point) *Detector {
	return &Detector{
		gridSize: gridSize,
	}
}

// Detect detects chessboard corners in an image.
// Returns true if corners were found, along with the corner points.
// The corners are refined to sub-pixel accuracy.
func (d *Detector) Detect(mat cv.Mat) (bool, []cv.Point2f, error) {
	if mat.Empty() {
		return false, nil, fmt.Errorf("empty mat")
	}

	// Convert to grayscale if needed
	gray := cv.NewMat()
	defer gray.Close()

	if mat.Channels() == 1 {
		gray = mat.Clone()
	} else {
		cv.CvtColor(mat, &gray, cv.ColorBGRToGray)
	}

	// Find chessboard corners
	patternSize := image.Point{X: d.gridSize.X, Y: d.gridSize.Y}
	cornersMat := cv.NewMat()
	defer cornersMat.Close()

	found := cv.FindChessboardCornersSB(
		gray,
		patternSize,
		&cornersMat,
		cv.CalibCBAdaptiveThresh|cv.CalibCBNormalizeImage,
	)

	if !found || cornersMat.Empty() {
		return false, nil, nil
	}

	// Refine corner positions (CornersSubPix modifies the Mat in-place)
	winSize := image.Point{X: 11, Y: 11}
	zeroZone := image.Point{X: -1, Y: -1}
	criteria := cv.NewTermCriteria(cv.Count+cv.EPS, 30, 0.001)

	if err := cv.CornerSubPix(gray, &cornersMat, winSize, zeroZone, criteria); err != nil {
		return false, nil, fmt.Errorf("corner refinement failed: %w", err)
	}

	// Convert refined corners Mat to Point2fVector and then to slice
	cornersVec := cv.NewPoint2fVectorFromMat(cornersMat)
	defer cornersVec.Close()

	if cornersVec.Size() == 0 {
		return false, nil, nil
	}

	imgPts := cornersVec.ToPoints()

	// Validate we have the correct number of corners
	expectedCorners := d.gridSize.X * d.gridSize.Y
	if len(imgPts) != expectedCorners {
		return false, nil, fmt.Errorf("found %d corners, expected %d", len(imgPts), expectedCorners)
	}

	return true, imgPts, nil
}

// Draw draws detected corners on the image.
// If corners are nil or empty, nothing is drawn.
func Draw(mat cv.Mat, gridSize image.Point, corners []cv.Point2f, found bool) error {
	if !found || len(corners) == 0 {
		return nil
	}

	// Create Mat from corners slice
	// DrawChessboardCorners expects corners as Mat in CV_32FC2 format (N x 1 x 2 channels)
	// where N is the number of corners
	cornersMat := cv.NewMatWithSize(len(corners), 1, cv.MatTypeCV32FC2)
	defer cornersMat.Close()

	// Fill the Mat with corner points
	// For CV_32FC2, each element has 2 channels (x, y)
	// We need to set channel 0 (x) and channel 1 (y) separately
	for i, pt := range corners {
		cornersMat.SetFloatAt3(i, 0, 0, pt.X) // row i, col 0, channel 0 (x)
		cornersMat.SetFloatAt3(i, 0, 1, pt.Y) // row i, col 0, channel 1 (y)
	}

	patternSize := image.Point{X: gridSize.X, Y: gridSize.Y}

	// DrawChessboardCorners expects corners as Mat
	if err := cv.DrawChessboardCorners(&mat, patternSize, cornersMat, found); err != nil {
		return fmt.Errorf("failed to draw corners: %w", err)
	}

	return nil
}

// GenerateObjectPoints generates 3D object points for the calibration pattern.
// Assumes square size of 1.0 (normalized units).
func GenerateObjectPoints(gridSize image.Point) []cv.Point3f {
	points := make([]cv.Point3f, 0, gridSize.X*gridSize.Y)
	for y := 0; y < gridSize.Y; y++ {
		for x := 0; x < gridSize.X; x++ {
			points = append(points, cv.Point3f{
				X: float32(x),
				Y: float32(y),
				Z: 0.0,
			})
		}
	}
	return points
}
