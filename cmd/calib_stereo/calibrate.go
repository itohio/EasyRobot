package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"os"

	"github.com/itohio/EasyRobot/cmd/calib_mono/corners"
	cv "gocv.io/x/gocv"
)

// StereoCalibrationProcessor handles the stereo calibration process
type StereoCalibrationProcessor struct {
	gridSize         image.Point
	targetSamples    int
	numSamples       int
	imageSize        image.Point
	objectPoints     [][]cv.Point3f
	leftImagePoints  [][]cv.Point2f
	rightImagePoints [][]cv.Point2f
	// Last detected corners for visualization
	leftLastCorners  []cv.Point2f
	rightLastCorners []cv.Point2f
	leftLastFound    bool
	rightLastFound   bool
}

// NewStereoCalibrationProcessor creates a new stereo calibration processor
func NewStereoCalibrationProcessor(gridSize image.Point, targetSamples int) *StereoCalibrationProcessor {
	return &StereoCalibrationProcessor{
		gridSize:         gridSize,
		targetSamples:    targetSamples,
		objectPoints:     make([][]cv.Point3f, 0),
		leftImagePoints:  make([][]cv.Point2f, 0),
		rightImagePoints: make([][]cv.Point2f, 0),
		leftLastCorners:  nil,
		rightLastCorners: nil,
		leftLastFound:    false,
		rightLastFound:   false,
	}
}

// Close releases resources held by the processor
func (scp *StereoCalibrationProcessor) Close() {
	// No resources to close (corners are slices, not Mats)
}

// ProcessStereoFrame processes a stereo frame pair to detect calibration patterns
// Returns true if patterns were found in both images and sample was collected
func (scp *StereoCalibrationProcessor) ProcessStereoFrame(leftMat, rightMat cv.Mat, leftDetector, rightDetector *corners.Detector) (bool, error) {
	if leftMat.Empty() || rightMat.Empty() {
		return false, fmt.Errorf("empty mat")
	}

	// Update image size if needed
	if scp.imageSize.X == 0 {
		leftSize := leftMat.Size()
		if len(leftSize) >= 2 {
			scp.imageSize = image.Point{X: leftSize[1], Y: leftSize[0]} // width = cols, height = rows
		}
	}

	// Detect corners in both images
	leftFound, leftCorners, err := leftDetector.Detect(leftMat)
	if err != nil {
		scp.leftLastFound = false
		scp.leftLastCorners = nil
		scp.rightLastFound = false
		scp.rightLastCorners = nil
		return false, fmt.Errorf("left detection failed: %w", err)
	}

	rightFound, rightCorners, err := rightDetector.Detect(rightMat)
	if err != nil {
		scp.leftLastFound = false
		scp.leftLastCorners = nil
		scp.rightLastFound = false
		scp.rightLastCorners = nil
		return false, fmt.Errorf("right detection failed: %w", err)
	}

	// Both must be found for a valid stereo sample
	if !leftFound || !rightFound {
		scp.leftLastFound = leftFound
		scp.leftLastCorners = leftCorners
		scp.rightLastFound = rightFound
		scp.rightLastCorners = rightCorners
		return false, nil
	}

	// Prepare object points for this pattern (same for all images)
	objPts := corners.GenerateObjectPoints(scp.gridSize)

	// Store sample
	scp.objectPoints = append(scp.objectPoints, objPts)
	scp.leftImagePoints = append(scp.leftImagePoints, leftCorners)
	scp.rightImagePoints = append(scp.rightImagePoints, rightCorners)

	// Store corners for visualization
	scp.leftLastCorners = leftCorners
	scp.rightLastCorners = rightCorners
	scp.leftLastFound = true
	scp.rightLastFound = true
	scp.numSamples++

	return true, nil
}

// DrawCorners draws detected corners on both images
func (scp *StereoCalibrationProcessor) DrawCorners(leftMat, rightMat cv.Mat, found bool) {
	if found && scp.leftLastFound && len(scp.leftLastCorners) > 0 {
		corners.Draw(leftMat, scp.gridSize, scp.leftLastCorners, true)
	}
	if found && scp.rightLastFound && len(scp.rightLastCorners) > 0 {
		corners.Draw(rightMat, scp.gridSize, scp.rightLastCorners, true)
	}
}

// Calibrate performs stereo camera calibration using collected samples
func (scp *StereoCalibrationProcessor) Calibrate() (*StereoCalibration, error) {
	if scp.numSamples < scp.targetSamples {
		return nil, fmt.Errorf("insufficient samples: %d < %d", scp.numSamples, scp.targetSamples)
	}

	if scp.imageSize.X == 0 || scp.imageSize.Y == 0 {
		return nil, fmt.Errorf("image size not set")
	}

	// Convert to Points3fVector and Points2fVector (vector of vectors)
	objectPointsVec := cv.NewPoints3fVectorFromPoints(scp.objectPoints)
	defer objectPointsVec.Close()

	leftImagePointsVec := cv.NewPoints2fVectorFromPoints(scp.leftImagePoints)
	defer leftImagePointsVec.Close()

	rightImagePointsVec := cv.NewPoints2fVectorFromPoints(scp.rightImagePoints)
	defer rightImagePointsVec.Close()

	// Prepare output matrices for stereo calibration
	leftCameraMatrix := cv.NewMat()
	defer leftCameraMatrix.Close()
	leftDistCoeffs := cv.NewMat()
	defer leftDistCoeffs.Close()
	rightCameraMatrix := cv.NewMat()
	defer rightCameraMatrix.Close()
	rightDistCoeffs := cv.NewMat()
	defer rightDistCoeffs.Close()
	R := cv.NewMat()
	defer R.Close()
	T := cv.NewMat()
	defer T.Close()
	E := cv.NewMat()
	defer E.Close()
	F := cv.NewMat()
	defer F.Close()

	// Calibrate stereo cameras
	// Note: GoCV may not have StereoCalibrate exposed directly
	// TODO: Need to implement stereo calibration
	// This could involve:
	// 1. Calibrating each camera individually using CalibrateCamera
	// 2. Computing stereo extrinsics (rotation/translation between cameras)
	// 3. Computing Essential and Fundamental matrices
	// For now, return an error - this needs proper implementation
	return nil, fmt.Errorf("StereoCalibrate not yet fully implemented - GoCV bindings may not expose stereoCalibrate function. Implementation requires: 1) Individual camera calibration, 2) Stereo extrinsic computation (rotation/translation between cameras), 3) Essential/Fundamental matrix computation")

	// TODO: The code below needs to be implemented once stereo calibration function is found:
	// Once StereoCalibrate is available or we implement it using individual calibration:
	// 1. Call StereoCalibrate (or equivalent) to get R, T, E, F
	// 2. Call StereoRectify with R, T to get R1, R2, P1, P2, Q
	// 3. Initialize rectification maps
	// 4. Return calibration with all matrices
	_ = leftCameraMatrix
	_ = leftDistCoeffs
	_ = rightCameraMatrix
	_ = rightDistCoeffs
	_ = R
	_ = T
	_ = E
	_ = F

	return nil, nil // This will never execute due to return above, but prevents compilation errors
}

// StereoCalibration holds stereo camera calibration parameters
type StereoCalibration struct {
	// Left camera intrinsic parameters
	LeftCameraMatrix     cv.Mat // 3x3 camera matrix (K1)
	LeftDistortionCoeffs cv.Mat // Distortion coefficients

	// Right camera intrinsic parameters
	RightCameraMatrix     cv.Mat // 3x3 camera matrix (K2)
	RightDistortionCoeffs cv.Mat // Distortion coefficients

	// Stereo extrinsics
	Rotation    cv.Mat // 3x3 rotation matrix (R)
	Translation cv.Mat // 3x1 translation vector (T)
	Essential   cv.Mat // 3x3 essential matrix (E)
	Fundamental cv.Mat // 3x3 fundamental matrix (F)

	// Rectification
	LeftRectification   cv.Mat // 3x3 rectification matrix (R1)
	RightRectification  cv.Mat // 3x3 rectification matrix (R2)
	LeftProjection      cv.Mat // 3x4 projection matrix (P1)
	RightProjection     cv.Mat // 3x4 projection matrix (P2)
	DisparityToDepthMap cv.Mat // 4x4 Q matrix

	// Rectification maps (for efficient remapping) - initialized by InitRectifyMaps
	LeftMap1  cv.Mat // Map1 for remap
	LeftMap2  cv.Mat // Map2 for remap
	RightMap1 cv.Mat // Map1 for remap
	RightMap2 cv.Mat // Map2 for remap

	// Image size used for calibration
	ImageSize image.Point // Width x Height

	// Calibration metadata
	NumSamples        int         // Number of samples used
	GridShape         image.Point // Calibration pattern grid (width x height)
	ReprojectionError float64     // RMS reprojection error
}

// Close releases resources held by calibration
func (sc *StereoCalibration) Close() {
	sc.LeftCameraMatrix.Close()
	sc.LeftDistortionCoeffs.Close()
	sc.RightCameraMatrix.Close()
	sc.RightDistortionCoeffs.Close()
	sc.Rotation.Close()
	sc.Translation.Close()
	sc.Essential.Close()
	sc.Fundamental.Close()
	sc.LeftRectification.Close()
	sc.RightRectification.Close()
	sc.LeftProjection.Close()
	sc.RightProjection.Close()
	sc.DisparityToDepthMap.Close()
	sc.LeftMap1.Close()
	sc.LeftMap2.Close()
	sc.RightMap1.Close()
	sc.RightMap2.Close()
}

// InitRectifyMaps initializes the rectification maps for efficient remapping
func (sc *StereoCalibration) InitRectifyMaps() error {
	// Initialize maps if empty
	if sc.LeftMap1.Empty() {
		sc.LeftMap1 = cv.NewMat()
	}
	if sc.LeftMap2.Empty() {
		sc.LeftMap2 = cv.NewMat()
	}
	if sc.RightMap1.Empty() {
		sc.RightMap1 = cv.NewMat()
	}
	if sc.RightMap2.Empty() {
		sc.RightMap2 = cv.NewMat()
	}

	// InitUndistortRectifyMap for left camera
	// Signature: cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2
	// Parameters are passed by value except map1 and map2 which are passed by pointer
	cv.InitUndistortRectifyMap(
		sc.LeftCameraMatrix,
		sc.LeftDistortionCoeffs,
		sc.LeftRectification,
		sc.LeftProjection,
		sc.ImageSize,
		int(cv.MatTypeCV16SC2),
		sc.LeftMap1,
		sc.LeftMap2,
	)

	// InitUndistortRectifyMap for right camera
	cv.InitUndistortRectifyMap(
		sc.RightCameraMatrix,
		sc.RightDistortionCoeffs,
		sc.RightRectification,
		sc.RightProjection,
		sc.ImageSize,
		int(cv.MatTypeCV16SC2),
		sc.RightMap1,
		sc.RightMap2,
	)

	return nil
}

// Rectify rectifies a stereo image pair using the calibration
func (sc *StereoCalibration) Rectify(left, right cv.Mat) (cv.Mat, cv.Mat, error) {
	if sc.LeftMap1.Empty() || sc.RightMap1.Empty() {
		return cv.Mat{}, cv.Mat{}, fmt.Errorf("rectification maps not initialized")
	}

	leftRect := cv.NewMat()
	rightRect := cv.NewMat()

	// Remap left image
	// Remap signature: src, dst, map1, map2, interpolation, borderMode, borderValue
	borderValue := color.RGBA{0, 0, 0, 0}
	cv.Remap(left, &leftRect, &sc.LeftMap1, &sc.LeftMap2, cv.InterpolationLinear, cv.BorderConstant, borderValue)

	// Remap right image
	cv.Remap(right, &rightRect, &sc.RightMap1, &sc.RightMap2, cv.InterpolationLinear, cv.BorderConstant, borderValue)

	return leftRect, rightRect, nil
}

// saveStereoCalibration saves stereo calibration to a file
func saveStereoCalibration(cal *StereoCalibration, path, format string) error {
	switch format {
	case "json", "yaml":
		return saveStereoCalibrationJSON(cal, path, format == "yaml")
	case "gocv":
		return saveStereoCalibrationGoCV(cal, path)
	default:
		return fmt.Errorf("unsupported format: %s (use json, yaml, or gocv)", format)
	}
}

// StereoCalibrationJSON is the JSON representation of stereo calibration
type StereoCalibrationJSON struct {
	LeftCameraMatrix      [][]float64 `json:"left_camera_matrix"`
	LeftDistortionCoeffs  []float64   `json:"left_distortion_coefficients"`
	RightCameraMatrix     [][]float64 `json:"right_camera_matrix"`
	RightDistortionCoeffs []float64   `json:"right_distortion_coefficients"`
	Rotation              [][]float64 `json:"rotation"`
	Translation           []float64   `json:"translation"`
	Essential             [][]float64 `json:"essential"`
	Fundamental           [][]float64 `json:"fundamental"`
	LeftRectification     [][]float64 `json:"left_rectification"`
	RightRectification    [][]float64 `json:"right_rectification"`
	LeftProjection        [][]float64 `json:"left_projection"`
	RightProjection       [][]float64 `json:"right_projection"`
	DisparityToDepthMap   [][]float64 `json:"disparity_to_depth_map"`
	ImageSize             [2]int      `json:"image_size"`
	NumSamples            int         `json:"num_samples"`
	GridShape             [2]int      `json:"grid_shape"`
	ReprojectionError     float64     `json:"reprojection_error"`
}

func saveStereoCalibrationJSON(cal *StereoCalibration, path string, yaml bool) error {
	data := StereoCalibrationJSON{
		LeftCameraMatrix:      matToSlice2D(cal.LeftCameraMatrix),
		LeftDistortionCoeffs:  matToSlice1D(cal.LeftDistortionCoeffs),
		RightCameraMatrix:     matToSlice2D(cal.RightCameraMatrix),
		RightDistortionCoeffs: matToSlice1D(cal.RightDistortionCoeffs),
		Rotation:              matToSlice2D(cal.Rotation),
		Translation:           matToSlice1D(cal.Translation),
		Essential:             matToSlice2D(cal.Essential),
		Fundamental:           matToSlice2D(cal.Fundamental),
		LeftRectification:     matToSlice2D(cal.LeftRectification),
		RightRectification:    matToSlice2D(cal.RightRectification),
		LeftProjection:        matToSlice2D(cal.LeftProjection),
		RightProjection:       matToSlice2D(cal.RightProjection),
		DisparityToDepthMap:   matToSlice2D(cal.DisparityToDepthMap),
		ImageSize:             [2]int{cal.ImageSize.X, cal.ImageSize.Y},
		NumSamples:            cal.NumSamples,
		GridShape:             [2]int{cal.GridShape.X, cal.GridShape.Y},
		ReprojectionError:     cal.ReprojectionError,
	}

	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal calibration: %w", err)
	}

	return os.WriteFile(path, bytes, 0644)
}

func saveStereoCalibrationGoCV(cal *StereoCalibration, path string) error {
	// For GoCV native format, we could use gob encoding
	// For now, fall back to JSON
	return saveStereoCalibrationJSON(cal, path, false)
}

// loadStereoCalibration loads stereo calibration from a file
func loadStereoCalibration(path string) (*StereoCalibration, error) {
	// Try JSON first
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read calibration file: %w", err)
	}

	var jsonData StereoCalibrationJSON
	if err := json.Unmarshal(data, &jsonData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal calibration: %w", err)
	}

	cal := &StereoCalibration{
		LeftCameraMatrix:      slice2DToMat(jsonData.LeftCameraMatrix),
		LeftDistortionCoeffs:  slice1DToMat(jsonData.LeftDistortionCoeffs),
		RightCameraMatrix:     slice2DToMat(jsonData.RightCameraMatrix),
		RightDistortionCoeffs: slice1DToMat(jsonData.RightDistortionCoeffs),
		Rotation:              slice2DToMat(jsonData.Rotation),
		Translation:           slice1DToMat(jsonData.Translation),
		Essential:             slice2DToMat(jsonData.Essential),
		Fundamental:           slice2DToMat(jsonData.Fundamental),
		LeftRectification:     slice2DToMat(jsonData.LeftRectification),
		RightRectification:    slice2DToMat(jsonData.RightRectification),
		LeftProjection:        slice2DToMat(jsonData.LeftProjection),
		RightProjection:       slice2DToMat(jsonData.RightProjection),
		DisparityToDepthMap:   slice2DToMat(jsonData.DisparityToDepthMap),
		ImageSize:             image.Point{X: jsonData.ImageSize[0], Y: jsonData.ImageSize[1]},
		NumSamples:            jsonData.NumSamples,
		GridShape:             image.Point{X: jsonData.GridShape[0], Y: jsonData.GridShape[1]},
		ReprojectionError:     jsonData.ReprojectionError,
	}

	// Initialize rectification maps
	if err := cal.InitRectifyMaps(); err != nil {
		cal.Close()
		return nil, fmt.Errorf("failed to init rectify maps: %w", err)
	}

	return cal, nil
}

// Helper functions for Mat <-> slice conversion
func matToSlice2D(mat cv.Mat) [][]float64 {
	rows := mat.Rows()
	cols := mat.Cols()
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = mat.GetDoubleAt(i, j)
		}
	}
	return result
}

func matToSlice1D(mat cv.Mat) []float64 {
	rows := mat.Rows()
	result := make([]float64, rows)
	for i := 0; i < rows; i++ {
		if mat.Cols() > 0 {
			result[i] = mat.GetDoubleAt(i, 0)
		}
	}
	return result
}

func slice2DToMat(data [][]float64) cv.Mat {
	if len(data) == 0 {
		return cv.NewMat()
	}
	rows := len(data)
	cols := len(data[0])
	mat := cv.NewMatWithSize(rows, cols, cv.MatTypeCV64F)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols && j < len(data[i]); j++ {
			mat.SetDoubleAt(i, j, data[i][j])
		}
	}
	return mat
}

func slice1DToMat(data []float64) cv.Mat {
	rows := len(data)
	mat := cv.NewMatWithSize(rows, 1, cv.MatTypeCV64F)
	for i := 0; i < rows; i++ {
		mat.SetDoubleAt(i, 0, data[i])
	}
	return mat
}
