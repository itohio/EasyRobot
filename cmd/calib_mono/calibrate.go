package main

import (
	"encoding/json"
	"fmt"
	"image"
	"os"

	"github.com/itohio/EasyRobot/cmd/calib_mono/corners"
	cv "gocv.io/x/gocv"
)

// CalibrationProcessor handles the calibration process
type CalibrationProcessor struct {
	gridSize      image.Point
	targetSamples int
	numSamples    int
	imageSize     image.Point
	objectPoints  [][]cv.Point3f
	imagePoints   [][]cv.Point2f
	detector      *corners.Detector
	lastCorners   []cv.Point2f // Store last detected corners for visualization
	lastFound     bool         // Whether last frame had detected corners
}

// NewCalibrationProcessor creates a new calibration processor
func NewCalibrationProcessor(gridSize image.Point, targetSamples int) *CalibrationProcessor {
	return &CalibrationProcessor{
		gridSize:      gridSize,
		targetSamples: targetSamples,
		objectPoints:  make([][]cv.Point3f, 0),
		imagePoints:   make([][]cv.Point2f, 0),
		detector:      corners.NewDetector(gridSize),
		lastCorners:   nil,
		lastFound:     false,
	}
}

// Close releases resources held by the processor
func (cp *CalibrationProcessor) Close() {
	// No resources to close (corners are slices, not Mats)
}

// ProcessFrame processes a frame to detect calibration pattern
// Returns true if pattern was found and sample was collected
func (cp *CalibrationProcessor) ProcessFrame(mat cv.Mat) (bool, error) {
	if mat.Empty() {
		return false, fmt.Errorf("empty mat")
	}

	// Update image size if needed
	// Mat.Size() returns [rows, cols] or [height, width]
	if cp.imageSize.X == 0 {
		size := mat.Size()
		if len(size) >= 2 {
			cp.imageSize = image.Point{X: size[1], Y: size[0]} // width = cols, height = rows
		}
	}

	// Detect corners using shared detector
	found, imgPts, err := cp.detector.Detect(mat)
	if err != nil {
		cp.lastFound = false
		cp.lastCorners = nil
		return false, err
	}

	if !found {
		cp.lastFound = false
		cp.lastCorners = nil
		return false, nil
	}

	// Prepare object points for this pattern (same for all images)
	objPts := corners.GenerateObjectPoints(cp.gridSize)

	// Store sample
	cp.objectPoints = append(cp.objectPoints, objPts)
	cp.imagePoints = append(cp.imagePoints, imgPts)

	// Store corners for visualization
	cp.lastCorners = imgPts
	cp.lastFound = true
	cp.numSamples++

	return true, nil
}

// DrawCorners draws detected corners on the image
func (cp *CalibrationProcessor) DrawCorners(mat cv.Mat, found bool) {
	if found && cp.lastFound && len(cp.lastCorners) > 0 {
		corners.Draw(mat, cp.gridSize, cp.lastCorners, true)
	}
}

// Calibrate performs camera calibration using collected samples
func (cp *CalibrationProcessor) Calibrate() (*CameraCalibration, error) {
	if cp.numSamples < cp.targetSamples {
		return nil, fmt.Errorf("insufficient samples: %d < %d", cp.numSamples, cp.targetSamples)
	}

	if cp.imageSize.X == 0 || cp.imageSize.Y == 0 {
		return nil, fmt.Errorf("image size not set")
	}

	// Convert to Points3fVector and Points2fVector (vector of vectors)
	// Use NewPoints3fVectorFromPoints which accepts [][]Point3f directly
	objectPointsVec := cv.NewPoints3fVectorFromPoints(cp.objectPoints)
	defer objectPointsVec.Close()

	imagePointsVec := cv.NewPoints2fVectorFromPoints(cp.imagePoints)
	defer imagePointsVec.Close()

	// Prepare output matrices
	cameraMatrix := cv.NewMat()
	defer cameraMatrix.Close()
	distCoeffs := cv.NewMat()
	defer distCoeffs.Close()
	rvecs := cv.NewMat()
	defer rvecs.Close()
	tvecs := cv.NewMat()
	defer tvecs.Close()

	// Calibrate camera
	// CalibrateCamera signature: objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags
	// Use 0 for flags to use default behavior (no special flags)
	rms := cv.CalibrateCamera(
		objectPointsVec,
		imagePointsVec,
		cp.imageSize,
		&cameraMatrix,
		&distCoeffs,
		&rvecs,
		&tvecs,
		0, // Default calibration flags
	)

	// Clone matrices for storage (calibration owns them)
	cameraMatrixCloned := cameraMatrix.Clone()
	distCoeffsCloned := distCoeffs.Clone()

	return &CameraCalibration{
		CameraMatrix:      cameraMatrixCloned,
		DistortionCoeffs:  distCoeffsCloned,
		ImageSize:         cp.imageSize,
		NumSamples:        cp.numSamples,
		GridShape:         cp.gridSize,
		ReprojectionError: rms,
	}, nil
}

// CameraCalibration holds camera calibration parameters
type CameraCalibration struct {
	CameraMatrix      cv.Mat      // 3x3 camera matrix (K)
	DistortionCoeffs  cv.Mat      // Distortion coefficients (k1, k2, p1, p2, k3, ...)
	ImageSize         image.Point // Width x Height
	NumSamples        int         // Number of samples used
	GridShape         image.Point // Calibration pattern grid (width x height)
	ReprojectionError float64     // RMS reprojection error
}

// Close releases resources held by calibration
func (c *CameraCalibration) Close() {
	c.CameraMatrix.Close()
	c.DistortionCoeffs.Close()
}

// saveCalibration saves calibration to a file
func saveCalibration(cal *CameraCalibration, path, format string) error {
	switch format {
	case "json", "yaml":
		return saveCalibrationJSON(cal, path, format == "yaml")
	case "gocv":
		return saveCalibrationGoCV(cal, path)
	default:
		return fmt.Errorf("unsupported format: %s (use json, yaml, or gocv)", format)
	}
}

// CameraCalibrationJSON is the JSON representation of calibration
type CameraCalibrationJSON struct {
	CameraMatrix      [][]float64 `json:"camera_matrix"`
	DistortionCoeffs  []float64   `json:"distortion_coefficients"`
	ImageSize         [2]int      `json:"image_size"`
	NumSamples        int         `json:"num_samples"`
	GridShape         [2]int      `json:"grid_shape"`
	ReprojectionError float64     `json:"reprojection_error"`
}

func saveCalibrationJSON(cal *CameraCalibration, path string, yaml bool) error {
	// Convert camera matrix to 2D slice
	rows := cal.CameraMatrix.Rows()
	cols := cal.CameraMatrix.Cols()
	camMat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		camMat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			camMat[i][j] = cal.CameraMatrix.GetDoubleAt(i, j)
		}
	}

	// Convert distortion coefficients to slice
	distRows := cal.DistortionCoeffs.Rows()
	distCoeffs := make([]float64, distRows)
	for i := 0; i < distRows; i++ {
		distCoeffs[i] = cal.DistortionCoeffs.GetDoubleAt(i, 0)
	}

	data := CameraCalibrationJSON{
		CameraMatrix:      camMat,
		DistortionCoeffs:  distCoeffs,
		ImageSize:         [2]int{cal.ImageSize.X, cal.ImageSize.Y},
		NumSamples:        cal.NumSamples,
		GridShape:         [2]int{cal.GridShape.X, cal.GridShape.Y},
		ReprojectionError: cal.ReprojectionError,
	}

	var bytes []byte
	var err error
	if yaml {
		// For YAML, we'd need to import yaml package
		// For now, use JSON indented
		bytes, err = json.MarshalIndent(data, "", "  ")
	} else {
		bytes, err = json.MarshalIndent(data, "", "  ")
	}
	if err != nil {
		return fmt.Errorf("failed to marshal calibration: %w", err)
	}

	return os.WriteFile(path, bytes, 0644)
}

func saveCalibrationGoCV(cal *CameraCalibration, path string) error {
	// For GoCV native format, we could use gob encoding
	// For now, fall back to JSON
	return saveCalibrationJSON(cal, path, false)
}

// loadCalibration loads calibration from a file
func loadCalibration(path string) (*CameraCalibration, error) {
	// Try JSON first
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read calibration file: %w", err)
	}

	var jsonData CameraCalibrationJSON
	if err := json.Unmarshal(data, &jsonData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal calibration: %w", err)
	}

	// Convert camera matrix
	cameraMatrix := cv.NewMatWithSize(3, 3, cv.MatTypeCV64F)
	for i := 0; i < len(jsonData.CameraMatrix); i++ {
		for j := 0; j < len(jsonData.CameraMatrix[i]); j++ {
			cameraMatrix.SetDoubleAt(i, j, jsonData.CameraMatrix[i][j])
		}
	}

	// Convert distortion coefficients
	distRows := len(jsonData.DistortionCoeffs)
	distCoeffs := cv.NewMatWithSize(distRows, 1, cv.MatTypeCV64F)
	for i := 0; i < distRows; i++ {
		distCoeffs.SetDoubleAt(i, 0, jsonData.DistortionCoeffs[i])
	}

	return &CameraCalibration{
		CameraMatrix:      cameraMatrix,
		DistortionCoeffs:  distCoeffs,
		ImageSize:         image.Point{X: jsonData.ImageSize[0], Y: jsonData.ImageSize[1]},
		NumSamples:        jsonData.NumSamples,
		GridShape:         image.Point{X: jsonData.GridShape[0], Y: jsonData.GridShape[1]},
		ReprojectionError: jsonData.ReprojectionError,
	}, nil
}
