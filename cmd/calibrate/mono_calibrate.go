package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"os"

	"github.com/itohio/EasyRobot/cmd/calibrate/corners"
	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	cv "gocv.io/x/gocv"
)

// MonoCalibrationProcessor handles the monocular calibration process
type MonoCalibrationProcessor struct {
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

// NumSamples returns the number of collected samples.
func (cp *MonoCalibrationProcessor) NumSamples() int {
	return cp.numSamples
}

// TargetSamples returns the target number of samples.
func (cp *MonoCalibrationProcessor) TargetSamples() int {
	return cp.targetSamples
}

// NewMonoCalibrationProcessor creates a new monocular calibration processor
func NewMonoCalibrationProcessor(gridSize image.Point, targetSamples int) *MonoCalibrationProcessor {
	return &MonoCalibrationProcessor{
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
func (cp *MonoCalibrationProcessor) Close() {
	// No resources to close (corners are slices, not Mats)
}

// ProcessFrame processes a frame to detect calibration pattern
// Returns true if pattern was found and sample was collected
func (cp *MonoCalibrationProcessor) ProcessFrame(mat cv.Mat) (bool, error) {
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
func (cp *MonoCalibrationProcessor) DrawCorners(mat cv.Mat, found bool) {
	if found && cp.lastFound && len(cp.lastCorners) > 0 {
		corners.Draw(mat, cp.gridSize, cp.lastCorners, true)
	}
}

// Calibrate performs camera calibration using collected samples
func (cp *MonoCalibrationProcessor) Calibrate() (*MonoCameraCalibration, error) {
	if cp.numSamples < cp.targetSamples {
		return nil, fmt.Errorf("insufficient samples: %d < %d", cp.numSamples, cp.targetSamples)
	}

	if cp.imageSize.X == 0 || cp.imageSize.Y == 0 {
		return nil, fmt.Errorf("image size not set")
	}

	// Convert to Points3fVector and Points2fVector (vector of vectors)
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

	return &MonoCameraCalibration{
		CameraMatrix:      cameraMatrixCloned,
		DistortionCoeffs:  distCoeffsCloned,
		ImageSize:         cp.imageSize,
		NumSamples:        cp.numSamples,
		GridShape:         cp.gridSize,
		ReprojectionError: rms,
	}, nil
}

// MonoCameraCalibration holds camera calibration parameters
type MonoCameraCalibration struct {
	CameraMatrix      cv.Mat      // 3x3 camera matrix (K)
	DistortionCoeffs  cv.Mat      // Distortion coefficients (k1, k2, p1, p2, k3, ...)
	ImageSize         image.Point // Width x Height
	NumSamples        int         // Number of samples used
	GridShape         image.Point // Calibration pattern grid (width x height)
	ReprojectionError float64     // RMS reprojection error
}

// Close releases resources held by calibration
func (c *MonoCameraCalibration) Close() {
	c.CameraMatrix.Close()
	c.DistortionCoeffs.Close()
}

func calibrateMonoCamera(ctx context.Context, src source.Source, processor *MonoCalibrationProcessor, dests []destination.Destination, outputPath, format string) error {
	fmt.Printf("Starting camera calibration...\n")
	fmt.Printf("Grid size: %dx%d\n", processor.gridSize.X, processor.gridSize.Y)
	fmt.Printf("Target samples: %d\n", processor.targetSamples)

	if err := processCalibrationLoop(ctx, src, processor, dests); err != nil {
		return err
	}

	calibration, err := processor.Calibrate()
	if err != nil {
		return fmt.Errorf("calibration failed: %w", err)
	}
	defer calibration.Close()

	if err := saveMonoCalibration(calibration, outputPath, format); err != nil {
		return fmt.Errorf("failed to save calibration: %w", err)
	}

	fmt.Printf("Calibration saved to %s\n", outputPath)
	fmt.Printf("Reprojection error: %.4f\n", calibration.ReprojectionError)

	return nil
}

func testMonoCalibration(ctx context.Context, src source.Source, dests []destination.Destination, calibPath string) error {
	// Load calibration file
	calibration, err := loadMonoCalibration(calibPath)
	if err != nil {
		return fmt.Errorf("failed to load calibration: %w", err)
	}
	defer calibration.Close()

	fmt.Printf("Loaded calibration from %s\n", calibPath)
	fmt.Printf("Grid size: %dx%d\n", calibration.GridShape.X, calibration.GridShape.Y)
	fmt.Printf("Image size: %dx%d\n", calibration.ImageSize.X, calibration.ImageSize.Y)
	fmt.Printf("Samples used: %d\n", calibration.NumSamples)
	fmt.Printf("Reprojection error: %.4f\n", calibration.ReprojectionError)

	// Preallocate mats for reuse
	emptyMat := cv.NewMat()
	defer emptyMat.Close()

	// Process frames and undistort
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Read frame from source
		frame, err := src.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				return nil
			}
			fmt.Fprintf(os.Stderr, "Warning: failed to read frame: %v\n", err)
			continue
		}

		if len(frame.Tensors) == 0 {
			continue
		}

		// Convert tensor to Mat
		mat, err := tensorToMat(frame.Tensors[0])
		if err != nil {
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			continue
		}

		// Undistort
		undistorted := cv.NewMat()
		if err := cv.Undistort(mat, &undistorted, calibration.CameraMatrix, calibration.DistortionCoeffs, emptyMat); err != nil {
			mat.Close()
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			continue
		}

		mat.Close()

		// Convert back to tensor
		undistortedTensor, err := matToTensor(undistorted)
		if err != nil {
			undistorted.Close()
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			continue
		}

		// Release original tensor
		if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
			releaser.Release()
		}

		// Use SharedTensor for multiple destinations
		if len(dests) > 1 {
			sharedTensor := tensor.WithRefcount(undistortedTensor)

			// Create frame with shared tensor for first destination
			sharedFrame := types.Frame{
				Tensors:  []types.Tensor{sharedTensor},
				Metadata: frame.Metadata,
			}

			// Create view frames for remaining destinations
			numViews := len(dests) - 1
			viewFrames := make([]types.Frame, numViews)
			for i := range numViews {
				viewFrames[i] = types.Frame{
					Tensors:  []types.Tensor{sharedTensor.View()},
					Metadata: frame.Metadata,
				}
			}

			// Send to first destination
			if err := dests[0].AddFrame(sharedFrame); err != nil {
				_ = err // Ignore destination errors
			}

			// Send views to remaining destinations
			for i := 1; i < len(dests); i++ {
				if err := dests[i].AddFrame(viewFrames[i-1]); err != nil {
					_ = err // Ignore destination errors
				}
			}
		} else if len(dests) == 1 {
			// Single destination
			displayFrame := types.Frame{
				Tensors:  []types.Tensor{undistortedTensor},
				Metadata: frame.Metadata,
			}

			if err := dests[0].AddFrame(displayFrame); err != nil {
				_ = err // Ignore destination errors
			}
		}
	}
}

// saveMonoCalibration saves calibration to a file
func saveMonoCalibration(cal *MonoCameraCalibration, path, format string) error {
	switch format {
	case "json", "yaml":
		return saveMonoCalibrationJSON(cal, path, format == "yaml")
	case "gocv":
		return saveMonoCalibrationGoCV(cal, path)
	default:
		return fmt.Errorf("unsupported format: %s (use json, yaml, or gocv)", format)
	}
}

// MonoCameraCalibrationJSON is the JSON representation of calibration
type MonoCameraCalibrationJSON struct {
	CameraMatrix      [][]float64 `json:"camera_matrix"`
	DistortionCoeffs  []float64   `json:"distortion_coefficients"`
	ImageSize         [2]int      `json:"image_size"`
	NumSamples        int         `json:"num_samples"`
	GridShape         [2]int      `json:"grid_shape"`
	ReprojectionError float64     `json:"reprojection_error"`
}

func saveMonoCalibrationJSON(cal *MonoCameraCalibration, path string, yaml bool) error {
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

	data := MonoCameraCalibrationJSON{
		CameraMatrix:      camMat,
		DistortionCoeffs:  distCoeffs,
		ImageSize:         [2]int{cal.ImageSize.X, cal.ImageSize.Y},
		NumSamples:        cal.NumSamples,
		GridShape:         [2]int{cal.GridShape.X, cal.GridShape.Y},
		ReprojectionError: cal.ReprojectionError,
	}

	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal calibration: %w", err)
	}

	return os.WriteFile(path, bytes, 0644)
}

func saveMonoCalibrationGoCV(cal *MonoCameraCalibration, path string) error {
	// For GoCV native format, we could use gob encoding
	// For now, fall back to JSON
	return saveMonoCalibrationJSON(cal, path, false)
}

// loadMonoCalibration loads calibration from a file
func loadMonoCalibration(path string) (*MonoCameraCalibration, error) {
	// Try JSON first
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read calibration file: %w", err)
	}

	var jsonData MonoCameraCalibrationJSON
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

	return &MonoCameraCalibration{
		CameraMatrix:      cameraMatrix,
		DistortionCoeffs:  distCoeffs,
		ImageSize:         image.Point{X: jsonData.ImageSize[0], Y: jsonData.ImageSize[1]},
		NumSamples:        jsonData.NumSamples,
		GridShape:         image.Point{X: jsonData.GridShape[0], Y: jsonData.GridShape[1]},
		ReprojectionError: jsonData.ReprojectionError,
	}, nil
}

