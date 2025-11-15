# Monocular Camera Calibration - Design Document

## Overview

This document describes the design for a monocular camera calibration utility that:
- Reads video/images/camera input using shared input components
- Displays frames on screen using shared display components
- Detects calibration patterns (chessboard corners)
- Collects sufficient calibration samples (configurable)
- Outputs camera calibration matrices and parameters
- Supports multiple output formats (JSON, YAML, GoCV native)

## Goals

1. Provide monocular camera calibration using rectangular checkerboard patterns
2. Reuse input/output code from `cmd/shared`
3. Use GoCV for pattern detection and calibration calculations
4. Support configurable calibration grid size
5. Support configurable number of samples
6. Display calibration progress and detected patterns
7. Output calibration results in multiple formats

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.go                               │
│  - Parse flags                                               │
│  - Validate input options                                    │
│  - Initialize calibration parameters                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Shared Input Source                              │
│  - Use cmd/shared.NewFrameStream                             │
│  - Supports: images, video, camera                           │
│  - Returns types.FrameStream                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Calibration Processor                            │
│  - Detect chessboard corners                                 │
│  - Collect calibration samples                               │
│  - Track calibration progress                                │
│  - Visualize detected patterns                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Shared Display Writer                            │
│  - Use cmd/shared.NewDisplayWriter                           │
│  - Show frames with detected patterns                        │
│  - Display calibration progress                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Calibration Result Output                        │
│  - Compute calibration using GoCV                            │
│  - Marshal to JSON/YAML/GoCV                                 │
│  - Write to output file                                      │
└─────────────────────────────────────────────────────────────┘
```

## Command-Line Interface

```bash
calib_mono [OPTIONS] --camera <device-id>
calib_mono [OPTIONS] --video <video-file>
calib_mono [OPTIONS] --images <image-path>
```

### Flags

**Input (mutually exclusive, one required):**
- `--camera <id>`: Camera device ID (can specify once for mono)
- `--video <path>`: Video file path
- `--images <path>`: Image file or directory path

**Calibration Parameters:**
- `--grid <width>,<height>`: Calibration grid shape (default: "9,7")
  - Example: `--grid 9,7` means 9x7 internal corners
- `--samples <N>`: Number of calibration samples to collect (default: 30)
- `--size <width>,<height>`: Frame resolution for cameras (default: "640,480")
- `--fps <rate>`: Frame rate for cameras (default: 5)

**Output:**
- `--output <file>`: Output file path (default: "camera.json")
- `--format <format>`: Output format: json, yaml, gocv (default: json)
- `--test`: Test mode - load existing calibration file and display results

**Display:**
- `--hide`: Hide display window
- `--title <title>`: Display window title (default: "Camera Calibration")

## Calibration Process

### Pattern Detection

1. **Convert Frame to Mat**: Extract first tensor from frame and convert to `gocv.Mat`
2. **Convert to Grayscale**: Convert color image to grayscale if needed
3. **Find Chessboard Corners**: Use `gocv.FindChessboardCornersSB` or `gocv.FindChessboardCorners`
4. **Refine Corners**: Use `gocv.CornerSubPix` to refine corner positions
5. **Visualize**: Draw detected corners on image for display

### Sample Collection

1. **Track Samples**: Maintain list of valid calibration samples
2. **Pattern Validation**: Only accept frames with successfully detected patterns
3. **Sample Storage**: Store object points (3D) and image points (2D) for each sample
4. **Progress Tracking**: Display number of collected samples vs. target

### Calibration Computation

1. **Prepare Object Points**: Generate 3D object points for each sample
2. **Collect Image Points**: Gather all 2D image points from valid samples
3. **Calibrate Camera**: Use `gocv.CalibrateCamera` to compute:
   - Camera matrix (3x3)
   - Distortion coefficients (5x1 or more)
   - Rotation vectors (optional)
   - Translation vectors (optional)
4. **Compute Reprojection Error**: Validate calibration quality

### Calibration Data Structure

```go
type CameraCalibration struct {
    // Camera intrinsic parameters
    CameraMatrix      gocv.Mat   // 3x3 camera matrix (K)
    DistortionCoeffs  gocv.Mat   // Distortion coefficients (k1, k2, p1, p2, k3, ...)
    
    // Image size used for calibration
    ImageSize         image.Point // Width x Height
    
    // Calibration metadata
    NumSamples        int         // Number of samples used
    GridShape         image.Point // Calibration pattern grid (width x height)
    ReprojectionError float64     // RMS reprojection error
    
    // Optional: per-image rotation and translation vectors
    Rvecs []gocv.Mat // Rotation vectors (if saved)
    Tvecs []gocv.Mat // Translation vectors (if saved)
}
```

## Implementation Details

### Pattern Detection

```go
import (
    cv "gocv.io/x/gocv"
    "github.com/itohio/EasyRobot/x/marshaller/gocv"
    "github.com/itohio/EasyRobot/x/marshaller/types"
)

type CalibrationProcessor struct {
    gridSize        image.Point
    objectPoints    [][]gocv.Point3f
    imagePoints     [][]gocv.Point2f
    numSamples      int
    targetSamples   int
    imageSize       image.Point
}

func (cp *CalibrationProcessor) ProcessFrame(frame types.Frame) (bool, error) {
    // Convert tensor to Mat
    mat, err := gocv.TensorToMat(frame.Tensors[0])
    if err != nil {
        return false, err
    }
    defer mat.Close()
    
    // Convert to grayscale
    gray := cv.NewMat()
    defer gray.Close()
    cv.CvtColor(mat, &gray, cv.ColorBGRToGray)
    
    // Prepare object points for this pattern
    objPts := cp.generateObjectPoints()
    
    // Find corners
    corners := gocv.NewPoint2fVector()
    defer corners.Close()
    
    found := cv.FindChessboardCornersSB(
        gray,
        image.Point{cp.gridSize.X, cp.gridSize.Y},
        corners,
        cv.CalibCbAdaptiveThresh|cv.CalibCbNormalizeImage,
    )
    
    if !found {
        return false, nil
    }
    
    // Refine corners
    term := cv.NewTermCriteria(cv.Count+cv.Eps, 30, 0.001)
    cv.CornerSubPix(gray, corners, image.Point{11, 11}, image.Point{-1, -1}, term)
    
    // Store sample
    cp.objectPoints = append(cp.objectPoints, objPts)
    
    imgPts := cp.point2fVectorToSlice(corners)
    cp.imagePoints = append(cp.imagePoints, imgPts)
    
    // Update image size if needed
    if cp.imageSize.X == 0 {
        cp.imageSize = mat.Size()
    }
    
    cp.numSamples++
    return true, nil
}

func (cp *CalibrationProcessor) generateObjectPoints() []gocv.Point3f {
    // Generate 3D object points for calibration pattern
    // Assume square size of 1.0 (normalized units)
    points := make([]gocv.Point3f, 0, cp.gridSize.X*cp.gridSize.Y)
    for y := 0; y < cp.gridSize.Y; y++ {
        for x := 0; x < cp.gridSize.X; x++ {
            points = append(points, gocv.Point3f{
                X: float32(x),
                Y: float32(y),
                Z: 0.0,
            })
        }
    }
    return points
}

func (cp *CalibrationProcessor) Calibrate() (*CameraCalibration, error) {
    if cp.numSamples < cp.targetSamples {
        return nil, fmt.Errorf("insufficient samples: %d < %d", cp.numSamples, cp.targetSamples)
    }
    
    // Prepare data for calibration
    objectPoints := cp.convertObjectPoints()
    imagePoints := cp.convertImagePoints()
    
    // Calibrate camera
    cameraMatrix := cv.NewMat()
    distCoeffs := cv.NewMat()
    rvecs := cv.NewMat()
    tvecs := cv.NewMat()
    
    rms := cv.CalibrateCamera(
        objectPoints,
        imagePoints,
        cp.imageSize,
        &cameraMatrix,
        &distCoeffs,
        &rvecs,
        &tvecs,
        cv.CalibZeroTangentDist|cv.CalibFixAspectRatio,
        cv.NewTermCriteria(cv.Count+cv.Eps, 30, 0.0001),
    )
    
    return &CameraCalibration{
        CameraMatrix:      cameraMatrix,
        DistortionCoeffs:  distCoeffs,
        ImageSize:         cp.imageSize,
        NumSamples:        cp.numSamples,
        GridShape:         cp.gridSize,
        ReprojectionError: rms,
    }, nil
}
```

### Output Formatting

```go
type CameraCalibrationJSON struct {
    CameraMatrix      [][]float64 `json:"camera_matrix"`
    DistortionCoeffs  []float64   `json:"distortion_coefficients"`
    ImageSize         [2]int      `json:"image_size"`
    NumSamples        int         `json:"num_samples"`
    GridShape         [2]int      `json:"grid_shape"`
    ReprojectionError float64     `json:"reprojection_error"`
}

func (c *CameraCalibration) ToJSON() (*CameraCalibrationJSON, error) {
    // Convert Mat to JSON-serializable format
    // ...
}

func (c *CameraCalibration) SaveJSON(path string) error {
    jsonData, err := c.ToJSON()
    if err != nil {
        return err
    }
    data, err := json.MarshalIndent(jsonData, "", "  ")
    if err != nil {
        return err
    }
    return os.WriteFile(path, data, 0644)
}
```

## Test Mode

When `--test` flag is used:
1. Load existing calibration file
2. Undistort input frames using loaded calibration
3. Display undistorted frames
4. Optionally compute reprojection error for new samples

## File Structure

```
cmd/calib_mono/
├── DESIGN.md (this file)
├── main.go              # Main entry point
├── calibrate.go         # Calibration processor
├── pattern.go           # Pattern detection
├── output.go            # Output formatting
└── README.md            # Usage documentation
```

## Dependencies

- `github.com/itohio/EasyRobot/cmd/shared` - Shared input/output components
- `github.com/itohio/EasyRobot/x/marshaller/gocv` - GoCV marshallers
- `gocv.io/x/gocv` - GoCV bindings
- Standard library: `encoding/json`, `encoding/yaml`, `os`, `flag`

## Implementation Plan

### Phase 1: Basic Structure
1. Create main.go with flag parsing
2. Integrate shared input/output components
3. Set up basic frame loop

### Phase 2: Pattern Detection
1. Implement chessboard corner detection
2. Add corner refinement
3. Visualize detected patterns on display

### Phase 3: Sample Collection
1. Implement sample tracking
2. Add progress display
3. Validate sample quality

### Phase 4: Calibration
1. Implement calibration computation
2. Add reprojection error calculation
3. Validate calibration results

### Phase 5: Output
1. Implement JSON output format
2. Implement YAML output format
3. Implement GoCV native format (optional)

### Phase 6: Test Mode
1. Implement calibration file loading
2. Add undistortion display
3. Add validation features

### Phase 7: Integration and Testing
1. End-to-end testing with camera
2. Testing with video files
3. Testing with image sequences
4. Performance testing

## Open Questions

1. **Square Size**: Should we support configurable square size in real units (mm, cm)?
   - Current: Normalized units (1.0)
   - Future: Configurable square size parameter

2. **Calibration Flags**: Which OpenCV calibration flags to use by default?
   - Current: Zero tangent distortion, fix aspect ratio
   - Future: Make configurable

3. **Sample Quality**: How to handle low-quality samples?
   - Reject samples with high reprojection error?
   - Manual sample selection?

4. **Output Format**: Should GoCV native format use gob encoding?
   - JSON/YAML for human-readable
   - GoCV for programmatic use

