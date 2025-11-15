# Stereo Camera Calibration - Design Document

## Overview

This document describes the design for a stereo camera calibration utility that:
- Reads from two camera sources (left and right) using shared input components
- Displays frames from both cameras on screen using shared display components
- Detects calibration patterns simultaneously in both images
- Collects sufficient calibration samples (configurable)
- Outputs stereo calibration parameters (intrinsics, extrinsics, rectification)
- Supports multiple output formats (JSON, YAML, GoCV native)

## Goals

1. Provide stereo camera calibration using rectangular checkerboard patterns
2. Reuse input/output code from `cmd/shared`
3. Use GoCV for pattern detection and stereo calibration calculations
4. Support configurable calibration grid size
5. Support configurable number of samples
6. Display calibration progress and detected patterns for both cameras
7. Output stereo calibration results in multiple formats
8. Support disparity map calculation for testing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.go                               │
│  - Parse flags                                               │
│  - Validate stereo input options                             │
│  - Initialize stereo calibration parameters                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Shared Input Sources (Left + Right)                 │
│  - Use cmd/shared.NewFrameStream for each camera            │
│  - Supports: camera, video, images                          │
│  - Synchronize frame streams                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Stereo Calibration Processor                        │
│  - Detect chessboard corners in both images                 │
│  - Validate pattern detection for both cameras              │
│  - Collect synchronized calibration samples                 │
│  - Track calibration progress                               │
│  - Visualize detected patterns                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Shared Display Writer                               │
│  - Use cmd/shared.NewDisplayWriter                           │
│  - Show side-by-side frames with detected patterns          │
│  - Display calibration progress                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Stereo Calibration Result Output                    │
│  - Compute stereo calibration using GoCV                    │
│  - Marshal to JSON/YAML/GoCV                                │
│  - Write to output file                                     │
└─────────────────────────────────────────────────────────────┘
```

## Command-Line Interface

```bash
calib_stereo [OPTIONS] --camera <left-id> <right-id>
calib_stereo [OPTIONS] --video <left-video> <right-video>
calib_stereo [OPTIONS] --images <left-path> <right-path>
```

### Flags

**Input (mutually exclusive, one required):**
- `--camera <left> <right>`: Camera device IDs (exactly two required)
- `--video <left> <right>`: Video file paths (exactly two required)
- `--images <left> <right>`: Image file or directory paths (exactly two required)

**Calibration Parameters:**
- `--grid <width>,<height>`: Calibration grid shape (default: "9,7")
  - Example: `--grid 9,7` means 9x7 internal corners
- `--samples <N>`: Number of calibration samples to collect (default: 30)
- `--size <width>,<height>`: Frame resolution for cameras (default: "640,480")
- `--fps <rate>`: Frame rate for cameras (default: 5)

**Output:**
- `--output <file>`: Output file path (default: "stereo_camera.json")
- `--format <format>`: Output format: json, yaml, gocv (default: json)
- `--test`: Test mode - load existing calibration file and display results
- `--disparity`: Calculate and display disparity map (test mode only)

**Display:**
- `--hide`: Hide display window
- `--title <title>`: Display window title (default: "Stereo Calibration")

## Calibration Process

### Synchronized Pattern Detection

1. **Read Paired Frames**: Read synchronized frames from both sources
2. **Detect Patterns**: Find chessboard corners in both images
3. **Validate Pair**: Both images must have valid pattern detection
4. **Refine Corners**: Refine corner positions in both images
5. **Visualize**: Draw detected corners on both images side-by-side

### Sample Collection

1. **Track Samples**: Maintain list of valid stereo calibration samples
2. **Pattern Validation**: Only accept frame pairs with both patterns detected
3. **Sample Storage**: Store object points (3D) and image points (2D) for each camera
4. **Progress Tracking**: Display number of collected samples vs. target

### Stereo Calibration Computation

1. **Monocular Calibration**: Optionally calibrate each camera individually first
2. **Prepare Object Points**: Generate 3D object points for each sample
3. **Collect Image Points**: Gather 2D image points from both cameras
4. **Stereo Calibrate**: Use `gocv.StereoCalibrate` to compute:
   - Left camera matrix (3x3)
   - Right camera matrix (3x3)
   - Left distortion coefficients
   - Right distortion coefficients
   - Rotation matrix (3x3) between cameras
   - Translation vector (3x1) between cameras
   - Essential matrix (3x3)
   - Fundamental matrix (3x3)
5. **Rectification**: Compute stereo rectification using `gocv.StereoRectify`
6. **Compute Reprojection Error**: Validate calibration quality

### Calibration Data Structure

```go
type StereoCalibration struct {
    // Left camera intrinsic parameters
    LeftCameraMatrix      gocv.Mat   // 3x3 camera matrix (K1)
    LeftDistortionCoeffs  gocv.Mat   // Distortion coefficients
    
    // Right camera intrinsic parameters
    RightCameraMatrix     gocv.Mat   // 3x3 camera matrix (K2)
    RightDistortionCoeffs gocv.Mat   // Distortion coefficients
    
    // Stereo extrinsics
    Rotation              gocv.Mat   // 3x3 rotation matrix (R)
    Translation           gocv.Mat   // 3x1 translation vector (T)
    Essential             gocv.Mat   // 3x3 essential matrix (E)
    Fundamental           gocv.Mat   // 3x3 fundamental matrix (F)
    
    // Rectification
    LeftRectification     gocv.Mat   // 3x3 rectification matrix (R1)
    RightRectification    gocv.Mat   // 3x3 rectification matrix (R2)
    LeftProjection        gocv.Mat   // 3x4 projection matrix (P1)
    RightProjection       gocv.Mat   // 3x4 projection matrix (P2)
    DisparityToDepthMap   gocv.Mat   // 4x4 Q matrix
    
    // Image size used for calibration
    ImageSize             image.Point // Width x Height
    
    // Calibration metadata
    NumSamples            int         // Number of samples used
    GridShape             image.Point // Calibration pattern grid (width x height)
    ReprojectionError     float64     // RMS reprojection error
}
```

## Implementation Details

### Synchronized Frame Reading

```go
import (
    cv "gocv.io/x/gocv"
    "github.com/itohio/EasyRobot/x/marshaller/gocv"
    "github.com/itohio/EasyRobot/x/marshaller/types"
)

type StereoFrame struct {
    Left  types.Frame
    Right types.Frame
}

func readSynchronizedFrames(leftStream, rightStream types.FrameStream) (StereoFrame, bool, error) {
    var leftFrame, rightFrame types.Frame
    var leftOk, rightOk bool
    
    select {
    case leftFrame, leftOk = <-leftStream.C:
        if !leftOk {
            return StereoFrame{}, false, nil
        }
    case rightFrame, rightOk = <-rightStream.C:
        if !rightOk {
            return StereoFrame{}, false, nil
        }
    default:
        return StereoFrame{}, false, nil
    }
    
    // Try to get matching frame from other stream
    // This is simplified - actual implementation should handle synchronization better
    // ...
    
    return StereoFrame{Left: leftFrame, Right: rightFrame}, true, nil
}
```

### Stereo Calibration Processor

```go
type StereoCalibrationProcessor struct {
    gridSize        image.Point
    objectPoints    [][]gocv.Point3f
    leftImagePoints [][]gocv.Point2f
    rightImagePoints [][]gocv.Point2f
    numSamples      int
    targetSamples   int
    imageSize       image.Point
}

func (scp *StereoCalibrationProcessor) ProcessStereoFrame(frame StereoFrame) (bool, error) {
    // Detect patterns in both images
    leftMat, err := gocv.TensorToMat(frame.Left.Tensors[0])
    if err != nil {
        return false, err
    }
    defer leftMat.Close()
    
    rightMat, err := gocv.TensorToMat(frame.Right.Tensors[0])
    if err != nil {
        return false, err
    }
    defer rightMat.Close()
    
    // Find corners in both images
    leftFound, leftCorners := scp.findCorners(leftMat)
    rightFound, rightCorners := scp.findCorners(rightMat)
    
    // Both must be found
    if !leftFound || !rightFound {
        return false, nil
    }
    
    // Store sample
    objPts := scp.generateObjectPoints()
    scp.objectPoints = append(scp.objectPoints, objPts)
    scp.leftImagePoints = append(scp.leftImagePoints, leftCorners)
    scp.rightImagePoints = append(scp.rightImagePoints, rightCorners)
    
    // Update image size if needed
    if scp.imageSize.X == 0 {
        scp.imageSize = leftMat.Size()
    }
    
    scp.numSamples++
    return true, nil
}

func (scp *StereoCalibrationProcessor) Calibrate() (*StereoCalibration, error) {
    if scp.numSamples < scp.targetSamples {
        return nil, fmt.Errorf("insufficient samples: %d < %d", scp.numSamples, scp.targetSamples)
    }
    
    // Prepare data for calibration
    objectPoints := scp.convertObjectPoints()
    leftImagePoints := scp.convertImagePoints(scp.leftImagePoints)
    rightImagePoints := scp.convertImagePoints(scp.rightImagePoints)
    
    // Calibrate stereo cameras
    leftCameraMatrix := cv.NewMat()
    leftDistCoeffs := cv.NewMat()
    rightCameraMatrix := cv.NewMat()
    rightDistCoeffs := cv.NewMat()
    R := cv.NewMat()
    T := cv.NewMat()
    E := cv.NewMat()
    F := cv.NewMat()
    
    rms := cv.StereoCalibrate(
        objectPoints,
        leftImagePoints,
        rightImagePoints,
        &leftCameraMatrix,
        &leftDistCoeffs,
        &rightCameraMatrix,
        &rightDistCoeffs,
        scp.imageSize,
        &R,
        &T,
        &E,
        &F,
        cv.CalibFixIntrinsic|cv.CalibFixAspectRatio,
        cv.NewTermCriteria(cv.Count+cv.Eps, 30, 0.0001),
    )
    
    // Compute rectification
    R1 := cv.NewMat()
    R2 := cv.NewMat()
    P1 := cv.NewMat()
    P2 := cv.NewMat()
    Q := cv.NewMat()
    
    cv.StereoRectify(
        &leftCameraMatrix,
        &leftDistCoeffs,
        &rightCameraMatrix,
        &rightDistCoeffs,
        scp.imageSize,
        &R,
        &T,
        &R1,
        &R2,
        &P1,
        &P2,
        &Q,
        cv.StereoRectifyFlagZeroDisparity,
        -1,
        scp.imageSize,
        nil,
        nil,
    )
    
    return &StereoCalibration{
        LeftCameraMatrix:      leftCameraMatrix,
        LeftDistortionCoeffs:  leftDistCoeffs,
        RightCameraMatrix:     rightCameraMatrix,
        RightDistortionCoeffs: rightDistCoeffs,
        Rotation:              R,
        Translation:           T,
        Essential:             E,
        Fundamental:           F,
        LeftRectification:     R1,
        RightRectification:    R2,
        LeftProjection:        P1,
        RightProjection:       P2,
        DisparityToDepthMap:   Q,
        ImageSize:             scp.imageSize,
        NumSamples:            scp.numSamples,
        GridShape:             scp.gridSize,
        ReprojectionError:     rms,
    }, nil
}
```

### Disparity Map Calculation (Test Mode)

```go
func (sc *StereoCalibration) ComputeDisparity(leftFrame, rightFrame types.Frame) (gocv.Mat, error) {
    // Undistort and rectify images
    leftUndistorted := cv.NewMat()
    rightUndistorted := cv.NewMat()
    
    leftMat, _ := gocv.TensorToMat(leftFrame.Tensors[0])
    rightMat, _ := gocv.TensorToMat(rightFrame.Tensors[0])
    
    cv.Undistort(leftMat, &leftUndistorted, &sc.LeftCameraMatrix, &sc.LeftDistortionCoeffs, nil)
    cv.Undistort(rightMat, &rightUndistorted, &sc.RightCameraMatrix, &sc.RightDistortionCoeffs, nil)
    
    leftRectified := cv.NewMat()
    rightRectified := cv.NewMat()
    
    cv.Remap(leftUndistorted, &leftRectified, &sc.LeftMap1, &sc.LeftMap2, cv.InterpolationLinear)
    cv.Remap(rightUndistorted, &rightRectified, &sc.RightMap1, &sc.RightMap2, cv.InterpolationLinear)
    
    // Compute disparity map
    stereo := cv.NewStereoBM()
    defer stereo.Close()
    
    disparity := cv.NewMat()
    stereo.Compute(leftRectified, rightRectified, &disparity)
    
    // Normalize for display
    disparityNormalized := cv.NewMat()
    cv.Normalize(disparity, &disparityNormalized, 0, 255, cv.NormMinMax, cv.MatTypeCV8U)
    
    return disparityNormalized, nil
}
```

## Test Mode

When `--test` flag is used:
1. Load existing stereo calibration file
2. Undistort and rectify input frames using loaded calibration
3. Display rectified frames side-by-side
4. Optionally compute and display disparity map (`--disparity` flag)
5. Optionally compute reprojection error for new samples

## File Structure

```
cmd/calib_stereo/
├── DESIGN.md (this file)
├── main.go              # Main entry point
├── calibrate.go         # Stereo calibration processor
├── pattern.go           # Pattern detection
├── rectification.go     # Rectification maps
├── disparity.go         # Disparity map computation
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
2. Integrate shared input/output components for dual sources
3. Set up synchronized frame reading
4. Set up basic frame loop

### Phase 2: Pattern Detection
1. Implement synchronized chessboard corner detection
2. Add corner refinement for both images
3. Visualize detected patterns side-by-side

### Phase 3: Sample Collection
1. Implement synchronized sample tracking
2. Add progress display
3. Validate sample quality for both cameras

### Phase 4: Stereo Calibration
1. Implement stereo calibration computation
2. Compute rectification matrices
3. Generate rectification maps
4. Add reprojection error calculation

### Phase 5: Output
1. Implement JSON output format
2. Implement YAML output format
3. Implement GoCV native format (optional)

### Phase 6: Test Mode
1. Implement calibration file loading
2. Add rectification display
3. Add disparity map computation and display
4. Add validation features

### Phase 7: Integration and Testing
1. End-to-end testing with stereo cameras
2. Testing with video file pairs
3. Testing with image sequence pairs
4. Performance testing

## Open Questions

1. **Frame Synchronization**: How to handle frame synchronization between two sources?
   - Timestamp-based matching?
   - Index-based matching?
   - Manual synchronization trigger?

2. **Rectification Maps**: Should we pre-compute and store rectification maps?
   - Current: Compute on-the-fly in test mode
   - Future: Store maps with calibration data

3. **Calibration Flags**: Which OpenCV stereo calibration flags to use by default?
   - Current: Fix intrinsic, fix aspect ratio
   - Future: Make configurable

4. **Sample Quality**: How to handle mismatched pattern detection?
   - Reject entire pair if one fails?
   - Allow independent validation?

5. **Disparity Algorithm**: Which stereo matching algorithm to use?
   - StereoBM (block matching) - fast
   - StereoSGBM (semi-global block matching) - better quality
   - Make configurable?

