# Vision Package Specification

## Overview

The vision package provides computer vision algorithms and processing capabilities. It is primarily built on GoCV (OpenCV bindings) backend, with support for multiple backends planned.

## Components

### 1. Reader (`pkg/vision/reader`)

**Purpose**: Image and video input from various sources

**Backends**:
- **GoCV**: Device capture (camera), video file, image file
- **Default**: Fallback implementation

**Types**:
- **Device Reader**: Camera capture (`reader.device.gocv.go`)
- **Video Reader**: Video file reading (`reader.video.gocv.go`)
- **Image Reader**: Image file reading (`reader.gocv.go`)

**Integration**:
- Pipeline step: `steps.Source`
- Registered as: `"rmat"` (GoCV reader)

**Characteristics**:
- Supports multiple paths/files
- Repeat mode (loop playback)
- Frame index and timestamp tracking

**Questions**:
1. Should reader support streaming from network?
2. How to handle camera disconnection?
3. Should reader support different camera backends (V4L2, DirectShow)?
4. How to handle video format compatibility?
5. Should reader support image sequences?
6. How to handle reader performance (buffering, threading)?
7. Should reader support frame rate control?
8. How to handle reader errors (file not found, format error)?

### 2. Detection (`x/vision/detect`)

**Purpose**: Detection of features and patterns in images

#### Corner Detection (`x/vision/detect/corners`)

**Purpose**: Chessboard corner detection for camera calibration

**Operations**:
- Chessboard corner detection (`FindChessboardCornersSB`)
- Corner refinement to sub-pixel accuracy (`CornerSubPix`)
- Corner drawing/visualization
- Object point generation for calibration

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, types.Tensor]` for pipeline usage
- Direct `Detect()` method for standalone usage
- Used by `cmd/calibrate` for camera calibration

**Questions**:
1. Should corner detection support other calibration targets (circles, ArUco)?
2. How to handle corner detection performance?
3. Should corner detection support GPU acceleration?

#### Feature Detection (`x/vision/detect/features`)

**Purpose**: Feature (keypoint) detection

**Operations**:
- Multiple detector types: ORB, SIFT, FAST, GFTT, KAZE, AKAZE, BRISK
- Keypoint detection only (no descriptors)
- Keypoint drawing/visualization

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[visiontypes.KeypointsWithImage, types.Tensor]`
- Outputs `KeypointsWithImage` structure containing image tensor and keypoints

**Questions**:
1. Should feature detection support GPU acceleration?
2. How to optimize for embedded systems?
3. Should feature detection support custom detector configurations?

### 3. Transform (`x/vision/transform`)

**Purpose**: Image transformations and conversions

#### Color Transform (`x/vision/transform/color`)

**Purpose**: Color space conversions

**Operations**:
- Color space conversion (RGB, BGR, HSV, Grayscale, etc.)
- Format conversion

**Backend**: GoCV

**Questions**:
1. Should color transform support gamma correction?
2. How to handle color transform performance?
3. Should color transform support custom color spaces?
4. How to optimize color transform for embedded systems?

#### Format Transform (`x/vision/transform/format`)

**Purpose**: Image format conversions

**Operations**:
- Format conversion (Mat ↔ image.Image)
- Stereo processing (`format.stereo.go`)

**Backend**: GoCV

**Status**: ⚠️ Uses old pipeline pattern - needs refactoring to `chain.Processor`

**Questions**:
1. Should format transform support different image formats?
2. How to handle format conversion performance?
3. Should format transform support lossless conversion?
4. How to optimize format transform for embedded systems?
5. Should format transform support batch processing?

#### Rectification (`x/vision/transform/rectify`)

**Purpose**: Stereo image rectification

**Operations**:
- Stereo rectification using precomputed rectification maps
- Remapping using `Remap` operation

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, types.Tensor]`
- Uses `RectificationMaps` structure (Map1, Map2)
- Maps should be precomputed using `InitUndistortRectifyMap`

**Questions**:
1. Should rectification support online map computation?
2. How to handle rectification performance?
3. Should rectification support different interpolation methods?

#### Undistortion (`x/vision/transform/undistort`)

**Purpose**: Lens distortion correction

**Operations**:
- Undistortion using camera calibration parameters (camera matrix, distortion coefficients)
- Optional new camera matrix support

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, types.Tensor]`
- Used for removing lens distortion from images

**Questions**:
1. Should undistortion support different distortion models?
2. How to handle undistortion performance?
3. Should undistortion support GPU acceleration?

#### Affine Transformation (`x/vision/transform/affine`)

**Purpose**: Affine image transformations

**Operations**:
- Affine transformation using 2x3 transformation matrix
- Warp affine operation
- Optional output size specification

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, types.Tensor]`
- Uses `WarpAffine` for transformation

**Questions**:
1. Should affine support other transformation types (perspective, etc.)?
2. How to handle affine transformation performance?

#### Crop (`x/vision/transform/crop`)

**Purpose**: Image cropping

**Operations**:
- Crop subimages from rectangles
- Multiple crops per input image
- Rectangle validation

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, CropRectangles]`
- Input: `CropRectangles` structure (image tensor + list of rectangles)
- Output: Multiple image tensors (one per rectangle)

**Questions**:
1. Should crop support different interpolation methods?
2. How to handle crop performance for many rectangles?
3. Should crop support rotation or scaling?

### 4. Extract (`x/vision/extract`)

**Purpose**: Feature extraction and DNN inference

#### Features (`x/vision/extract/features`)

**Purpose**: Feature descriptor extraction

**Algorithms**:
- **ORB**: Oriented FAST and Rotated BRIEF
- **SIFT**: Scale-Invariant Feature Transform
- **KAZE**: KAZE feature detector
- **AKAZE**: Accelerated KAZE
- **BRISK**: Binary Robust Invariant Scalable Keypoints

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[visiontypes.DescriptorsWithImage, visiontypes.KeypointsWithImage]`
- Takes `KeypointsWithImage` as input (from feature detection)
- Outputs `DescriptorsWithImage` (image + keypoints + descriptors)

**Status**: ⚠️ Partially implemented - uses `DetectAndCompute` instead of separate `Compute` method

**Questions**:
1. Should features support different extractor configurations?
2. How to handle feature extraction performance?
3. Should features support GPU acceleration?
4. How to optimize features for embedded systems?
5. Should features support computing descriptors for existing keypoints only?

### 5. Matching (`x/vision/match`)

**Purpose**: Feature and corner matching

#### Feature Matching (`x/vision/match/features`)

**Purpose**: Feature matching between images

**Matchers**:
- **FLANN**: Fast Library for Approximate Nearest Neighbors

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, types.Tensor]`
- Matches descriptors between two images
- Uses KNN matching (k=2) for Lowe's ratio test

**Status**: ⚠️ Basic implementation - needs enhancement (Lowe's ratio test, filtering, etc.)

**Questions**:
1. Should feature matching support different matcher types (brute force)?
2. How to handle feature matching performance?
3. Should feature matching support GPU acceleration?
4. How to optimize feature matching for embedded systems?
5. Should feature matching support Lowe's ratio test and filtering?

#### Corner Matching (`x/vision/match/corners`)

**Purpose**: Corner matching between images

**Backend**: GoCV

**Integration**:
- Implements `chain.Processor[types.Tensor, types.Tensor]`
- Placeholder implementation

**Status**: ⚠️ Placeholder - not yet implemented

**Questions**:
1. What corner matching algorithm should be used?
2. How to handle corner matching for stereo calibration?
3. Should corner matching support corner tracking?

#### DNN (`x/vision/extract/dnn`)

**Purpose**: Deep neural network inference

**Backends**:
- **Default**: CPU
- **OpenVINO**: Intel OpenVINO (planned)
- **CUDA**: NVIDIA CUDA (planned)
- **TensorRT**: NVIDIA TensorRT (planned)

**Targets**:
- **CPU**: CPU inference
- **GPU**: GPU inference (planned)
- **VPU**: Vision Processing Unit (planned)

**Model Formats**:
- Caffe (`.caffemodel`)
- TensorFlow (`.pb`)
- ONNX (`.onnx`) (planned)
- TensorFlow Lite (`.tflite`) (planned)

**Integration**:
- Pipeline step: `steps.Processor`
- Registered as: `"dnn"`

**Questions**:
1. Should DNN support multiple models simultaneously?
2. How to handle DNN inference performance?
3. Should DNN support model quantization?
4. How to optimize DNN for embedded systems?
5. Should DNN support different input formats?
6. How to handle DNN model loading errors?
7. Should DNN support dynamic batch processing?
8. How to handle DNN model versioning?
9. Should DNN support model caching?
10. How to handle DNN backend selection?

### 6. Display (`x/vision/display`)

**Purpose**: Image visualization

**Backend**: GoCV

**Integration**:
- Pipeline step: `steps.Sink`
- Registered as: `"display"`

**Characteristics**:
- Window-based display
- Frame rate control

**Questions**:
1. Should display support multiple windows?
2. How to handle display performance?
3. Should display support different display modes (fullscreen, windowed)?
4. How to optimize display for embedded systems?
5. Should display support annotations (text, shapes)?
6. How to handle display errors (window closed)?

### 7. Writer (`x/vision/writer`)

**Purpose**: Image and video output

**Backends**:
- **GoCV**: Video writer, image writer
- **Default**: Null sink (testing)

**Types**:
- **Video Writer**: Video file writing (`write.gocv.go`)
- **Image Writer**: Image file writing (`write.images.go`)
- **Null Sink**: Drop frames (`sink.null.go`)

**Integration**:
- Pipeline step: `steps.Sink`

**Characteristics**:
- Supports multiple output formats
- Frame rate control

**Questions**:
1. Should writer support streaming to network?
2. How to handle writer performance (encoding)?
3. Should writer support different video codecs?
4. How to optimize writer for embedded systems?
5. Should writer support image sequences?
6. How to handle writer errors (disk full, format error)?
7. Should writer support compression settings?

## Backend Abstraction

### Current Implementation

**GoCV Backend**:
- Primary backend for vision operations
- OpenCV bindings
- Full feature set

**Default Backend**:
- Fallback implementations
- Limited features

### Planned Backends

**TensorFlow**:
- TensorFlow inference
- TensorFlow Lite support

**TensorFlow Lite**:
- Lightweight inference for embedded
- Optimized for mobile/embedded

**Native**:
- Pure Go implementations
- No external dependencies
- Embedded-friendly

**Questions**:
1. Should backends support runtime switching?
2. How to handle backend-specific features?
3. Should backends support fallback mechanisms?
4. How to handle backend compatibility?
5. Should backends support backend-specific optimizations?

## Pipeline Integration

### Step Types

**Source Steps**:
- Reader (camera, video, image)

**Processor Steps**:
- Detect (corners, features)
- Transform (color, format, rectify, undistort, affine, crop)
- Extract (features, DNN)
- Match (features, corners)

**Sink Steps**:
- Display
- Writer (video, image, null)

**Questions**:
1. Should vision steps support dynamic configuration?
2. How to handle step lifecycle (start/stop/pause/resume)?
3. Should steps support resource cleanup?
4. How to handle step errors (backend unavailable)?

## Performance

### Current Characteristics

- GoCV backend: Full OpenCV features
- Memory usage: Dependent on image size and backend
- Performance: CPU-bound operations

**Questions**:
1. Should we support GPU acceleration?
2. How to optimize for embedded systems (memory, CPU)?
3. Should we support memory pooling?
4. How to handle performance profiling?
5. Should we support batch processing?
6. How to handle performance monitoring?

### Optimization Strategies

1. **Memory**:
   - Image buffer pooling
   - Lazy loading
   - In-place operations where possible

2. **Computation**:
   - SIMD optimization (if available)
   - GPU acceleration (planned)
   - Parallel processing (planned)

3. **I/O**:
   - Buffered I/O
   - Async I/O (planned)
   - Zero-copy where possible

**Questions**:
1. Should optimizations be automatic or explicit?
2. How to handle optimization selection?
3. Should optimizations support runtime tuning?
4. How to benchmark optimizations?

## Design Questions

### Architecture

1. **Backend Design**:
   - Should backends be pluggable?
   - How to handle backend-specific features?
   - Should backends support backend chaining?

2. **Step Design**:
   - Should steps support composability?
   - How to handle step configuration?
   - Should steps support step chaining?

3. **Data Flow**:
   - How to handle data conversion between backends?
   - Should data flow support zero-copy?
   - How to handle data versioning?

### Compatibility

4. **Platform Support**:
   - How to handle missing backends on embedded platforms?
   - Should we provide fallback implementations?
   - How to handle platform-specific optimizations?

5. **Backward Compatibility**:
   - How to handle breaking changes?
   - Should we support migration tools?
   - How to handle API deprecation?

6. **Versioning**:
   - How to handle model versioning?
   - Should we support model migration?
   - How to handle algorithm versioning?

## Known Issues

1. **Backend Limitation**: Only GoCV backend fully implemented
2. **Limited Features**: Many features are partial or missing (stereo, calibration)
3. **No Testing**: Missing comprehensive tests
4. **No Documentation**: Incomplete API documentation
5. **No Optimization**: Missing performance optimizations
6. **No Embedded Support**: Limited embedded system support

## Potential Improvements

1. **Complete Implementation**: Finish missing features (stereo, calibration, odometry)
2. **Backend Support**: Support for TensorFlow, TensorFlow Lite, native backends
3. **Testing**: Comprehensive test suite
4. **Documentation**: Complete API documentation
5. **Optimization**: Performance optimizations (GPU, SIMD, memory pooling)
6. **Embedded Support**: Better embedded system support (optimizations, fallbacks)
7. **Calibration**: Camera calibration support
8. **Odometry**: Monocular and stereo odometry
9. **SLAM**: Simultaneous Localization and Mapping support
10. **Visualization**: Better visualization tools

