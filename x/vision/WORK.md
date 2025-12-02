# Vision Package - Work Tracking

## Overview

This document tracks the current work session progress, next session plans, and overall progress (0-100%) for implementing a full suite of algorithms for stereo/mono visual odometry, place recognition, and graph-based navigation.

**IMMUTABLE SECTIONS** (Do not modify):
- Rules and Guidelines
- Overall Progress Tracking
- Potential Issues

**MUTABLE SECTIONS** (Update as work progresses):
- Current Work Session Tasks
- Next Work Session Tasks

---

## Rules and Guidelines (IMMUTABLE)

### Development Principles

1. **LEAN Principles**: Write maintainable and expandable Go code
2. **Usability Priority**: User experience is paramount
3. **No Code Repetition**: DRY (Don't Repeat Yourself) - repeating code is punishable
4. **Use Existing Packages**: Implementing loops when `mat/vec/tensor/dsp/filter/graph` can be used is punishable even more
5. **Type Safety**: No need to convert matrices/vectors from interfaces - know exact types (e.g., `vec.Vector` vs `vec.Vector3D`) and cast directly
6. **Test Coverage**: Misuse of types must be caught with unit tests using `testify` package
7. **UP TO DATE**: Keep this document and relevant SPEC.md and PLAN.md(if any) up to date
8. **NO STUTTERING**: Avoid stuttering and repeating, e.g. such names as obj.NewObj(), obj.ObjectQualifier structs... e.g. `odometry.OdometryFilter` => `odometry.Filter`.
9. **Options**: Prefer options patterns over long argument lists
10. **Vararg**: Variable arguments can be used more often for convenience where applicable
11. **SIMPLE NAMES**: Use simple, concise names. Avoid verbose names like `StartMeasurementListener` → use `Start`. `StopMeasurementListener` → use `Stop`.
12. **GETTERS**: Getters must NOT start with "Get"! Use simple names like `Rotation()` not `GetRotation()`, `BackgroundNum()` not `GetBackgroundNum()`.
13. **TODO MANAGEMENT**: ALWAYS update WORK.md when completing tasks. Move tasks from pending → in_progress → completed. Verify implementation matches TODO.
14. **TASK PRIORITIZATION**: Prioritize tasks in the queue, keep queue full, work on ONE task at a time. Update WORK.md before and after each task.

### Go Best Practices

- **Go Version**: Go 1.25 (assumed latest features)
- **Logging**: Use `log/slog` package with `-v=N` (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE) and `-vv` (shortcut for `-v=4`)
- **Testing**: Use `testify` package for all tests
- **Error Handling**: Never ignore errors, wrap with context
- **Function Length**: Maximum 30 lines (extract when makes sense, but don't split just for splitting sake)
- **SOLID Principles**: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **Destination pattern**: Avoid allocating new matrices/tensors/vectors - attempt to preallocate needed objects and use them as destination in calculations (destination parameter goes first). Only allocate when returning values or creating objects for callbacks. Preallocate reusable buffers in structs. BE VERY MINDFUL of sharing vectors/matrices/arrays/slices/tensors between goroutines. **CRITICAL**: Methods called repeatedly (e.g., `Process()`) MUST use destination pattern. Methods called once (e.g., `New()`) can allocate.
- **NO ALLOCATIONS IN Process() CALLSTACK**: All allocations in `chain.Processor.Process()` must be preallocated in struct fields. Preallocate scratch buffers (cv.Mat, slices, etc.) to avoid allocations during processing. This is critical for real-time performance.
- **Tensors are NOT zero-initialized**: Never assume tensor data is zero-initialized. Always explicitly initialize tensor data when creating new tensors or buffers.

### Package Usage

**MUST READ before implementing**:
- `x/math/vec` - Vector operations (`vec.Vector`, `vec.Vector3D`, etc.)
- `x/math/mat` - Matrix operations (`mat.Matrix`, row-major layout)
- `x/math/tensor` - Tensor operations (`tensorgocv.Tensor`, `types.Tensor`, etc.)
- `x/math/graph` - Graph data structures and algorithms (A*, Dijkstra, BFS, DFS, trees)
- `x/buffer` - Generic buffers (Ring, Linear, etc.) - use `buffer.Ring[types.Tensor]` for tensor buffers
- `x/chain` - Processor chaining infrastructure (`chain.Processor`, `chain.New`, `chain.NewAdapter`)
- `x/marshaller/gocv` - GoCV marshaller and camera controller
- `cmd/display/source` - Camera/video source interfaces
- `cmd/display/destination` - Display/output interfaces

**Know exact types**: Cast directly (e.g., `vec.Vector3D(xyz).XYZ()`, `mat.Row(0).(vec.Vector)`)

### Code Quality Standards

- **Testability**: Write testable code, avoid hard-to-mock dependencies
- **Composition over Inheritance**: Prefer composition
- **Return Early**: Reduce nesting
- **Keep Arguments Low**: Fewer arguments is better
- **Package by Feature**: Not by layer
- **Accept Interfaces, Return Structs**: Keep interfaces small (1-3 methods)
- **Context for Cancellation**: Use `context.Context` for cancellation and timeouts
- **REUSABILITY**: Always implement reusable functionality in shared packages (`x/*`), not in application-specific code (`cmd/*`). Application commands should reuse shared functionality.
- **Build Tags**: properly utilize build tags - if there is platform specific implementations and you need to add features - just add to those platform specific implementations, not some weird ass wrapper. users should know what tags they need to use and on what platforms and would create platform independent wrappers if they need it.

---

## Current Work Session Tasks (MUTABLE)

**CRITICAL**: Always update this section when starting/finishing tasks. Move tasks from pending → in_progress → completed. Verify implementation matches TODO.

### Session Goal

Implement full suite of visual odometry algorithms, place recognition, and graph-based navigation. ⭐ **IN PROGRESS**

### Chain Processor Integration (NEW)

**Status**: ✅ **COMPLETED** - Foundation implemented

Vision algorithms can now implement `chain.Processor[types.Tensor, types.Tensor]` for composable processing pipelines:

- [x] **Use Existing Generic Buffers**: No need for custom TensorBuffer - use `buffer.Ring[types.Tensor]`, `buffer.Linear[types.Tensor]`, etc.
- [x] **Feature Extraction Processor** (`x/vision/extract/features/processor.go`): Example implementation
  - [x] Implements `chain.Processor[types.Tensor, types.Tensor]`
  - [x] Extracts ORB features from input image tensors
  - [x] Returns descriptor tensors as output
  - [x] Preallocates scratch space (cv.Mat) to avoid allocations in Process()
- [x] **Example Usage** (`x/vision/extract/features/processor_example.go`): Demonstrates usage
  - [x] Single processor usage with `buffer.Ring[types.Tensor]`
  - [x] Chain composition example
- [x] **Design Document** (`x/vision/CHAIN_PROCESSOR_DESIGN.md`): Architecture documentation
  - [x] Updated to use existing generic buffers
  - [x] Documents preallocation requirements
  - [x] Notes tensor initialization requirements

**Next Steps**:
- [ ] Implement additional processors (matching, stereo, odometry) using chain.Processor
- [ ] Ensure all processors preallocate scratch space (no allocations in Process())
- [ ] Create helper utilities for common chains
- [ ] Integrate with `cmd/odometry` command

### Tasks (In Order)

1. **Read Required SPECs/PLANs and Python Reference**: ⚠️ **PENDING**
   - [ ] Read `x/vision/SPEC.md` - Current specification (OUTDATED - needs review)
   - [ ] Read Python `EasyVision/engine/visualodometry_*.py` - Reference implementations
   - [ ] Read Python `EasyVision/engine/bowvocabulary.py` - Bag of Words reference
   - [ ] Read Python `EasyVision/engine/topologicalmap.py` - Graph-based navigation reference
   - [ ] Read Python `EasyVision/processors/featureextractor.py` - Feature extraction reference
   - [ ] Read `x/math/graph/SPEC.md` - Graph package specification
   - [ ] Read `x/math/vec/vec.go` - Vector types and operations
   - [ ] Read `x/math/mat/mat.go` - Matrix types and operations
   - [ ] Read `x/marshaller/gocv/SPEC.md` - GoCV marshaller and camera API
   - [ ] Review existing `x/vision` code structure

2. **Clean Up Outdated Code**: ✅ **COMPLETED**
   - [x] Review existing `x/vision/extract/features` implementation ✅
   - [x] Review existing `x/vision/extract/dnn` implementation ✅
   - [x] Fixed allocation issue in `features.gocv.go` (preallocated scratchMask) ✅
   - [x] Improved error context in `dnn.gocv.go` (fmt.Errorf with details) ✅
   - [x] Ensured alignment with current Go patterns ✅
   - Note: Old pipeline pattern code kept for backward compatibility

3. **Detection, Extraction, and Matching** (Priority: HIGH - Foundation for odometry): ⚠️ **PARTIALLY COMPLETED**
   - [x] **Folder Structure**: Created `detect/`, `extract/`, `match/` folders ✅
   - [x] **Detect/Corners**: Corner detection processor (chessboard corners for calibration) ✅
   - [x] **Detect/Features**: Feature detection processor (keypoints) ✅
   - [x] **Extract/Features**: Feature extraction processor (descriptors) ✅
   - [x] **Match/Corners**: Corner matching processor (placeholder) ✅
   - [x] **Match/Features**: Feature matching processor (basic implementation) ✅
   - [x] **Detect/Corners Integration**: `cmd/calibrate` updated to use `x/vision/detect/corners` ✅
     - [x] Moved corner detection from `cmd/calibrate/corners` to `x/vision/detect/corners` ✅
     - [x] Added backwards compatibility (`Detector` type alias, `NewDetector()` function) ✅
     - [x] Added `Detect()`, `Draw()`, `GenerateObjectPoints()` functions ✅
   
   - [x] **Feature Detection** (`detect/features`): ✅
     - [x] ORB detector ✅
     - [x] SIFT detector ✅
     - [x] FAST detector ✅
     - [x] GFTT (Good Features to Track) ✅
     - [x] KAZE/AKAZE detectors ✅
     - [x] BRISK detector ✅
     - [x] Draw method for visualization ✅
     - **NOTE**: `gocv.DrawKeyPoints()` requires `color.RGBA` type (from `image/color` package) for color parameter, NOT `gocv.Scalar` or `image.RGBA`
   - [ ] **Feature Extraction** (`extract/features`):
     - [x] ORB descriptors ✅
     - [ ] SIFT descriptors
     - [ ] Binary descriptor support (ORB, BRISK, AKAZE)
     - [ ] Floating-point descriptor support (SIFT, SURF, KAZE)
   - [x] **Blob Extraction** (`x/vision/extract/blobs`): ✅
     - [x] Implements `chain.Processor[visiontypes.KeypointsWithImage, visiontypes.ImageWithMask]` ✅
     - [x] Blob extraction from masks using SimpleBlobDetector ✅
     - [x] Configurable area constraints (min/max blob size) ✅
     - [x] Mask preprocessing (invert, blur, threshold) ✅
     - [x] Draw method for visualization ✅
   - [ ] **Feature Matching** (`match/features`):
     - [x] FLANN matcher (basic implementation) ✅
     - [ ] Brute-force matcher (Hamming distance for binary, L2 for float)
     - [ ] Lowe's ratio test implementation
     - [ ] Distance threshold filtering
     - [ ] Cross-check matching
   - [ ] **Corner Matching** (`match/corners`):
     - [ ] Corner correspondence matching
     - [ ] Corner tracking across frames
   - [ ] **Feature Tracking**:
     - [ ] Lucas-Kanade optical flow (for FAST/GFTT)
     - [ ] Pyramidal LK optical flow
     - [ ] Feature tracking across frames
     - [ ] Outlier rejection (RANSAC)

4. **Transform Processors** (Priority: HIGH - Image transformations): ✅ **COMPLETED**
   - [x] **Rectify** (`x/vision/transform/rectify`): Stereo image rectification ✅
     - [x] Implements `chain.Processor[types.Tensor, types.Tensor]` ✅
     - [x] Uses precomputed rectification maps (Map1, Map2) ✅
     - [x] Uses `Remap` for rectification ✅
   - [x] **Undistort** (`x/vision/transform/undistort`): Lens distortion correction ✅
     - [x] Implements `chain.Processor[types.Tensor, types.Tensor]` ✅
     - [x] Uses camera matrix and distortion coefficients ✅
     - [x] Optional new camera matrix support ✅
   - [x] **Affine** (`x/vision/transform/affine`): Affine transformations ✅
     - [x] Implements `chain.Processor[types.Tensor, types.Tensor]` ✅
     - [x] Uses 2x3 transformation matrix ✅
     - [x] Optional output size specification ✅
   - [x] **Crop** (`x/vision/transform/crop`): Image cropping ✅
     - [x] Implements `chain.Processor[types.Tensor, CropRectangles]` ✅
     - [x] Multiple crops per input image ✅
     - [x] Rectangle validation ✅
   - [x] **Format** (`x/vision/transform/format`): ✅ **REFACTORED**
     - [x] Refactored to use `chain.Processor` pattern ✅
     - [x] MatToImageProcessor: `chain.Processor[ImageWithTensor, types.Tensor]` ✅
     - [x] ImageToMatProcessor: `chain.Processor[types.Tensor, ImageWithTensor]` ✅
     - [x] DataToStereoProcessor: `chain.Processor[StereoPair, types.Tensor]` ✅
     - [x] Old pipeline pattern code kept for backward compatibility ✅
   - [x] **Histogram Backprojection** (`x/vision/transform/histogram`): ✅
     - [x] Implements `chain.Processor[visiontypes.ImageWithMask, types.Tensor]` ✅
     - [x] Histogram backprojection for mask generation ✅
     - [x] Supports single or multiple histograms ✅
     - [x] Configurable channels and ranges (default: HSV H and S) ✅
     - [x] Optional color range filtering ✅
     - [x] Option to combine multiple masks ✅
     - [x] Option to invert masks ✅
     - [x] Helper function `CalculateHistogram()` for computing histograms ✅
   - [x] **Background Separation** (`x/vision/transform/background`): ✅
     - [x] Implements `chain.Processor[visiontypes.ImageWithMask, types.Tensor]` ✅
     - [x] Background subtraction using MOG2 algorithm ✅
     - [x] Configurable learning frames (default: 200 frames) ✅
     - [x] Frame counter for controlling learning phase ✅
     - [x] Morphological filtering and thresholding ✅
     - [x] MOG and GMG algorithms can be added when gocv supports them ✅

5. **Camera Types** (Priority: HIGH - Essential camera utilities): ✅ **COMPLETED**
   - [x] **Camera** (`x/vision/types/camera.go`): ✅
     - [x] Pinhole camera model with intrinsic parameters ✅
     - [x] Camera matrix, distortion coefficients, resolution ✅
     - [x] Optional rectification and projection matrices ✅
     - [x] Utility methods: Width(), Height(), FocalPoint(), Center() ✅
     - [x] Tensor access methods for camera matrix and distortion ✅
     - [x] ProjectPoints(), BackprojectPoints(), UndistortPoints() (TODOs for gocv API verification) ✅
   - [x] **StereoCamera** (`x/vision/types/stereo_camera.go`): ✅
     - [x] Two Camera instances (Left, Right) ✅
     - [x] Stereo extrinsics: R, T, E, F, Q matrices ✅
     - [x] Validation (camera sizes must match) ✅
     - [x] Baseline calculation ✅
     - [x] ReprojectTo3D() using Q matrix ✅
     - [x] TriangulatePoints() (TODO for gocv API verification) ✅
     - [x] Getters: Rotation(), Translation(), Essential(), Fundamental(), Q() ✅
     - **NOTE**: Getters must NOT start with "Get" - use simple names like `Rotation()` not `GetRotation()`

6. **Stereo Processing** (Priority: HIGH - Required for stereo odometry): ⚠️ **PARTIALLY COMPLETED**
   - [x] **Stereo Rectification** (`x/vision/transform/rectify`): ✅
     - [x] Stereo rectification processor implemented ✅
     - [ ] ⚠️ **Integration**: `cmd/calibrate` has its own `Rectify()` method - should use processor
   - [ ] **Rectification Map Precomputation**:
     - [ ] Helper function/processor for computing rectification maps
     - [ ] Integration with calibration results
   - [x] **Stereo Matching** (`x/vision/stereo/matching`): ✅
     - [x] Semi-global block matching (SGBM) processor ✅
     - [x] Disparity map computation ✅
   - [x] **Depth Computation** (`x/vision/stereo/depth`): ✅
     - [x] Disparity-to-depth conversion processor (using Q matrix) ✅
     - [x] Combined processor (stereo pair → depth in one step) ✅
   - [ ] **3D Reconstruction**:
     - [ ] Triangulation (3D point from stereo pair + disparity)
     - [ ] Point cloud generation
     - [ ] Depth filtering (min/max distance, disparity consistency)

7. **Monocular Visual Odometry** (Priority: HIGH): ⚠️ **PENDING**
   - [ ] **2D Visual Odometry** (reference: Python `VisualOdometry2DEngine`):
     - [ ] Essential matrix estimation (5-point algorithm or 8-point algorithm)
     - [ ] RANSAC for robust essential matrix estimation
     - [ ] Pose recovery from essential matrix (4 solutions, disambiguate)
     - [ ] Reprojection error calculation
     - [ ] Bundle adjustment (optional, for refinement)
     - [ ] Scale estimation (for monocular - requires initialization or scale-aware matching)
   - [ ] **Feature Tracking Mode**:
     - [ ] FAST/GFTT feature tracking (Lucas-Kanade optical flow)
     - [ ] Track management (new tracks, lost tracks, outlier rejection)
     - [ ] Essential matrix from tracked features
   - [x] **Feature Matching Mode**: ✅
     - [x] ORB/SIFT feature matching between frames ✅ (via `match/features`)
     - [x] Essential matrix from matched features ✅ (via `odometry/pose`)
     - [ ] Lowe's ratio test for match filtering (can be added to `match/features`)
   - [ ] **Pose Estimation** (`x/vision/odometry/pose`): ⚠️ **PARTIALLY COMPLETED - PLACEHOLDER**
     - [ ] Essential matrix computation from matched features
       - **LIMITATION**: gocv doesn't expose `FindEssentialMat` or `FindFundamentalMat`
       - Requires manual implementation using 5-point/8-point algorithm + RANSAC
     - [ ] Pose recovery (rotation + translation) from essential matrix
       - **LIMITATION**: gocv may not expose `RecoverPose` - requires verification
     - [x] Processor structure and buffer handling ✅
     - [x] Inliers mask output structure ✅

8. **Stereo Visual Odometry** (Priority: HIGH): ⚠️ **PENDING**
   - [ ] **Stereo Odometry** (reference: Python `VisualOdometryStereoEngine`):
     - [ ] Stereo matching and 3D point triangulation
     - [ ] SolvePnP with RANSAC (3D-2D matching between frames)
     - [ ] Reprojection error thresholding
     - [ ] Iterative refinement (adjust parameters if error too high)
     - [ ] Pose integration (relative pose → absolute pose)
   - [ ] **Stereo Feature Matching**:
     - [ ] Left-right stereo matching (epipolar constraint)
     - [ ] Temporal matching (left-left, right-right)
     - [ ] 3D point triangulation from stereo pair
     - [ ] 3D point matching across frames

9. **3D-2D Visual Odometry** (Priority: MEDIUM): ⚠️ **PENDING**
   - [ ] **3D-2D Odometry** (reference: Python `VisualOdometry3D2DEngine`):
     - [ ] Previous frame 3D points (from stereo or prior)
     - [ ] Current frame 2D features
     - [ ] 3D-2D matching (using descriptors)
     - [ ] SolvePnP with RANSAC
     - [ ] Pose refinement

10. **Bag of Words (BOW)** (Priority: MEDIUM - For place recognition): ⚠️ **PENDING**
   - [ ] **Vocabulary Building** (reference: Python `BOWVocabularyBuilderEngine`):
     - [ ] Feature collection from training images
     - [ ] Vocabulary construction (k-means clustering or DBoW3-style hierarchical k-means)
     - [ ] Vocabulary saving/loading (file format: DBoW3 compatible or custom)
     - [ ] TF-IDF weighting
     - [ ] Vocabulary statistics (word frequencies, etc.)
   - [ ] **BOW Descriptor Generation**:
     - [ ] Feature extraction from image
     - [ ] Vocabulary matching (assign features to visual words)
     - [ ] BOW histogram generation (TF-IDF weighted)
     - [ ] Normalization
   - [ ] **BOW Matching**:
     - [ ] Database for BOW descriptors (place database)
     - [ ] Query matching (L1/L2 distance or cosine similarity)
     - [ ] Top-K place retrieval
     - [ ] Score thresholding

11. **Place Recognition** (Priority: MEDIUM): ⚠️ **PENDING**
   - [ ] **Place Database**:
     - [ ] Place storage (BOW descriptor + pose + image/view)
     - [ ] Place indexing (for fast retrieval)
     - [ ] Place matching (BOW similarity + spatial consistency)
   - [ ] **Pose Recognition**:
     - [ ] Given current view and target view, calculate required pose transformation
     - [ ] Essential matrix from current-target view matching
     - [ ] Relative pose estimation (rotation + translation)
     - [ ] Absolute pose if both views have known poses
   - [ ] **Neural Network-Based** (Future work):
     - [ ] Pre-trained place recognition models (NetVLAD, SuperPoint, etc.)
     - [ ] Model loading and inference
     - [ ] Descriptor extraction from neural networks
     - [ ] Integration with BOW system

12. **Graph-Based Navigation** (Priority: MEDIUM - Using `x/math/graph`): ⚠️ **PENDING**
    - [ ] **Topological Map** (reference: Python `TopologicalMap`):
      - [ ] Node representation (pose + BOW descriptor + links)
      - [ ] Graph construction using `x/math/graph` primitives
      - [ ] Place recognition for node matching
      - [ ] Edge creation (spatial proximity + visual similarity)
      - [ ] Graph saving/loading
    - [ ] **Path Planning**:
      - [ ] Graph-based pathfinding (A* or Dijkstra using `x/math/graph`)
      - [ ] Path smoothing (optional)
      - [ ] Multi-goal pathfinding
    - [ ] **Navigation**:
      - [ ] Current pose localization (place recognition in graph)
      - [ ] Target node selection
      - [ ] Path execution (visual servoing or waypoint following)
      - [ ] Loop closure detection

13. **Command Programs** (Priority: HIGH - User-facing): ⚠️ **PENDING**
    - [ ] **`cmd/odometry`** (reference: Python `bin/performance_visualodometry_*.py`):
      - [ ] Monocular odometry command (`odometry mono`)
      - [ ] Stereo odometry command (`odometry stereo`)
      - [ ] 3D-2D odometry command (`odometry 3d2d`)
      - [ ] Source flags (camera, video, images)
      - [ ] Display flags (window, output video, output images)
      - [ ] Calibration flags (camera calibration file)
      - [ ] Odometry parameters (feature type, number of features, matching thresholds)
      - [ ] Pose output (JSON, CSV, trajectory file)
      - [ ] Visualization (trajectory, matches, features, 3D map)
    - [ ] **Integration with existing `cmd/calibrate`**:
      - [ ] Ensure calibration output compatible with odometry input
      - [ ] Test calibration → odometry pipeline

14. **Testing and Validation** (Priority: HIGH): ⚠️ **PENDING**
    - [ ] **Unit Tests**:
      - [ ] Feature extraction tests
      - [ ] Feature matching tests
      - [ ] Essential matrix tests
      - [ ] Pose recovery tests
      - [ ] Triangulation tests
      - [ ] BOW vocabulary tests
      - [ ] Graph navigation tests
    - [ ] **Integration Tests**:
      - [ ] End-to-end odometry pipeline tests
      - [ ] KITTI dataset validation (if available)
      - [ ] EuRoC dataset validation (if available)
    - [ ] **Performance Tests**:
      - [ ] Real-time performance benchmarks
      - [ ] Memory usage profiling
      - [ ] Accuracy metrics (ATE, RPE)

---

## Next Work Session Tasks (MUTABLE)

**CRITICAL**: Prioritize tasks in the queue, keep queue full, work on ONE task at a time. Update WORK.md before and after each task.

### Planned Next Steps (Prioritized)

1. **Phase 0 - Foundation** (Priority: HIGHEST) ⭐ **START HERE**
   - [ ] **Task**: Clean up and modernize existing `x/vision` code
     - [ ] Review `x/vision/extract/features` implementation
     - [ ] Review `x/vision/extract/dnn` implementation
     - [ ] Update to use current Go patterns and `.cursorrules`
     - [ ] Ensure destination pattern usage
     - [ ] Fix loop syntax (`for i := range n`)
     - [ ] Add proper error wrapping
   - [ ] **Task**: Read Python reference implementations
     - [ ] Document algorithm details from Python code
     - [ ] Identify required gocv functions
     - [ ] Plan Go implementation structure

2. **Phase 1 - Feature Extraction and Matching** (Priority: HIGH)
   - [ ] **Task**: Implement feature detectors
     - [ ] ORB detector wrapper
     - [ ] FAST detector wrapper
     - [ ] GFTT detector wrapper
     - [ ] Unified detector interface
   - [ ] **Task**: Implement feature descriptors
     - [ ] ORB descriptor wrapper
     - [ ] Unified descriptor interface
   - [ ] **Task**: Implement feature matchers
     - [ ] Brute-force matcher (binary and float)
     - [ ] FLANN matcher
     - [ ] Lowe's ratio test
     - [ ] Unified matcher interface
   - [ ] **Task**: Implement feature tracking
     - [ ] Lucas-Kanade optical flow
     - [ ] Track management
     - [ ] Outlier rejection

3. **Phase 2 - Stereo Processing** (Priority: HIGH)
   - [ ] **Task**: Implement stereo rectification
     - [ ] Rectification using calibration matrices
     - [ ] Rectification map computation
     - [ ] Left/right image rectification
   - [ ] **Task**: Implement stereo matching
     - [ ] SGBM wrapper
     - [ ] Disparity computation
     - [ ] Disparity filtering
   - [ ] **Task**: Implement 3D reconstruction
     - [ ] Triangulation from stereo pair
     - [ ] Depth map generation
     - [ ] Point cloud creation

4. **Phase 3 - Monocular Visual Odometry** (Priority: HIGH)
   - [ ] **Task**: Essential matrix estimation
     - [ ] 8-point algorithm
     - [ ] 5-point algorithm (more accurate, but more complex)
     - [ ] RANSAC for robustness
   - [ ] **Task**: Pose recovery
     - [ ] Extract rotation and translation from essential matrix
     - [ ] Disambiguate 4 solutions using cheirality
   - [ ] **Task**: Reprojection error
     - [ ] Calculate reprojection error
     - [ ] Use for pose validation and refinement
   - [ ] **Task**: Monocular odometry engine
     - [ ] Frame processing pipeline
     - [ ] Pose integration
     - [ ] Trajectory tracking

5. **Phase 4 - Stereo Visual Odometry** (Priority: HIGH)
   - [ ] **Task**: SolvePnP implementation
     - [ ] PnP solver wrapper (gocv)
     - [ ] RANSAC for robust estimation
   - [ ] **Task**: Stereo odometry engine
     - [ ] 3D point generation from stereo
     - [ ] 3D point matching across frames
     - [ ] Pose estimation from 3D-3D correspondences
     - [ ] Pose integration

6. **Phase 5 - Bag of Words** (Priority: MEDIUM)
   - [ ] **Task**: Vocabulary builder
     - [ ] Feature collection from images
     - [ ] k-means clustering (or hierarchical k-means)
     - [ ] Vocabulary serialization
   - [ ] **Task**: BOW descriptor generation
     - [ ] Feature-to-word assignment
     - [ ] TF-IDF weighting
     - [ ] Histogram generation
   - [ ] **Task**: BOW matching
     - [ ] Database for place descriptors
     - [ ] Similarity computation
     - [ ] Top-K retrieval

7. **Phase 6 - Place Recognition** (Priority: MEDIUM)
   - [ ] **Task**: Place database
     - [ ] Place storage structure
     - [ ] Place indexing
     - [ ] Place matching
   - [ ] **Task**: Pose recognition
     - [ ] Current-to-target view matching
     - [ ] Essential matrix estimation
     - [ ] Relative pose calculation

8. **Phase 7 - Graph-Based Navigation** (Priority: MEDIUM)
   - [ ] **Task**: Topological map using `x/math/graph`
     - [ ] Node structure (pose + BOW + links)
     - [ ] Graph construction
     - [ ] Integration with place recognition
   - [ ] **Task**: Path planning
     - [ ] A* pathfinding (use `x/math/graph/astar`)
     - [ ] Dijkstra pathfinding (use `x/math/graph/dijkstra`)
     - [ ] Path representation
   - [ ] **Task**: Navigation execution
     - [ ] Localization in graph
     - [ ] Path following
     - [ ] Visual servoing

9. **Phase 8 - Command Programs** (Priority: HIGH)
   - [ ] **Task**: Create `cmd/odometry`
     - [ ] Command structure (Cobra)
     - [ ] Subcommands: `mono`, `stereo`, `3d2d`
     - [ ] Source integration (`cmd/display/source`)
     - [ ] Display integration (`cmd/display/destination`)
     - [ ] Calibration loading
     - [ ] Pose output (file formats)
     - [ ] Visualization

10. **Phase 9 - Neural Network Integration** (Priority: LOW - Future work)
    - [ ] **Task**: Pre-trained model support
      - [ ] Model loading (ONNX, TensorFlow Lite)
      - [ ] Inference engine integration
      - [ ] Descriptor extraction
    - [ ] **Task**: Place recognition with neural networks
      - [ ] NetVLAD integration
      - [ ] SuperPoint integration
      - [ ] Hybrid BOW + neural network approach

---

## Progress Summary

**Overall Progress**: **35%** (Existing `x/vision` structure ✅, Folder structure organized ✅: `detect/`, `extract/`, `match/`, `transform/` created, Basic processors implemented ✅, Transform processors (rectify, undistort, affine, crop, histogram, background) ✅, Extract processors (features, blobs) ✅, Format refactored to chain.Processor ✅, Camera types (Camera, StereoCamera) ✅, SPEC.md updated ✅, `cmd/calibrate` integrated with new processors ✅, Code cleanup completed ✅, Python reference code available ✅)

**Next Milestone**: Phase 0 - Foundation (Code cleanup and reference review) - ⚠️ **PENDING**

**Blockers**: None currently identified

**Known Issues & Integration Tasks**:
1. ✅ **`x/vision/transform/format` refactored**: Now uses `chain.Processor` pattern ✅
   - [x] MatToImageProcessor: `chain.Processor[ImageWithTensor, types.Tensor]` ✅
   - [x] ImageToMatProcessor: `chain.Processor[types.Tensor, ImageWithTensor]` ✅
   - [x] DataToStereoProcessor: `chain.Processor[StereoPair, types.Tensor]` ✅
   - [x] Old pipeline pattern code kept for backward compatibility ✅
2. ✅ **`x/vision/transform/color` refactored**: Now uses `chain.Processor` pattern ✅
   - [x] Processor: `chain.Processor[types.Tensor, types.Tensor]` for color space conversion ✅
   - [x] Supports all gocv color conversion codes (BGR to RGB, BGR to grayscale, etc.) ✅
   - [x] Old pipeline pattern code kept for backward compatibility ✅
2. ✅ **`cmd/calibrate` integration**: ✅
   - [x] Updated to use `x/vision/transform/rectify.Rectify()` helper function ✅
   - [x] Updated to use `x/vision/transform/undistort.Undistort()` helper function ✅
   - [x] Uses `x/vision/detect/corners` ✅
3. ✅ **Build verification**: All packages build correctly ✅
4. ✅ **Test files created**: Comprehensive tests for detect processors ✅
   - [x] Corner detection tests (`detect/corners/processor_test.go`) ✅
   - [x] Feature detection tests (`detect/features/processor_test.go`) ✅
   - [x] Feature extraction tests (`extract/features/processor_test.go`) ✅
   - **NOTE**: Tests use `buffer.Ring` (not `buffer.Linear`) since only Ring implements ViewReader/ViewWriter
   - **NOTE**: `gocv.DrawKeyPoints()` requires `color.RGBA` type (from `image/color` package) for color parameter, NOT `gocv.Scalar` or `image.RGBA`

**Dependencies Resolved**: 
- `x/math/graph` available ✅
- `x/math/vec` and `x/math/mat` available ✅
- `x/marshaller/gocv` available ✅
- `cmd/display/source` and `cmd/display/destination` available ✅
- `cmd/calibrate` exists ✅

---

## Algorithm Categories

### 1. Extraction
- **Feature Detection**: ORB, SIFT, FAST, GFTT, KAZE, AKAZE, BRISK
- **Feature Description**: Binary descriptors (ORB, BRISK), Float descriptors (SIFT, SURF, KAZE)
- **BOW Descriptors**: Visual word histograms (TF-IDF weighted)

### 2. Matching
- **Feature Matching**: Brute-force, FLANN, Lowe's ratio test
- **Feature Tracking**: Lucas-Kanade optical flow
- **Stereo Matching**: SGBM, block matching
- **Place Matching**: BOW similarity, spatial consistency

### 3. Odometry
- **Monocular**: Essential matrix → pose recovery
- **Stereo**: 3D triangulation → SolvePnP
- **3D-2D**: Previous 3D points → current 2D features → SolvePnP

### 4. Place Recognition
- **BOW-Based**: Vocabulary → descriptors → matching
- **Neural Network-Based**: Pre-trained models (future)

### 5. Navigation
- **Graph Construction**: Topological map with `x/math/graph`
- **Path Planning**: A*, Dijkstra using `x/math/graph`
- **Pose Recognition**: Current view → target view → required pose

---

## Potential Issues (IMMUTABLE)

### Technical Risks

1. **Feature Matching Performance**:
   - **Risk**: Real-time matching may be slow with large feature sets
   - **Mitigation**: Use FLANN for large sets, optimize hot paths, consider GPU acceleration

2. **Stereo Matching Accuracy**:
   - **Risk**: Disparity computation may be noisy
   - **Mitigation**: Implement filtering, validate with ground truth datasets

3. **Monocular Scale Ambiguity**:
   - **Risk**: Monocular odometry has scale ambiguity
   - **Mitigation**: Document scale estimation requirements, provide initialization options

4. **BOW Vocabulary Quality**:
   - **Risk**: Vocabulary may not generalize well
   - **Mitigation**: Provide vocabulary training tools, allow custom vocabularies

5. **Graph Navigation Complexity**:
   - **Risk**: Large graphs may slow pathfinding
   - **Mitigation**: Use efficient algorithms from `x/math/graph`, consider graph pruning

### Code Quality Risks

1. **Type Safety**:
   - **Risk**: Incorrect type casting may cause runtime panics
   - **Mitigation**: Write comprehensive unit tests, use type assertions carefully

2. **Performance in Hot Paths**:
   - **Risk**: Feature extraction/matching called every frame
   - **Mitigation**: Preallocate buffers, use destination pattern, profile early

3. **Integration Complexity**:
   - **Risk**: Multiple components (extraction, matching, odometry) need to work together
   - **Mitigation**: Clear interfaces, integration tests, incremental development

---

## Known Issues

1. **Outdated Code**: Existing `x/vision` code may not follow current Go patterns
2. **Missing Documentation**: SPEC.md is outdated and needs revision
3. **No Tests**: Missing comprehensive test coverage
4. **Incomplete Implementation**: Many features from Python version not yet implemented

---

## Reference Materials

### Python Implementation
- **Location**: `d:\Projects\python\EasyVision\EasyVision`
- **Key Files**:
  - `engine/visualodometry_*.py` - Odometry implementations
  - `engine/bowvocabulary.py` - Bag of Words
  - `engine/topologicalmap.py` - Graph-based navigation
  - `processors/featureextractor.py` - Feature extraction
  - `bin/performance_visualodometry_*.py` - Command programs

### Datasets
- **KITTI**: Visual odometry dataset (if available)
- **EuRoC**: Visual-inertial dataset (if available)

### External Libraries
- **DBoW3**: C++ library for Bag of Words (Python binding: pyDBoW3)
- **OpenCV**: Computer vision algorithms (via gocv)

---

## SPEC/PLAN Reading Guide

### Before Implementing Features
- ✅ Read `x/vision/SPEC.md` (note: outdated, needs update)
- [ ] Read Python `processors/featureextractor.py`
- [ ] Read `x/math/vec/vec.go`
- [ ] Read `x/math/mat/mat.go`
- [ ] Read `x/marshaller/gocv/SPEC.md`

### Before Implementing Odometry
- [ ] Read Python `engine/visualodometry_*.py`
- [ ] Read `x/math/mat/mat.go` (for matrix operations)
- [ ] Read OpenCV documentation for essential matrix, SolvePnP

### Before Implementing BOW
- [ ] Read Python `engine/bowvocabulary.py`
- [ ] Read DBoW3 documentation (if using)
- [ ] Understand k-means clustering

### Before Implementing Navigation
- [ ] Read `x/math/graph/SPEC.md`
- [ ] Read Python `engine/topologicalmap.py`
- [ ] Read `x/math/graph/astar.go` and `x/math/graph/dijkstra.go`

### Before Creating Commands
- [ ] Read `cmd/calibrate` structure (Cobra integration)
- [ ] Read `cmd/display/source` and `cmd/display/destination`
- [ ] Read `cmd/spectrometer/main.go` (example Cobra command structure)

