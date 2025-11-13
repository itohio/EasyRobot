# Math Library Specification

## Overview

The math library provides comprehensive mathematical primitives optimized for robotics applications, with emphasis on embedded systems compatibility and performance. The library includes working implementations of core mathematical operations (vectors, matrices, tensors, neural networks) as well as draft implementations of advanced algorithms for control, filtering, path planning, and learning.

**Status Note**: Core components (Vector, Matrix, Tensor, Neural Networks) are working and tested. Most advanced algorithms (control, filters, graph algorithms, interpolation) are in draft state and have not been tested with real hardware (TODO).

## Components

### 1. Vector Operations (`x/math/vec`)

**Purpose**: Vector mathematics for 2D, 3D, 4D, and arbitrary dimensions

**Status**: ✅ **Working** - Fully implemented and tested

**Types**:
- `vec.Vector`: Generic vector (slice of float32)
- `vec.Vec2D`: 2D vector
- `vec.Vec3D`: 3D vector
- `vec.Vec4D`: 4D vector
- `vec.Quaternion`: Quaternion representation

**Operations**:
- Arithmetic: Add, Sub, Mul, Div, MulC (scalar multiplication), DivC (scalar division)
- Geometric: Dot, Cross, Length, Normalize, Distance
- Transformations: Rotate, Translate, Scale
- Quaternion: Rotation conversions, SLERP

**Characteristics**:
- In-place operations (minimize allocations)
- `float32` precision (embedded-friendly)
- Zero-copy where possible
- Uses `math/primitive/fp32` for optimized operations
- Generated code from templates for specific-sized implementations

### 2. Matrix Operations (`x/math/mat`)

**Purpose**: Matrix mathematics for transformations and linear algebra

**Status**: ✅ **Working** - Fully implemented and tested

**Types**:
- `mat.Matrix`: Generic matrix (2D slice of float32)
- `mat.Mat2x2`: 2x2 matrix
- `mat.Mat3x3`: 3x3 matrix
- `mat.Mat4x4`: 4x4 matrix
- `mat.Mat3x4`: 3x4 matrix
- `mat.Mat4x3`: 4x3 matrix

**Operations**:
- Arithmetic: Add, Sub, Mul, MulC, DivC
- Transformations: Transpose, Inverse, SVD, QR decomposition, Cholesky decomposition, Pseudo-inverse
- Geometric: Rotation (X, Y, Z, 2D), Orientation (quaternion), Homogeneous transformations
- Linear Algebra: LU decomposition, Determinant
- Conversions: Quaternion to matrix, matrix to quaternion

**Characteristics**:
- In-place operations where possible
- `float32` precision
- Destination-based API (mutate destination matrix)
- Row-major storage layout
- Uses `math/primitive/fp32` for optimized operations
- Generated code from templates for specific-sized implementations

### 3. Tensor Operations (`x/math/tensor`)

**Purpose**: Multi-dimensional tensor operations for numerical computing, computer vision, and machine learning

**Status**: ✅ **Working** - Fully implemented and tested

**Key Features**:
- Interface-based design (`types.Tensor` interface, `eager_tensor.Tensor` implementation)
- Row-major storage layout matching Go nested arrays
- Zero allocations in hot paths using `math/primitive/fp32`
- Support for multiple data types: FP32, FP64, INT32, INT64, INT, INT16, INT8
- 80+ operations including element-wise, linear algebra, convolutions, pooling, activations

**Operations**:
- Core: Data access, shape operations, cloning, reshaping, slicing
- Element-wise: Add, Sub, Mul, Div, Scale, Fill, Square, Sqrt, Exp, Log, Pow, Abs, Sign, Cos, Sin
- Linear Algebra: MatMul (2D and batched), Transpose, Dot, Norm, Normalize
- Convolutions: Conv1D, Conv2D, Conv3D, DepthwiseConv2D, GroupConv2D, DilatedConv2D
- Pooling: MaxPool2D, AvgPool2D, GlobalAvgPool2D, AdaptiveAvgPool2D
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Reductions: Sum, Mean, Max, Min, ArgMax
- Broadcasting: BroadcastTo
- Gradient operations: ScatterAdd, Unpad, Im2Col, Col2Im

**See**: `x/math/tensor/SPEC.md` for comprehensive documentation

### 4. Neural Networks (`x/math/nn`)

**Purpose**: High-level neural network framework with TensorFlow Lite compatibility

**Status**: ✅ **Working** - Fully implemented and tested

**Components**:
- **Layer Interface**: Abstract layer operations (Forward, Backward, Init)
- **Model**: Sequential model builder with layer management
- **Loss Functions**: MSE, CrossEntropy, CategoricalCrossEntropy
- **Layers**: Dense, Conv1D, Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D, Flatten, Reshape, Unsqueeze, Squeeze, Transpose, Pad, Concatenate, ReLU, Sigmoid, Tanh, Softmax, Dropout

**Features**:
- Layers store input/output state internally
- Backward pass uses stored state (no need to pass input/output)
- `CanLearn()` controls gradient computation (default: false for inference-only)
- Xavier initialization support
- Shape validation during construction and forward pass

**See**: `x/math/nn/SPEC.md` for detailed documentation

### 5. Learning/Training (`x/math/learn`)

**Purpose**: Training utilities, optimizers, and quantization for neural networks

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

**Components**:
- **Training Loop**: `TrainStep` function for single training step (forward, loss, backward, update)
- **Optimizers**: SGD, Adam
- **Quantization**: Post-training quantization for model compression (symmetric, asymmetric, per-tensor, per-channel)

**See**: `x/math/learn/SPEC.md` for detailed documentation

### 6. Control Algorithms (`x/math/control`)

**Purpose**: Robot control algorithms including kinematics and motion planning

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

#### 6.1 Kinematics (`x/math/control/kinematics`)

**Types**:
- **Joints**: Planar (2DOF, 3DOF), Denavit-Hartenberg (DH parameters)
- **Wheels**: Differential drive, Mecanum, Steer4, Steer4Dual, Steer6
- **Rigid Body**: 6DOF rigid body kinematics
- **Thrusters**: Thruster array kinematics (forward/inverse)

**Interfaces**:
- `ForwardKinematics`: Map actuator state to end-effector/chassis state
- `BackwardKinematics`: Solve for actuator commands from desired end-effector/chassis state
- `Bidirectional`: Both forward and backward kinematics

**Key Features**:
- Matrix-based API for consistent interface
- State/control dimension management
- Constraint handling (velocity, acceleration limits)
- Support for various robot configurations

**See**: `x/math/control/kinematics/SPEC.md` for detailed documentation

#### 6.2 Motion Planning (`x/math/control/motion`)

**Types**:
- **Rigid Body Planner**: Path following with velocity/acceleration/jerk limits, curvature constraints
- **VAJ1D**: Velocity-Acceleration-Jerk 1D filter for smooth motion profiles
- **Gait Planner**: Legged robot gait planning (support phases, swing trajectories)

**Key Features**:
- Path following with lookahead
- PID controllers for heading/speed/lateral control
- Curvature-based velocity limiting
- Yaw alignment with path tangent
- Support endpoint planning for legged robots

**See**: `x/math/control/motion/rigidbody/SPEC.md` and related DESIGN.md files

#### 6.3 PID Controller (`x/math/control/pid`)

**Purpose**: Multi-dimensional PID controller

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

**Features**:
- Vector-based (multi-dimensional)
- Configurable gains (P, I, D)
- Output clamping (min, max)
- Integral term management

### 7. Filters (`x/math/filter`)

**Purpose**: Control theory algorithms and sensor fusion

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

**Base Interface**:
```go
type Filter interface {
    Reset() Filter
    Update(timestep float32) Filter
    GetInput() vec.Vector
    GetOutput() vec.Vector
    GetTarget() vec.Vector
}
```

#### 7.1 Kalman Filter (`x/math/filter/kalman`)

**Purpose**: Linear Kalman filter for state estimation

**Features**:
- State prediction and update
- Process and measurement noise models
- Covariance propagation

#### 7.2 Extended Kalman Filter (`x/math/filter/ekalman`)

**Purpose**: Extended Kalman filter for non-linear systems

**Features**:
- Non-linear state and measurement models
- Jacobian computation
- First-order linearization

#### 7.3 AHRS (`x/math/filter/ahrs`)

**Purpose**: Attitude and Heading Reference System

**Algorithms**:
- **Madgwick**: Madgwick filter for sensor fusion
- **Mahony**: Mahony filter for sensor fusion

**Inputs**:
- Acceleration (3-axis)
- Gyroscope (3-axis)
- Magnetometer (3-axis)

**Output**:
- Orientation (quaternion or euler angles)

#### 7.4 SLAM (`x/math/filter/slam`)

**Purpose**: Simultaneous Localization and Mapping

**Status**: ⚠️ **Early Draft** - Basic implementation

**Features**:
- Occupancy grid mapping
- Pose estimation integration
- Grid-based SLAM algorithms

### 8. Interpolation (`x/math/interpolation`)

**Purpose**: Interpolation and extrapolation algorithms

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

**Types**:
- `lerp.Lerp`: Linear interpolation
- `cosine.Cosine`: Cosine interpolation
- `bezier.Bezier`: Bezier curve interpolation
- `spline.Spline`: Spline interpolation
- `kriging.Kriging`: Kriging interpolation (spatial interpolation)
- `rbf.RBF`: Radial Basis Function interpolation
- **Upsampling**: Vector and matrix upsampling (linear, cubic, bicubic)

**Operations**:
- 1D interpolation: Value at parameter t
- Multi-dimensional: Vector/matrix interpolation
- Extrapolation: Value beyond range
- Matrix upsampling: 2D image/data grid upsampling

**See**: `x/math/interpolation/UPSAMPLING_SPEC.md` for upsampling details

### 9. Graph Algorithms (`x/math/graph`)

**Purpose**: Graph algorithms for path planning and decision making

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

**Algorithms**:
- **A\***: A* pathfinding algorithm
- **BFS**: Breadth-first search
- **DFS**: Depth-first search
- **Dijkstra**: Dijkstra's shortest path algorithm
- **Cycle Detection**: Graph cycle detection
- **Decision Trees**: Decision tree construction and traversal
- **KD-Trees**: K-dimensional tree for spatial indexing
- **Binary Trees**: Binary tree data structures

**Graph Types**:
- `GenericGraph`: Generic graph implementation with adjacency lists
- `GridGraph`: Graph for 2D grids
- `MatrixGraph`: Graph represented as adjacency matrix
- `TreeGraph`: Tree data structure
- `ForestGraph`: Forest (collection of trees)
- `BinaryTreeGraph`: Binary tree graph

**Use Cases**:
- Path planning
- Spatial indexing
- Decision making
- Graph traversal

### 10. Grid Operations (`x/math/grid`)

**Purpose**: 2D grid and occupancy grid operations

**Status**: ⚠️ **Draft** - Implemented but not tested with real hardware

**Operations**:
- **Ray Casting**: Single and batch ray casting with Bresenham algorithm
- **Path Planning**: A*, Fast A*, Dijkstra, Fast Dijkstra
- **Shape Extraction**: Rectangle, circle, ellipse extraction from grids
- **Matrix Operations**: Masking, extraction utilities
- **Drawing**: Grid drawing utilities

**Use Cases**:
- SLAM algorithms
- Path planning
- Sensor fusion
- Map processing
- Occupancy grid operations

**See**: `x/math/grid/README.md` and `x/math/grid/SPEC.md` for detailed documentation

### 11. Primitives (`x/math/primitive`)

**Purpose**: Low-level optimized mathematical primitives

**Status**: ✅ **Working** - Core foundation for higher-level operations

**Types**:
- **FP32**: BLAS levels 1-3 operations with zero allocations
- **Generics**: Generic implementations for multiple data types (ST/MT variants)
- **Integer Types**: Qu8, Qi8, Qi16, Qi32, Qi64 for quantized operations

**Features**:
- Stride-based access for non-contiguous data
- Row-major storage layout
- Zero allocations in hot paths
- Architecture-specific optimizations potential (SIMD, NEON)

## Design Principles

### In-Place Operations

Most operations mutate the destination to minimize allocations:
```go
vec.Vector(v).Add(v2)  // Mutates v
```

### Destination-Based API

Matrix operations require destination matrix:
```go
dst.Mul(a, b)  // dst = a * b
```

### Float32 Precision

All operations use `float32` for embedded compatibility:
- Reduced memory usage
- Faster on embedded systems
- Potential precision issues (acceptable for robotics)

### Zero Allocations

Critical paths use `primitive` package for allocation-free operations:
- Element-wise operations when tensors are contiguous
- Matrix operations via `primitive.Gemm_*`
- Convolution operations via `primitive.Conv2D`

### Interface-Based Design

Higher-level components use interfaces for extensibility:
- `types.Tensor` interface for tensor operations
- `nn.Layer` interface for neural network layers
- `filter.Filter` interface for filter algorithms
- `graph.Graph` interface for graph algorithms

### Row-Major Storage

All matrices and tensors use row-major storage layout:
- Matches Go nested arrays layout: `[][]float32`
- Consistent with `primitive` package conventions
- Compatible with BLAS/LAPACK operations

## Code Generation

### Auto-Generated Code

Matrix and vector operations are auto-generated from templates:
- Templates: `gen/*.tpl`
- Parameters: `gen/*.json`
- Generated: Specific-sized implementations (2x2, 3x3, etc.)

**Development Cycle**:
1. Implement generic version
2. Test generic version
3. Generate specific-sized versions
4. Optimize specific-sized versions if needed

## Platform Support

### Go (Standard)

- Full feature set
- All optimizations available
- Development and testing

### TinyGo

- Subset of features
- Limited stdlib
- Embedded-friendly

### Architecture-Specific Optimizations

Potential optimizations for:
- **Intel/AMD (x86_64)**: SIMD support potential
- **ARM (AArch64, ARMv7)**: NEON support potential
- **Xtensa**: Native operations only
- **RP2040**: Native operations only
- **WASM**: Limited optimizations

## Implementation Status Summary

### ✅ Working and Tested

1. **Vector Operations** (`x/math/vec`): Fully implemented, tested
2. **Matrix Operations** (`x/math/mat`): Fully implemented, tested (includes inverse, SVD, QR, Cholesky)
3. **Tensor Operations** (`x/math/tensor`): Fully implemented, tested (80+ operations)
4. **Neural Networks** (`x/math/nn`): Fully implemented, tested (layers, models, losses)
5. **Primitives** (`x/math/primitive`): Core foundation, working

### ⚠️ Draft - Not Tested with Real Hardware

1. **Control Algorithms** (`x/math/control`): Implemented but not hardware-tested
   - Kinematics (joints, wheels, rigid body, thrusters)
   - Motion planning (rigid body planner, gait planner)
   - PID controller

2. **Filters** (`x/math/filter`): Implemented but not hardware-tested
   - Kalman filter
   - Extended Kalman filter
   - AHRS (Madgwick, Mahony)
   - SLAM (early draft)

3. **Learning/Training** (`x/math/learn`): Implemented but not hardware-tested
   - Optimizers (SGD, Adam)
   - Training loops
   - Quantization

4. **Interpolation** (`x/math/interpolation`): Implemented but not hardware-tested
   - Linear, cosine, bezier, spline
   - Kriging, RBF
   - Upsampling (vector, matrix)

5. **Graph Algorithms** (`x/math/graph`): Implemented but not hardware-tested
   - A*, BFS, DFS, Dijkstra
   - Decision trees, KD-trees
   - Cycle detection

6. **Grid Operations** (`x/math/grid`): Implemented but not hardware-tested
   - Ray casting
   - Path planning (A*, Dijkstra)
   - Shape extraction

## Testing

### Current State

- ✅ Core components (Vector, Matrix, Tensor, NN) have comprehensive unit tests
- ✅ Integration tests for neural network training (MNIST, XOR examples)
- ⚠️ Limited integration tests for control/filter algorithms
- ⚠️ No real hardware testing for advanced algorithms

### Test Coverage

**Working Components**:
- Vector: Unit tests for all operations
- Matrix: Unit tests for arithmetic, transformations, decompositions
- Tensor: Comprehensive test suite (80+ operations)
- Neural Networks: Unit tests for layers, integration tests for models

**Draft Components**:
- Basic unit tests exist
- Integration tests needed
- Real hardware validation needed (TODO)

## Known Issues and Limitations

### Precision

- `float32` precision issues in some operations (acceptable for robotics)
- Precision loss in iterative algorithms (inverse, SVD)

### Performance

- Missing architecture-specific optimizations (SIMD, NEON)
- Strided tensor operations may be slower than contiguous

### Features

1. **Control**: Not tested with real hardware (TODO)
2. **Filters**: Not tested with real sensors (TODO)
3. **Graph**: Basic implementations, may need optimization for large graphs
4. **Interpolation**: Implemented but not validated on real data (TODO)
5. **Grid**: Basic operations, may need optimization for large grids

### Documentation

- Core components: Well documented
- Advanced algorithms: Some documentation exists, may need expansion
- Usage examples: Limited for advanced algorithms

## Potential Improvements

### Short Term

1. **Testing**: Comprehensive integration tests for control/filter algorithms
2. **Hardware Validation**: Test control and filter algorithms with real hardware (TODO)
3. **Documentation**: Expand usage examples for advanced algorithms
4. **Performance**: Optimize strided operations

### Long Term

1. **SIMD**: Architecture-specific SIMD optimizations (x86_64, ARM)
2. **GPU**: GPU acceleration support (if applicable)
3. **Fixed-Point**: Fixed-point arithmetic for ultra-low-power embedded
4. **Advanced Features**: Complete missing features in draft components
5. **Code Generation**: Enhanced code generation for optimizations
6. **Testing**: Real hardware testing for all algorithms (TODO)

## Dependencies

- `github.com/chewxy/math32`: Float32 math operations
- Standard library: `math`, `unsafe`, `sync`

## Package Structure

```
x/math/
├── vec/              # Vector operations (✅ Working)
├── mat/              # Matrix operations (✅ Working)
├── tensor/           # Tensor operations (✅ Working)
├── nn/               # Neural networks (✅ Working)
├── learn/            # Training utilities (⚠️ Draft)
├── control/          # Control algorithms (⚠️ Draft)
│   ├── kinematics/   # Forward/backward kinematics
│   ├── motion/       # Motion planning
│   └── pid/          # PID controller
├── filter/           # Filters (⚠️ Draft)
│   ├── kalman/       # Kalman filter
│   ├── ekalman/      # Extended Kalman filter
│   ├── ahrs/         # AHRS filters
│   └── slam/         # SLAM algorithms
├── interpolation/    # Interpolation algorithms (⚠️ Draft)
├── graph/            # Graph algorithms (⚠️ Draft)
├── grid/             # Grid operations (⚠️ Draft)
└── primitive/        # Low-level primitives (✅ Working)
```

## Notes

- **Working Components**: Vector, Matrix, Tensor, and Neural Networks are production-ready and tested
- **Draft Components**: Most advanced algorithms are implemented but require hardware validation (TODO)
- **Performance**: Core components are optimized; advanced algorithms may need optimization
- **Testing**: Core components have comprehensive tests; advanced algorithms need more testing
- **Documentation**: Core components are well-documented; advanced algorithms have basic documentation
