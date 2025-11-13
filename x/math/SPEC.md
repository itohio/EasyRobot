# Math Library Specification

## Overview

The math library provides mathematical primitives optimized for robotics applications, with emphasis on embedded systems compatibility and performance.

## Components

### 1. Vector Operations (`pkg/core/math/vec`)

**Purpose**: Vector mathematics for 2D, 3D, 4D, and arbitrary dimensions

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

**Questions**:
1. Should vectors support SIMD optimization?
2. How to handle vector errors (NaN, Inf)?
3. Should vectors support fixed-point arithmetic?
4. How to optimize for different architectures (ARM, Xtensa)?
5. Should vectors support GPU acceleration?
6. How to handle precision issues?
7. Should vectors support complex numbers?

### 2. Matrix Operations (`pkg/core/math/mat`)

**Purpose**: Matrix mathematics for transformations and linear algebra

**Types**:
- `mat.Matrix`: Generic matrix (2D slice of float32)
- `mat.Mat2x2`: 2x2 matrix
- `mat.Mat3x3`: 3x3 matrix
- `mat.Mat4x4`: 4x4 matrix
- `mat.Mat3x4`: 3x4 matrix
- `mat.Mat4x3`: 4x3 matrix
- `mat.Sparse`: Sparse matrix representation

**Operations**:
- Arithmetic: Add, Sub, Mul, MulC, DivC
- Transformations: Transpose, Inverse (planned), SVD (planned)
- Geometric: Rotation (X, Y, Z, 2D), Orientation (quaternion)
- Linear Algebra: LU decomposition, Determinant
- Conversions: Quaternion to matrix, matrix to quaternion

**Characteristics**:
- In-place operations where possible
- `float32` precision
- Destination-based API (mutate destination matrix)

**Questions**:
1. Should matrices support SIMD optimization?
2. How to handle matrix errors (singular, ill-conditioned)?
3. Should matrices support sparse representation optimization?
4. How to optimize for different architectures?
5. Should matrices support GPU acceleration?
6. How to handle precision issues in inverse/SVD?
7. Should matrices support block algorithms for large matrices?
8. How to implement efficient matrix multiplication?

### 3. Interpolation (`pkg/core/math/interpolation`)

**Purpose**: Interpolation and extrapolation algorithms

**Types**:
- `lerp.Lerp`: Linear interpolation
- `cosine.Cosine`: Cosine interpolation
- `bezier.Bezier`: Bezier curve interpolation
- `spline.Spline`: Spline interpolation

**Operations**:
- 1D interpolation: Value at parameter t
- Multi-dimensional: Vector/matrix interpolation
- Extrapolation: Value beyond range

**Questions**:
1. Should interpolation support different curve types?
2. How to handle extrapolation beyond range?
3. Should interpolation support time-based interpolation?
4. How to optimize interpolation for real-time?
5. Should interpolation support constraint-based interpolation?
6. How to handle interpolation errors?

### 4. Filters (`pkg/core/math/filter`)

**Purpose**: Control theory algorithms and sensor fusion

#### PID Controller (`pkg/core/math/filter/pid`)

**Purpose**: Proportional-Integral-Derivative controller

**Characteristics**:
- Multi-dimensional (vector-based)
- Configurable gains (P, I, D)
- Output clamping (min, max)
- Integral term management

**Questions**:
1. Should PID support anti-windup mechanisms?
2. How to handle derivative kick?
3. Should PID support different derivative modes?
4. How to optimize for real-time constraints?
5. Should PID support adaptive gains?
6. How to handle PID tuning?

#### AHRS (`pkg/core/math/filter/ahrs`)

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

**Questions**:
1. Should AHRS support different sensor configurations?
2. How to handle sensor calibration?
3. Should AHRS support magnetic declination correction?
4. How to optimize for embedded systems?
5. Should AHRS support different update rates?
6. How to handle sensor errors (outliers, noise)?

#### Kinematics Filter (`pkg/core/math/filter/kinematics`)

**Purpose**: Velocity and Acceleration Jerk filter

**Characteristics**:
- 1D filter for motion profiles
- Smooth acceleration/deceleration

**Questions**:
1. Should kinematics filter support multi-dimensional?
2. How to handle constraint violations?
3. Should kinematics filter support different profiles?
4. How to optimize for real-time?

### 5. Tensor Operations (`pkg/core/math/tensor`)

**Purpose**: Multi-dimensional array operations

**Types**:
- `tensor.Tensor`: Dense tensor representation

**Status**: Planned/partial implementation

**Questions**:
1. Should tensors support different data types (int, float)?
2. How to handle tensor operations (convolution, pooling)?
3. Should tensors support GPU acceleration?
4. How to optimize for embedded systems?
5. Should tensors support automatic differentiation?

## Design Principles

### In-Place Operations

Most operations mutate the destination to minimize allocations:
```go
vec.Vector(v).Add(v2)  // Mutates v
```

**Questions**:
1. Should we provide non-mutating versions?
2. How to document mutating vs non-mutating operations?
3. Should we support operation chaining?

### Destination-Based API

Matrix operations require destination matrix:
```go
dst.Mul(a, b)  // dst = a * b
```

**Questions**:
1. Should we provide convenience functions that allocate?
2. How to handle in-place operations?
3. Should we support operation chaining?

### Float32 Precision

All operations use `float32` for embedded compatibility:
- Reduced memory usage
- Faster on embedded systems
- Potential precision issues

**Questions**:
1. Should we support `float64` option?
2. How to handle precision issues?
3. Should we support fixed-point arithmetic?

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

**Questions**:
1. Should we support code generation for other operations?
2. How to handle code generation for different architectures?
3. Should we support code generation for SIMD?

## Platform Support

### Go (Standard)

- Full feature set
- All optimizations available
- Development and testing

### TinyGo

- Subset of features
- Limited stdlib
- Embedded-friendly

**Questions**:
1. How to handle TinyGo limitations?
2. Should we provide TinyGo-specific implementations?
3. How to test TinyGo compatibility?

### Architecture-Specific Optimizations

- **Intel/AMD (x86_64)**: SIMD support potential
- **ARM (AArch64, ARMv7)**: NEON support potential
- **Xtensa**: Native operations only
- **RP2040**: Native operations only
- **WASM**: Limited optimizations

**Questions**:
1. Should we provide architecture-specific optimizations?
2. How to handle missing optimizations?
3. Should we support runtime optimization selection?

## Testing

### Current State

- Some unit tests exist
- Limited integration tests
- No performance benchmarks

**Questions**:
1. How to test mathematical correctness?
2. How to test performance?
3. How to test platform compatibility?
4. Should we provide test fixtures?
5. How to test edge cases (NaN, Inf, overflow)?

## Known Issues

1. **Precision**: `float32` precision issues in some operations
2. **Performance**: Missing optimizations for specific architectures
3. **Testing**: Limited test coverage
4. **Documentation**: Incomplete API documentation
5. **Features**: Missing inverse, SVD, pseudo-inverse

## Potential Improvements

1. **SIMD**: Architecture-specific SIMD optimizations
2. **GPU**: GPU acceleration support
3. **Fixed-Point**: Fixed-point arithmetic for embedded
4. **Testing**: Comprehensive test suite with benchmarks
5. **Documentation**: Complete API documentation
6. **Features**: Complete missing features (inverse, SVD)
7. **Optimization**: Better algorithms for large matrices
8. **Code Generation**: Enhanced code generation for optimizations

