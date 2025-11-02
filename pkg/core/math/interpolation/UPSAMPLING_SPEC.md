# Upsampling and Matrix Interpolation Specification

## Overview

This document specifies the design and implementation plan for upsampling functions that convert vectors and matrices from smaller to larger sizes, and advanced matrix interpolation methods.

## Components

### 1. Vector Upsampling

**Purpose**: Upsample 1D vectors to larger sizes using various interpolation methods.

**Input**: Original vector of size N
**Output**: Upsampled vector of size M (where M > N)

**Methods**:

#### 1.1 Linear Upsampling
- **Algorithm**: Uses linear interpolation between adjacent points
- **Key Features**:
  - Simple and fast
  - Preserves endpoint values
  - Smooth transition between points
  - Uniform spacing of upsampled points

#### 1.2 Cubic Upsampling
- **Algorithm**: Uses cubic spline interpolation for smoother curves
- **Key Features**:
  - Better continuity (C2) compared to linear
  - Natural boundary conditions
  - Smoother results but more computationally expensive
  - Better for preserving smoothness of underlying data

**Edge Cases**:
- N = 1: Replicate single value
- M = N: Return copy
- M < N: Return error or downsample

### 2. Matrix Upsampling

**Purpose**: Upsample 2D matrices (images/data grids) to larger sizes.

#### 2.1 Linear Matrix Upsampling
- **Algorithm**: Separate interpolation in each dimension
- **Approach**: 
  - First upsample rows, then upsample columns (or vice versa)
  - Uses separable filter design
- **Complexity**: O(rows * cols * (newRows * newCols) / (rows * cols))

#### 2.2 Bicubic Matrix Upsampling
- **Algorithm**: 2D cubic interpolation using 16 nearest neighbors
- **Approach**:
  - For each output pixel, use 4x4 neighborhood of input pixels
  - Apply cubic weights in both dimensions
- **Key Features**:
  - High quality for natural images
  - Better edge preservation than linear
  - C2 continuous
- **Cubic Kernel**: Uses cubic B-spline or Mitchell-Netravali filter

**Edge Cases**:
- Single row/column matrices
- UpSample to same size
- Boundary handling (reflect, wrap, zero-pad)

### 3. Advanced Matrix Interpolation

For sparse or irregularly sampled data.

#### 3.1 Kriging Interpolation
- **Algorithm**: Gaussian Process-based spatial interpolation
- **Purpose**: Best for data with spatial correlation
- **Key Components**:
  - Variogram model (exponential, spherical, Gaussian)
  - Covariance function
  - Kriging weights computation
  - Optional nugget effect for measurement errors

**Types**:
- **Ordinary Kriging**: Assumes unknown but constant mean
- **Simple Kriging**: Assumes known mean
- **Universal Kriging**: Polynomial trend in mean

**Complexity**: O(N³) due to covariance matrix inversion

#### 3.2 Radial Basis Function (RBF) Interpolation
- **Algorithm**: Weighted sum of radial basis functions
- **Purpose**: Interpolate scattered data points
- **Key Components**:
  - RBF kernel selection (Gaussian, Multiquadric, Inverse Multiquadric, Thin Plate Spline)
  - System solving for coefficients
  - Distance-based weighting

**Common RBF Kernels**:
- Gaussian: φ(r) = exp(-(εr)²), ε is shape parameter
- Multiquadric: φ(r) = √(1 + (εr)²)
- Inverse Multiquadric: φ(r) = 1/√(1 + (εr)²)
- Thin Plate Spline: φ(r) = r²log(r)

**Complexity**: O(N³) for coefficient computation, O(N) per evaluation

## API Design

### Vector Upsampling

```go
package interpolation

// LinearUpsample upsamples a vector from size N to size M using linear interpolation
// dst must be pre-allocated with size M
func LinearUpsample(src vec.Vector, dst vec.Vector) vec.Vector

// CubicUpsample upsamples a vector from size N to size M using cubic spline interpolation
// dst must be pre-allocated with size M
func CubicUpsample(src vec.Vector, dst vec.Vector) vec.Vector
```

### Matrix Upsampling

```go
package interpolation

// LinearMatrixUpsample upsamples a matrix using separable linear interpolation
// dst must be pre-allocated with target rows and cols
func LinearMatrixUpsample(src mat.Matrix, dst mat.Matrix) mat.Matrix

// BicubicMatrixUpsample upsamples a matrix using bicubic interpolation
// dst must be pre-allocated with target rows and cols
// Uses Mitchell-Netravali cubic kernel by default
func BicubicMatrixUpsample(src mat.Matrix, dst mat.Matrix) mat.Matrix

// Options for bicubic interpolation
type BicubicOptions struct {
    Boundary string // "reflect", "wrap", "zero", "clamp"
}
```

### Advanced Interpolation

```go
package interpolation

// Kriging interpolates values at target locations using spatial correlation
type Kriging struct {
    // ... implementation details
}

func NewKriging(variogram VariogramModel) *Kriging

// AddSample adds a known data point at location (x, y) with value v
func (k *Kriging) AddSample(x, y, v float32)

// Interpolate computes interpolated value at location (x, y)
func (k *Kriging) Interpolate(x, y float32) float32

// RBF interpolates values using radial basis functions
type RBF struct {
    // ... implementation details
}

type RBFKernel func(r float32) float32

func NewRBF(kernel RBFKernel, epsilon float32) *RBF

// AddSample adds a known data point at location (x, y) with value v
func (r *RBF) AddSample(x, y, v float32)

// Interpolate computes interpolated value at location (x, y)
func (r *RBF) Interpolate(x, y float32) float32

// Common RBF kernels
func GaussianKernel(epsilon float32) RBFKernel
func MultiquadricKernel(epsilon float32) RBFKernel
func InverseMultiquadricKernel(epsilon float32) RBFKernel
func ThinPlateSplineKernel() RBFKernel
```

## Implementation Details

### Vector Upsampling

**Linear**:
```python
# Pseudocode
def linear_upsample(src, dst):
    n = len(src)
    m = len(dst)
    
    if n == 1:
        dst.fill(src[0])
        return
    
    scale = (n - 1.0) / (m - 1.0)
    
    for i in range(m):
        pos = i * scale
        idx = floor(pos)
        frac = pos - idx
        
        if idx >= n - 1:
            dst[i] = src[n-1]
        else:
            dst[i] = lerp(src[idx], src[idx+1], frac)
```

**Cubic**:
```python
# Pseudocode
def cubic_upsample(src, dst):
    n = len(src)
    m = len(dst)
    
    # Compute cubic spline coefficients
    # Natural boundary conditions
    # For each output point, interpolate using cubic spline
```

### Matrix Upsampling

**Bicubic**:
```python
# Pseudocode for bicubic kernel (using Mitchell-Netravali)
def cubic_kernel(t):
    B = 1/3  # Mitchell-Netravali parameter
    C = 1/3
    
    t_abs = abs(t)
    t_abs2 = t_abs * t_abs
    t_abs3 = t_abs2 * t_abs
    
    if t_abs < 1:
        return ((12 - 9*B - 6*C) * t_abs3 + 
                (-18 + 12*B + 6*C) * t_abs2 + 
                (6 - 2*B)) / 6
    elif t_abs < 2:
        return ((-B - 6*C) * t_abs3 + 
                (6*B + 30*C) * t_abs2 + 
                (-12*B - 48*C) * t_abs + 
                (8*B + 24*C)) / 6
    else:
        return 0

def bicubic_upsample(src, dst):
    for y in range(dst_rows):
        for x in range(dst_cols):
            # Map to source coordinates
            sx = x * src_cols / dst_cols
            sy = y * src_rows / dst_rows
            
            # Get integer and fractional parts
            ix, fx = floor(sx), sx - floor(sx)
            iy, fy = floor(sy), sy - floor(sy)
            
            # Get 4x4 neighborhood (with boundary handling)
            # Compute weights
            value = 0
            for dy in -1..2:
                for dx in -1..2:
                    weight = cubic_kernel(dx - fx) * cubic_kernel(dy - fy)
                    value += weight * src[iy+dy, ix+dx]
            dst[y, x] = value
```

### Kriging Implementation

```python
# Simplified Kriging pseudocode
class Kriging:
    def __init__(self, variogram):
        self.samples = []
        self.variogram = variogram
    
    def add_sample(self, x, y, v):
        self.samples.append((x, y, v))
    
    def interpolate(self, x, y):
        n = len(self.samples)
        
        # Build covariance matrix
        C = matrix(n+1, n+1)
        for i in range(n):
            for j in range(n):
                dist = distance(self.samples[i], self.samples[j])
                C[i,j] = self.variogram(dist)
            C[i,n] = 1
            C[n,i] = 1
        C[n,n] = 0
        
        # Build right-hand side
        b = vector(n+1)
        for i in range(n):
            dist = distance(self.samples[i], (x,y))
            b[i] = self.variogram(dist)
        b[n] = 1
        
        # Solve system: C * weights = b
        weights = solve(C, b)
        
        # Compute interpolated value
        value = 0
        for i in range(n):
            value += weights[i] * self.samples[i].v
        
        return value
```

### RBF Implementation

```python
# Simplified RBF pseudocode
class RBF:
    def __init__(self, kernel, epsilon):
        self.samples = []
        self.kernel = kernel
        self.epsilon = epsilon
        self.coefficients = None
    
    def add_sample(self, x, y, v):
        self.samples.append((x, y, v))
    
    def fit(self):
        n = len(self.samples)
        
        # Build interpolation matrix
        A = matrix(n, n)
        for i in range(n):
            for j in range(n):
                dist = distance(self.samples[i], self.samples[j])
                A[i,j] = self.kernel(self.epsilon * dist)
        
        # Solve for coefficients
        b = [s.v for s in self.samples]
        self.coefficients = solve(A, b)
    
    def interpolate(self, x, y):
        if self.coefficients is None:
            self.fit()
        
        value = 0
        for i, sample in enumerate(self.samples):
            dist = distance(sample, (x,y))
            value += self.coefficients[i] * self.kernel(self.epsilon * dist)
        
        return value
```

## Testing Strategy

### Unit Tests

1. **Vector Upsampling**
   - Test with N=1, M arbitrary
   - Test with N=M (should return copy)
   - Test with various upsampling factors (2x, 3x, non-integer)
   - Test with known patterns (linear ramp, sinusoidal)
   - Verify endpoint preservation

2. **Matrix Upsampling**
   - Test with 1xN, Nx1 matrices
   - Test with various upsampling factors
   - Test with known images (checkerboard, gradient)
   - Verify boundary handling

3. **Advanced Interpolation**
   - Test with known functions (linear, quadratic)
   - Test with scattered data
   - Test with irregular spacing
   - Verify reproduction of sample points (interpolating polynomial property)

### Benchmarks

- Measure performance for different sizes
- Compare quality metrics (PSNR, SSIM for images)
- Memory usage profiling

## Design Principles

1. **In-Place Operations**: Follow existing math library pattern
2. **Destination-Based API**: Pre-allocated destination for zero-copy
3. **Float32 Precision**: Consistent with rest of math library
4. **Error Handling**: Validate inputs, return errors for invalid operations
5. **Minimize Allocations**: Reuse buffers where possible
6. **Function Length**: Keep functions under 30 lines, extract helpers

## Questions to Resolve

1. Should we support non-uniform upsampling (variable spacing)?
2. How to handle extrapolation beyond source range?
3. Should Kriging and RBF support higher dimensions (>2D)?
4. How to optimize for real-time performance?
5. Should we provide GPU-accelerated versions?
6. How to handle edge cases for very small/large scaling factors?
7. Should we support batch processing for multiple vectors/matrices?

## Dependencies

- Existing: `vec.Vector`, `mat.Matrix`
- New: Linear algebra operations for Kriging/RBF (matrix solve)
- Consider: Using existing matrix operations or add new solve functions

## Performance Considerations

- **Upsampling**: O(N) for vectors, O(N*M) for matrices
- **Kriging**: O(N³) setup, O(N²) per query
- **RBF**: O(N³) setup, O(N) per query
- For large N, consider approximation methods or GPU

## Future Enhancements

1. Lanczos resampling for image upscaling
2. Adaptive upsampling based on local features
3. Edge-preserving interpolation (guided filtering)
4. GPU-accelerated implementations
5. Sparse matrix optimizations for Kriging/RBF
6. Multi-threaded processing for large matrices

