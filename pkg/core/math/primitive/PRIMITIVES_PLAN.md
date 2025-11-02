# Math Primitives Implementation Plan

## Overview
Low-level optimized primitives for matrix-vector operations. These will eventually support architecture-specific implementations (ARM NEON, x86 SSE, etc.) via build tags, but initially provide highly optimized Go-only implementations.

## Design Principles

1. **Zero Allocations**: All operations work on raw `[]float32` slices
2. **No Bounds Checking**: Use slice indexing that compiler can optimize away
3. **Stride Support**: Flexible memory layout for efficient operations
4. **Hot Path Optimization**: Critical loops written for compiler optimization
5. **Pointer Arithmetic Pattern**: Use slice indexing to avoid bounds checks

## Primitive Categories

### 1. Array Operations (`array.go`)
Element-wise operations on arrays with stride support:
- `SumArr(dst, a, b, num, strideA, strideB)` - Element-wise addition
- `DiffArr(dst, a, b, num, strideA, strideB)` - Element-wise subtraction
- `MulArr(dst, a, b, num, strideA, strideB)` - Element-wise multiplication
- `DivArr(dst, a, b, num, strideA, strideB)` - Element-wise division
- `SumArrConst(dst, src, c, num, stride)` - Add constant to array
- `MulArrConst(dst, src, c, num, stride)` - Multiply array by constant
- `SumArrAdd(dst, src, c, num, stride)` - Add src+c to dst (accumulate)
- `MulArrAdd(dst, src, c, num, stride)` - Add src*c to dst (accumulate)
- `Sum(a, num, stride)` - Sum of array (returns float32)
- `SqrSum(a, num, stride)` - Sum of squares
- `MinArr(a, num, stride)` - Minimum value and index
- `MaxArr(a, num, stride)` - Maximum value and index
- `MeanArr(a, num, stride)` - Mean of array
- `MomentsArr(mean, stddev, a, num, stride)` - Compute mean and stddev

### 2. Vector Operations (`vector.go`)
Vector-specific operations:
- `HadamardProduct(dst, a, b, num, strideA, strideB)` - Element-wise multiply
- `HadamardProductAdd(dst, a, b, num, strideA, strideB)` - Multiply and add
- `DotProduct(a, b, num, strideA, strideB)` - Dot product (returns float32)
- `DotProduct2D(a, b, N, M, K, L)` - Dot product of KxL submatrix
- `NormalizeVec(dst, num, stride)` - Normalize vector
- `OuterProduct(dst, u, v, N, M, bias)` - Outer product (u * v^T)
- `OuterProductConst(dst, u, v, N, M, alpha, bias)` - Scaled outer product
- `OuterProductAddConst(dst, u, v, N, M, alpha, bias)` - Scaled outer product add

### 3. Matrix Operations (`matrix.go`)
Matrix-vector and matrix operations:
- `MatMulVec(dst, vec, mat, N, M, transposed, bias)` - Matrix * vector
- `MatMulVecAdd(dst, vec, mat, N, M, transposed, bias)` - Matrix * vector (add to dst)
- `MatTranspose(dst, src, width, height)` - Transpose matrix
- `MinMat(a, inWidth, width, height, stride)` - Find minimum in matrix
- `MaxMat(a, inWidth, width, height, stride)` - Find maximum in matrix
- `MeanMat(a, inWidth, width, height, stride)` - Mean of matrix

### 4. Convolution Operations (`conv.go`)
Convolution primitives:
- `Convolve1DAdd(dst, vec, kernel, N, M, stride, transposed)` - 1D convolution
- `Convolve2DAdd(dst, mat, kernel, N, M, K, L, stride, transposed)` - 2D convolution

## Implementation Patterns

### Pattern 1: Hot Loop Optimization
```go
// Use indexed access with pre-computed bounds to help compiler optimize
func DotProduct(a, b []float32, num int, strideA, strideB int) float32 {
    if num == 0 {
        return 0
    }
    
    acc := float32(0.0)
    i := 0
    pa := 0
    pb := 0
    
    // Unroll small loops for better optimization
    for num > 0 {
        acc += a[pa] * b[pb]
        pa += strideA
        pb += strideB
        num--
    }
    
    return acc
}
```

### Pattern 2: No Bounds Checking
```go
// Use slice operations that compiler can optimize
func SumArr(dst, a, b []float32, num int, strideA, strideB int) {
    pa := 0
    pb := 0
    pd := 0
    
    // Compiler will optimize these bounds checks away in hot paths
    for i := 0; i < num; i++ {
        dst[pd] = a[pa] + b[pb]
        pa += strideA
        pb += strideB
        pd++
    }
}
```

### Pattern 3: Stride-based Access
```go
// Flexible stride support for different memory layouts
func MatMulVec(dst, vec, mat []float32, N, M int, transposed, bias bool) {
    if !transposed {
        // Row-major: mat[i + j * (N + bias_offset)]
        matRowSize := N
        if bias {
            matRowSize++
        }
        for j := 0; j < M; j++ {
            acc := float32(0.0)
            for i := 0; i < N; i++ {
                acc += vec[i] * mat[i + j*matRowSize]
            }
            if bias {
                acc += mat[N + j*matRowSize]
            }
            dst[j] = acc
        }
    } else {
        // Column-major: mat[j + i * N]
        for j := 0; j < N; j++ {
            acc := float32(0.0)
            matRowSize := N
            if bias {
                matRowSize++
            }
            for i := 0; i < M; i++ {
                acc += vec[i] * mat[j + i*matRowSize]
            }
            dst[j] = acc
        }
    }
}
```

## Architecture Tags (Future)

```go
// +build !arm,!arm64,!amd64

// pure.go - Go-only implementation (default)

// +build arm arm64

// neon.go - ARM NEON optimized implementation

// +build amd64

// sse.go - x86 SSE optimized implementation
```

## File Structure

```
pkg/core/math/primitive/
├── array.go          # Array operations
├── vector.go         # Vector operations
├── matrix.go         # Matrix operations
├── conv.go           # Convolution operations
├── array_test.go     # Tests
├── vector_test.go    # Tests
├── matrix_test.go    # Tests
├── conv_test.go      # Tests
└── PRIMITIVES_PLAN.md # This file
```

## Usage in Tensor Package

These primitives will be used by the tensor package:

```go
// In tensor package
import "github.com/itohio/EasyRobot/pkg/core/math/primitive"

// Example: tensor element-wise addition
func (t *Tensor) Add(other *Tensor) {
    primitive.SumArr(t.Data, t.Data, other.Data, t.Size(), 1, 1)
}

// Example: tensor matrix multiplication
func (t *Tensor) MatMul(other *Tensor) *Tensor {
    // Use primitive.MatMulVec for efficient computation
    result := NewTensor(resultShape)
    primitive.MatMulVec(result.Data, t.Data, other.Data, N, M, false, false)
    return result
}
```

## Performance Considerations

1. **Loop Unrolling**: Compiler can unroll small fixed-size loops
2. **SIMD**: Future architecture-specific implementations will use SIMD
3. **Cache Alignment**: Operations work on contiguous memory when possible
4. **Branch Prediction**: Minimize branches in hot loops
5. **Register Allocation**: Use local variables for accumulator patterns

