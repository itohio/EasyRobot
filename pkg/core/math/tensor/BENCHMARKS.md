# Tensor Implementation Benchmarks

This document contains performance comparisons between different tensor implementations: `eager_tensor` (Go native) and `gorgonia` (uses optimized BLAS/LAPACK libraries).

## Benchmark Environment

- **CPU**: Intel(R) Core(TM) i7-8750H @ 2.20GHz
- **OS**: Linux (WSL2)
- **Go Version**: 1.24.9
- **Date**: November 11, 2025

## Running Benchmarks

To run the comparative benchmarks:

```bash
cd /home/andrius/projects/itohio/EasyRobot
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24 go test -bench=. -benchmem -benchtime=1s ./pkg/core/math/tensor/bench_test.go
```

To run gorgonia-specific benchmarks:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24 go test -bench=. -benchmem ./pkg/core/math/tensor/gorgonia/
```

## Performance Summary

### Matrix Multiplication (MatMul)

**Key Finding**: Gorgonia is **2-43x faster** than eager_tensor for matrix multiplication, with performance gains increasing for larger matrices.

| Size | Eager (ns/op) | Gorgonia (ns/op) | Speedup |
|------|---------------|------------------|---------|
| 32×32 | 44,625 | 19,685 | **2.27x** |
| 128×128 | 4,013,304 | 565,858 | **7.09x** |
| 512×512 | 552,651,694 | 12,764,015 | **43.3x** |
| 64×128×256 | 3,757,413 | 641,429 | **5.86x** |

**Analysis**: Gorgonia's use of optimized BLAS libraries provides dramatic speedups for matrix operations. The speedup increases with matrix size, making it ideal for deep learning workloads.

### Element-wise Addition

**Key Finding**: Eager tensor is **2-2.5x faster** for element-wise operations.

| Size | Eager (ns/op) | Gorgonia (ns/op) | Winner |
|------|---------------|------------------|--------|
| 1K | 2,325 | 5,307 | Eager (2.3x) |
| 64K | 61,978 | 257,216 | Eager (4.1x) |
| 1M | 1,350,077 | 3,381,956 | Eager (2.5x) |

**Analysis**: Eager tensor's direct memory access is more efficient for simple element-wise operations. Gorgonia has overhead from its abstraction layer.

### Element-wise Multiplication

**Key Finding**: Eager tensor is **~1.2x faster** for element-wise multiplication.

| Size | Eager (ns/op) | Gorgonia (ns/op) | Winner |
|------|---------------|------------------|--------|
| 64K | 122,420 | 146,144 | Eager (1.19x) |

### ReLU Activation

**Key Finding**: Eager tensor is **~2.2x faster** for ReLU.

| Size | Eager (ns/op) | Gorgonia (ns/op) | Winner |
|------|---------------|------------------|--------|
| 64K | 68,489 | 147,632 | Eager (2.16x) |

### Transpose

**Key Finding**: Eager tensor is **~234x faster** for transpose (uses view, not copy).

| Size | Eager (ns/op) | Gorgonia (ns/op) | Winner |
|------|---------------|------------------|--------|
| 128×256 | 253 | 59,219 | Eager (234x) |

**Analysis**: Eager tensor's transpose is a view operation (O(1)), while Gorgonia performs a full copy.

### Sum Reduction

**Key Finding**: Gorgonia is **~3.8x faster** for reduction operations.

| Size | Eager (ns/op) | Gorgonia (ns/op) | Winner |
|------|---------------|------------------|--------|
| 64K | 306,190 | 80,088 | Gorgonia (3.82x) |

**Analysis**: Gorgonia's optimized reduction operations benefit from vectorization.

### Convolution Operations

**Note**: Gorgonia convolution operations are not yet implemented. Benchmarks show eager_tensor performance:

#### Conv1D

| Configuration | Time (ns/op) | Memory (B/op) |
|---------------|--------------|---------------|
| Kernel=3, Stride=1 | 25,146,512 | 262,573 |
| Kernel=5, Stride=2 | 175,988,080 | 524,812 |

#### Conv2D

| Configuration | Time (ns/op) | Memory (B/op) |
|---------------|--------------|---------------|
| 3×3, Stride=1, Small (4×32×32×32) | 336,071,149 | 1,048,904 |
| 3×3, Stride=1, Medium (8×64×64×64) | 14,478,798,521 | 16,777,544 |
| 3×3, Stride=2, Large (4×128×128×128) | 30,989,662,737 | 16,777,544 |
| 1×1, Stride=1 (8×256×56×56) | 5,553,639,274 | 12,845,384 |

## Memory Allocation Comparison

### MatMul (128×128)

- **Eager**: 65,846 B/op, 6 allocs/op
- **Gorgonia**: 67,854 B/op, 24 allocs/op

**Analysis**: Eager tensor has fewer allocations but similar total memory usage.

### Add (64K elements)

- **Eager**: 544 B/op, 5 allocs/op
- **Gorgonia**: 263,368 B/op, 7 allocs/op

**Analysis**: Eager tensor allocates significantly less memory for element-wise operations due to in-place optimization opportunities.

## Recommendations

### Use Gorgonia When:

1. **Matrix Multiplication** is a primary operation (neural networks, linear algebra)
2. Working with **large matrices** (>128×128)
3. Performing **reduction operations** (sum, mean, etc.)
4. Need compatibility with existing Gorgonia code
5. CPU-bound workloads where BLAS optimization matters

### Use Eager Tensor When:

1. Performing **element-wise operations** (add, multiply, ReLU, etc.)
2. **Transpose** operations are frequent
3. Working with **smaller tensors** (<32×32)
4. Memory allocation overhead is a concern
5. Need fast prototyping with simple operations

### Hybrid Approach:

For complex deep learning workloads, consider:
- Use **Gorgonia for matmul** (forward/backward passes in dense layers)
- Use **Eager for activations** (ReLU, Sigmoid, etc.)
- Convert between implementations using `gorgonia.ToEagerTensor()` and `gorgonia.FromEagerTensor()`

The conversion overhead is minimal compared to operation costs for reasonably sized tensors.

## Future Work

1. **Implement Conv1D/Conv2D in Gorgonia**: Expected to provide significant speedups using optimized convolution libraries.
2. **TFLite Integration**: Add TFLite tensor implementation for comparison with hardware-accelerated inference.
3. **GPU Acceleration**: Explore CUDA/ROCm backends for Gorgonia.
4. **Vectorization**: Optimize eager_tensor element-wise operations with SIMD instructions.
5. **Benchmark on Different Hardware**: ARM, Apple Silicon, AMD CPUs.

## Notes

- Gorgonia benchmarks use the `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24` flag for compatibility with Go 1.24.
- All benchmarks use `FP32` (float32) data type unless specified otherwise.
- Benchmark times are wall clock time, not CPU time.
- Results may vary based on CPU architecture, cache size, and system load.

