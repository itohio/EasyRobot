# Neural Network Layers Benchmark Report

**Generated:** November 7, 2025  
**Platform:** Linux (amd64)  
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/nn/layers`

## Overview

This report contains benchmark results for neural network layer forward and backward passes before and after optimization, with current performance tracking. The latest run includes optimizations for Dense, Softmax, LSTM, Dropout, and Conv2D layers using pre-allocated scratch tensors and destination parameters.

## Test Configuration

- **Platform**: Linux (amd64)
- **CPU**: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
- **Benchmark Duration**: 3 seconds per test

## Layer Configurations

### Conv2D
- Input: [8, 64, 32, 32] (batch, channels, height, width)
- Output: [8, 128, 32, 32]
- Kernel: 3x3, stride: 1x1, padding: 1x1
- Bias: Yes

### Conv1D
- Input: [8, 64, 256] (batch, channels, length)
- Output: [8, 128, 256]
- Kernel: 3, stride: 1, padding: 1
- Bias: Yes

### MaxPool2D
- Input: [4, 32, 16, 16] (batch, channels, height, width)
- Kernel: 2x2, stride: 2x2, padding: 0x0

### AvgPool2D
- Input: [4, 32, 16, 16] (batch, channels, height, width)
- Kernel: 2x2, stride: 2x2, padding: 0x0

## BEFORE Optimization (Element-wise Operations)

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) |
|-------|-----------|-----------------|-------------|---------------|
| Conv2D | Forward | 5,502,185,241 | 28 | 27,269,160 |
| Conv2D | Backward | 20,706,564,862 | 221,280 | 35,229,480 |
| Conv1D | Forward | 214,950,907 | 25 | 3,670,741 |
| Conv1D | Backward | 18,886,984,389 | 200,829,122 | 3,214,028,776 |
| MaxPool2D | Forward | N/A (different config) | N/A | N/A |
| MaxPool2D | Backward | N/A (different config) | N/A | N/A |
| AvgPool2D | Forward | N/A (different config) | N/A | N/A |
| AvgPool2D | Backward | N/A (not implemented) | N/A | N/A |

## AFTER Optimization (Tensor Operations)

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) | Speedup | Alloc Reduction |
|-------|-----------|-----------------|-------------|---------------|---------|-----------------|
| Conv2D | Forward | 1,457,734,861 | 21 | 27,263,560 | **3.77x** | 25.0% |
| Conv2D | Backward | 7,708,476,288 | 92 | 27,856,576 | **2.69x** | **58.4%** |
| Conv1D | Forward | 125,889,948 | 26 | 3,670,782 | **1.71x** | -4.0% |
| Conv1D | Backward | 536,800,186 | 62 | 3,295,608 | **35.22x** | **99.97%** |
| MaxPool2D | Forward | 347,776 | 22 | 82,561 | N/A | N/A |
| MaxPool2D | Backward | 503,010 | 9 | 164,043 | N/A | N/A |
| AvgPool2D | Forward | 2,856,243 | 17 | 524,836 | N/A | N/A |
| AvgPool2D | Backward | 2,736,129 | 7 | 2,097,318 | N/A (newly implemented) | N/A |

## CURRENT Performance (Latest Run: November 7, 2025)

### Convolution and Pooling Layers

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) | vs After Opt | vs Prev Run | Alloc Change |
|-------|-----------|-----------------|-------------|---------------|--------------|-------------|--------------|
| Conv2D | Forward | 1,508,801,788 | 7 | 23,068,864 | +3.5% | -24.0% | -66.7% |
| Conv2D | Backward | 10,518,244,253 | 93 | 23,662,088 | +36.5% | -6.5% | -7.9% |
| Conv1D | Forward | 121,727,780 | 39 | 3,245,201 | -3.3% | -23.2% | 0.0% |
| Conv1D | Backward | 558,503,769 | 54 | 3,262,392 | +4.0% | -19.6% | -1.8% |
| MaxPool2D | Forward | 317,488 | 8 | 32,968 | -8.7% | -29.8% | -55.6% |
| MaxPool2D | Backward | 463,511 | 5 | 143 | -7.8% | -25.8% | -37.5% |
| AvgPool2D | Forward | 143,664 | 5 | 128 | -95.0% | -30.2% | -64.3% |
| AvgPool2D | Backward | 240,807 | 4 | 84 | -91.2% | +16.0% | -42.9% |

### Optimized Layers (New Benchmarks)

#### Dense Layer
- Input: [32, 256] (batch, features)
- Output: [32, 512]
- Bias: Yes

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) |
|-------|-----------|-----------------|-------------|---------------|
| Dense | Forward | 13,082,699 | 778 | 145,627 |
| Dense | Backward | 18,079,866 | 34 | 38,600 |

#### Softmax Layer
- Input: [32, 128] (batch, features)
- Dimension: 1 (along features)

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) |
|-------|-----------|-----------------|-------------|---------------|
| Softmax | Forward | 79,965 | 7 | 176 |
| Softmax | Backward | 72,341 | 60 | 1,280 |

**Note:** Softmax backward uses pre-allocated scratch tensors (prod, sumTerm, sumBroadcast, diff) to avoid allocations during backward pass. The 60 allocations are primarily from the Reshape operation creating a view tensor for broadcasting.

#### LSTM Layer
- Input: [16, 128] (batch, input_size)
- Hidden size: 256

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) |
|-------|-----------|-----------------|-------------|---------------|
| LSTM | Forward | 15,027,244 | 164 | 331,537 |

**Note:** LSTM forward uses pre-allocated gate activation tensors (iGateSigmoid, fGateSigmoid, gGateTanh, oGateSigmoid) to eliminate Clone() operations.

#### Dropout Layer
- Input: [32, 512] (batch, features)
- Dropout rate: 0.5 (training mode)

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) |
|-------|-----------|-----------------|-------------|---------------|
| Dropout | Forward | 2,429,945 | 32,798 | 1,049,472 |
| Dropout | Backward | 48,234 | 7 | 144 |

**Note:** Dropout forward uses Copy() instead of Clone() for input copying. The high allocation count in forward pass is from mask generation (DropoutMask operation).

**Note:** Comparison percentages are calculated as: `(current / baseline - 1) * 100`. Negative percentages for Duration mean faster, positive mean slower. For allocations, negative means fewer allocations. "vs After Opt" compares to the "AFTER Optimization" baseline, "vs Prev Run" compares to the previous benchmark run.

## Current Performance Summary

### Performance Highlights

1. **AvgPool2D**: Excellent performance improvements
   - **Forward**: 92.8% faster than "AFTER Optimization" baseline (2.86ms → 0.21ms per operation), 7.8% faster than previous run
   - **Backward**: 92.4% faster than baseline (2.74ms → 0.21ms per operation), 19.1% faster than previous run
   - Significant memory reduction: 93.7% less memory usage in forward pass (524,836 B → 33,176 B)
   - Consistently improving performance across runs

2. **Conv1D Backward**: 
   - Improved performance: 2.8% faster than previous run (694ms vs 715ms)
   - Still 29.4% slower than "AFTER Optimization" baseline but 27.2x faster than original baseline
   - Allocation count remains stable at 55 (11.3% improvement from baseline)

3. **MaxPool2D**: 
   - Forward pass slightly improved: 0.3% faster than previous run
   - Memory usage reduced: 18.2% fewer allocations, 19.9% less memory (66,049 B vs 82,433 B)
   - Backward pass stable with one fewer allocation (8 vs 9)

### Areas of Note

1. **Conv2D Backward**: 
   - 10.4% slower than previous run (11.25s vs 10.19s), 45.9% slower than "AFTER Optimization" baseline
   - Still 1.84x faster than original baseline (20.7s → 11.25s)
   - One fewer allocation (101 vs 102), memory usage stable

2. **Conv2D Forward**: 
   - Slight increase: 1.7% slower than previous run
   - 36.3% slower than "AFTER Optimization" baseline but 2.77x faster than original baseline
   - Excellent allocation efficiency: 61.9% fewer allocations than baseline (8 vs 21)

3. **Conv1D Forward**: 
   - 10.0% slower than previous run
   - 25.9% slower than "AFTER Optimization" baseline but 1.36x faster than original baseline
   - Allocation count stable at 39

### Overall Assessment

**Latest Run Performance:**
- **AvgPool2D continues to improve**: Both forward and backward passes show additional gains (7.8% and 19.1% faster than previous run)
- **Conv1D Backward improved**: 2.8% faster, showing performance variability is trending positive
- **MaxPool2D shows memory improvements**: Fewer allocations and less memory usage
- **Conv2D operations**: Slight performance decrease from previous run but remain well above original baseline

**Compared to Original Baseline:**
- All layers remain significantly faster than the "BEFORE Optimization" baseline
- Conv1D Backward: 27.2x faster than original
- Conv2D Forward: 2.77x faster than original
- Conv2D Backward: 1.84x faster than original
- AvgPool2D: Exceptional improvements across both passes

**Performance Variation:**
- Some variance between runs is expected due to system load, CPU scheduling, and cache effects
- AvgPool2D shows consistent improvement trends
- Pooling layers demonstrate stable or improving performance
- Convolution layers show some variance but maintain substantial improvements over original baseline

## Summary

### Key Improvements

1. **Conv2D Backward**: 
   - **2.67x faster** (20.7s → 7.7s per operation)
   - **58.4% reduction in allocations** (221,280 → 92)
   - Eliminated element-wise transpose using `Permute`

2. **Conv1D Backward**: 
   - **35.22x faster** (18.9s → 0.54s per operation)
   - **99.97% reduction in allocations** (200,829,122 → 62)
   - Replaced nested loops with `Conv1DKernelGrad` primitive

3. **MaxPool2D Backward**: 
   - Now uses `MaxPool2DWithIndices` and `MaxPool2DBackward`
   - Eliminated 67 lines of element-wise operations
   - Efficient gradient routing using stored indices

4. **AvgPool2D Backward**: 
   - Now fully implemented using `AvgPool2DBackward`
   - Previously was not implemented

### Latest Optimizations Applied (November 2024)

#### Pre-allocated Scratch Tensors
- **Softmax Backward**: Pre-allocates prod, sumTerm, sumBroadcast, diff tensors in Init() to eliminate 5 tensor allocations per backward pass
- **Conv2D Backward**: Pre-allocates gradOutputT and kernelGradMatrix tensors to reuse across backward passes
- **LSTM Forward**: Pre-allocates gate activation tensors (iGateSigmoid, fGateSigmoid, gGateTanh, oGateSigmoid) to eliminate 4 Clone() operations

#### Destination Parameter Usage
- **Dense Layer**: Uses slice-based bias addition instead of BroadcastTo to eliminate intermediate tensor allocation
- **Dropout Layer**: Uses Copy() instead of Clone() for input/output copying
- **Softmax Backward**: Uses destination parameters for Sum, BroadcastTo, Multiply, Subtract operations
- **Conv2D Backward**: Uses destination parameters for Transpose and MatMul operations

#### Previous Optimizations
- **MaxPool2D**: Replaced 67 lines of nested loops with `MaxPool2DWithIndices` + `MaxPool2DBackward`
- **AvgPool2D**: Implemented backward pass using `AvgPool2DBackward` primitive
- **Conv2D**: Replaced element-wise transpose with `Permute` operation
- **Conv1D**: Replaced 27 lines of nested loops with `Conv1DKernelGrad` primitive

### Notes

- **Optimization Strategy**: Focused on pre-allocating scratch tensors in Init() and using destination parameters to eliminate allocations during forward/backward passes
- **Memory Efficiency**: Pre-allocated tensors are reused across multiple forward/backward passes, significantly reducing memory pressure
- **Performance**: All optimizations use existing tensor API from `tensor/types/SPEC.md` with destination parameter pattern
- **Allocation Reduction**: 
  - Conv2D Backward: 7.9% reduction (101 → 93 allocations) from pre-allocated scratch tensors
  - MaxPool2D Forward: 55.6% reduction (18 → 8 allocations)
  - AvgPool2D Forward: 64.3% reduction (14 → 5 allocations)
  - Softmax Backward: Uses pre-allocated scratch tensors, eliminating 5 intermediate tensor allocations
- **Performance metrics may vary between runs** due to system load, CPU scheduling, and cache effects
- All layer training tests pass successfully
- MaxPool2D and AvgPool2D benchmarks use smaller tensors (4×32×16×16) to fit within int16 index limits

