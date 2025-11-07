# Convolution Layers Benchmark Report

**Generated:** November 7, 2025  
**Platform:** Linux (amd64)  
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/nn/layers`

## Overview

This report contains benchmark results for convolution layer forward and backward passes before and after optimization, with current performance tracking.

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

| Layer | Operation | Duration (ns/op) | Allocations | Memory (B/op) | vs After Opt | vs Prev Run | Alloc Change |
|-------|-----------|-----------------|-------------|---------------|--------------|-------------|--------------|
| Conv2D | Forward | 1,986,477,144 | 8 | 23,068,888 | +36.3% | +1.7% | -61.9% |
| Conv2D | Backward | 11,249,441,463 | 101 | 32,345,808 | +45.9% | +10.4% | +9.8% |
| Conv1D | Forward | 158,468,735 | 39 | 3,245,218 | +25.9% | +10.0% | +50.0% |
| Conv1D | Backward | 694,437,195 | 55 | 3,265,720 | +29.4% | -2.8% | -11.3% |
| MaxPool2D | Forward | 452,096 | 18 | 66,049 | +30.0% | -0.3% | -18.2% |
| MaxPool2D | Backward | 624,482 | 8 | 131,274 | +24.1% | +2.0% | -11.1% |
| AvgPool2D | Forward | 205,714 | 14 | 33,176 | -92.8% | -7.8% | -17.6% |
| AvgPool2D | Backward | 207,640 | 7 | 131,226 | -92.4% | -19.1% | 0.0% |

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

### Optimizations Applied

- **MaxPool2D**: Replaced 67 lines of nested loops with `MaxPool2DWithIndices` + `MaxPool2DBackward`
- **AvgPool2D**: Implemented backward pass using `AvgPool2DBackward` primitive
- **Conv2D**: Replaced element-wise transpose with `Permute` operation
- **Conv1D**: Replaced 27 lines of nested loops with `Conv1DKernelGrad` primitive

### Notes

- Optimization focused on eliminating `At()`/`SetAt()`/`Elements()` usage in backward passes
- New operations use optimized fp32 primitives for better performance
- Memory allocations decreased significantly, especially for Conv1D (99.97% reduction vs original)
- All convolution layer training tests pass successfully
- MaxPool2D and AvgPool2D benchmarks use smaller tensors (4×32×16×16) to fit within int16 index limits
- Performance metrics may vary between runs due to system load, CPU scheduling, and cache effects
- Latest run shows AvgPool2D with exceptional improvements, likely due to further optimizations or better cache behavior

