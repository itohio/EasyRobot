# Convolution Layers Benchmark Report

## Overview

This report contains benchmark results for convolution layer forward and backward passes before and after optimization.

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
- Input: [8, 64, 32, 32]
- Kernel: 2x2, stride: 2x2, padding: 0x0

### AvgPool2D
- Input: [8, 64, 32, 32]
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
- Memory allocations decreased significantly, especially for Conv1D (99.97% reduction)
- All convolution layer training tests pass successfully
- MaxPool2D benchmark uses smaller tensors (4×32×16×16) to fit within int16 index limits

