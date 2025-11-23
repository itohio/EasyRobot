# C64 Primitive Operations Reference

This document catalogs all primitive operations implemented in the `c64` package for complex64 operations, optimized for embedded systems.

## Table of Contents

1. [DSP Operations](#dsp-operations)
2. [Utility Operations](#utility-operations)

## DSP Operations

Located in `dsp_fft.go`. Digital Signal Processing operations optimized for embedded systems.

### Fast Fourier Transform (FFT)

| Operation | Function | Description |
|-----------|----------|-------------|
| FFT | `FFT(data []complex64)` | In-place complex FFT using Cooley-Tukey algorithm. **Requires power-of-2 size.** |
| IFFT | `IFFT(data []complex64)` | In-place complex inverse FFT. **Requires power-of-2 size.** |
| ComputeFFTLUT | `ComputeFFTLUT(fftSize int) int` | Returns the required buffer size (in float32 elements) for FFT LUT of given size. |
| FillFFTLUT | `FillFFTLUT(dst []float32, fftSize int) int` | Fills the LUT buffer with precomputed twiddle factors. Returns number of factors. |
| FFTWithLUT | `FFTWithLUT(data []complex64, lut []float32, totalFactors int)` | Performs FFT using precomputed LUT for better performance. |
| NextPowerOfTwo | `nextPowerOfTwo(n int) int` | Returns the next power of two >= n (internal). |
| IsPowerOfTwo | `isPowerOfTwo(n int) bool` | Checks if a number is a power of two (internal). |

**Notes:**
- All FFT operations require power-of-2 sizes for optimal performance
- Use `nextPowerOfTwo` to pad non-power-of-2 data
- LUT version (`FFTWithLUT`) provides better performance for repeated FFTs of the same size
- Optimized with BCE techniques: reslicing and for range loops
- Uses `math32` package for all floating-point operations (faster than `math`)

## Utility Operations

| Operation | Function | Description |
|-----------|----------|-------------|
| Pool | `Pool` | Memory pool for complex64 allocations (from generics/helpers) |

**Total Operations**: 7 functions (FFT: 6, Utility: 1)

