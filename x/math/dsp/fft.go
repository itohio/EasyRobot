// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/itohio/EasyRobot/x/math/primitive/c64"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// FFT performs forward and inverse Fast Fourier Transforms on tensors.
// Automatically detects tensor type (F32 or C64) and shape to call appropriate primitive functions.
//
// For F32 tensors:
//   - Input: Real-valued float32 tensor
//   - Output: Packed complex format (real parts in [0:N], imaginary parts in [N:2*N])
//
// For C64 tensors:
//   - Input: Complex-valued complex64 tensor
//   - Output: Complex-valued complex64 tensor
//
// The FFT size is determined by the tensor shape, which will be padded to the next power of 2 if needed.
type FFT struct {
	shape    []int       // FFT dimensions (next power of 2 for each dimension)
	rank     int         // Tensor rank (1D, 2D, etc.)
	tempBuf  []float32   // temporary buffer for F32 operations (allocated when needed)
	tempBufC []complex64 // temporary buffer for C64 operations (allocated when needed)
	lut      []float32   // precomputed LUT for twiddle factors (optional)
}

// NewFFT creates an FFT processor for tensors of the specified shape.
// The shape will be rounded up to the next power of 2 for each dimension.
// If lut is provided (vararg), it will be used for optimized FFT operations.
// The LUT can be created using fp32.NewFFTLUT or c64.NewFFTLUT.
// Returns a value (not a pointer).
func NewFFT(shape []int, lut ...[]float32) FFT {
	if len(shape) == 0 {
		panic("FFT: shape must have at least one dimension")
	}
	for _, s := range shape {
		if s <= 0 {
			panic("FFT: all shape dimensions must be positive")
		}
	}

	// Round up to next power of 2 for each dimension
	nextPow2Shape := make([]int, len(shape))
	for i, s := range shape {
		nextPow2Shape[i] = fp32.NextPowerOfTwo(s)
	}

	fft := FFT{
		shape: nextPow2Shape,
		rank:  len(shape),
		// Buffers allocated only when needed (lazy allocation)
	}

	// Use LUT if provided
	if len(lut) > 0 && len(lut[0]) > 0 {
		fft.lut = lut[0]
	}

	return fft
}

// Forward performs a forward FFT transform on a tensor.
// Automatically detects tensor type (F32 or C64) and calls appropriate primitive function.
func (f *FFT) Forward(dst, src types.Tensor) {
	srcType := src.DataType()
	dstType := dst.DataType()

	if srcType != dstType {
		panic("FFT.Forward: source and destination must have same data type")
	}

	switch srcType {
	case types.FP32:
		f.forwardF32(dst, src)
	case types.C64:
		f.forwardC64(dst, src)
	default:
		panic("FFT.Forward: unsupported data type, must be FP32 or C64")
	}
}

// Backward performs an inverse FFT transform on a tensor.
// Automatically detects tensor type (F32 or C64) and calls appropriate primitive function.
func (f *FFT) Backward(dst, src types.Tensor) {
	srcType := src.DataType()
	dstType := dst.DataType()

	if srcType != dstType {
		panic("FFT.Backward: source and destination must have same data type")
	}

	switch srcType {
	case types.FP32:
		f.backwardF32(dst, src)
	case types.C64:
		f.backwardC64(dst, src)
	default:
		panic("FFT.Backward: unsupported data type, must be FP32 or C64")
	}
}

// forwardF32 performs forward FFT on F32 tensor (real-valued input, packed complex output)
func (f *FFT) forwardF32(dst, src types.Tensor) {
	if src.Rank() != f.rank {
		panic("FFT.Forward: source rank must match FFT rank")
	}
	if dst.Rank() != f.rank {
		panic("FFT.Forward: destination rank must match FFT rank")
	}

	srcData := types.GetTensorData[[]float32](src)
	dstData := types.GetTensorData[[]float32](dst)

	// For 1D: use packed format directly
	if f.rank == 1 {
		size := f.shape[0]
		// Allocate buffer only when needed
		if len(f.tempBuf) < 2*size {
			f.tempBuf = make([]float32, 2*size)
		}
		tempBuf := f.tempBuf[:2*size]
		srcLen := len(srcData)
		if srcLen > size {
			srcLen = size
		}

		// Copy real input to real part of packed format, zero imaginary parts
		copy(tempBuf[:srcLen], srcData[:srcLen])
		for i := srcLen; i < size; i++ {
			tempBuf[i] = 0
		}
		for i := range size {
			tempBuf[i+size] = 0
		}

		// Perform FFT
		if len(f.lut) > 0 {
			totalFactors := len(f.lut) / 2
			fp32.FFTWithLUT(tempBuf, f.lut, totalFactors)
		} else {
			fp32.FFT(tempBuf)
		}

		// Copy result to destination
		copyLen := 2 * size
		if copyLen > len(dstData) {
			copyLen = len(dstData)
		}
		copy(dstData, tempBuf[:copyLen])
		return
	}

	// For 2D+: delegate to FFT2D
	if f.rank == 2 {
		// FFT2D expects mat.Matrix, so we need to handle this
		// For now, delegate - but this needs proper tensor-to-matrix conversion
		panic("FFT.Forward: 2D FFT for F32 tensors requires matrix conversion, use FFT2D directly with matrices")
	}
	panic("FFT.Forward: rank > 2 not yet implemented")
}

// backwardF32 performs inverse FFT on F32 tensor (packed complex input, real-valued output)
func (f *FFT) backwardF32(dst, src types.Tensor) {
	if src.Rank() != f.rank {
		panic("FFT.Backward: source rank must match FFT rank")
	}
	if dst.Rank() != f.rank {
		panic("FFT.Backward: destination rank must match FFT rank")
	}

	srcData := types.GetTensorData[[]float32](src)
	dstData := types.GetTensorData[[]float32](dst)

	// For 1D: use packed format directly
	if f.rank == 1 {
		size := f.shape[0]
		// Allocate buffer only when needed
		if len(f.tempBuf) < 2*size {
			f.tempBuf = make([]float32, 2*size)
		}
		tempBuf := f.tempBuf[:2*size]
		srcLen := len(srcData)
		if srcLen > 2*size {
			srcLen = 2 * size
		}

		// Copy packed complex data
		copy(tempBuf, srcData[:srcLen])
		for i := srcLen; i < 2*size; i++ {
			tempBuf[i] = 0
		}

		// Perform inverse FFT
		// Note: IFFTWithLUT could be added later for optimization
		fp32.IFFT(tempBuf)

		// Copy real part to destination
		copyLen := size
		if copyLen > len(dstData) {
			copyLen = len(dstData)
		}
		copy(dstData, tempBuf[:copyLen])
		return
	}

	// For 2D+: delegate to FFT2D
	if f.rank == 2 {
		// FFT2D expects mat.Matrix, so we need to handle this
		panic("FFT.Backward: 2D IFFT for F32 tensors requires matrix conversion, use FFT2D directly with matrices")
	}
	panic("FFT.Backward: rank > 2 not yet implemented")
}

// forwardC64 performs forward FFT on C64 tensor (complex-valued)
func (f *FFT) forwardC64(dst, src types.Tensor) {
	if src.Rank() != f.rank {
		panic("FFT.Forward: source rank must match FFT rank")
	}
	if dst.Rank() != f.rank {
		panic("FFT.Forward: destination rank must match FFT rank")
	}

	srcData := types.GetTensorData[[]complex64](src)
	dstData := types.GetTensorData[[]complex64](dst)

	// For 1D
	if f.rank == 1 {
		size := f.shape[0]
		// Allocate buffer only when needed
		if len(f.tempBufC) < size {
			f.tempBufC = make([]complex64, size)
		}
		tempBuf := f.tempBufC[:size]
		srcLen := len(srcData)
		if srcLen > size {
			srcLen = size
		}

		// Copy and zero-pad
		copy(tempBuf, srcData[:srcLen])
		for i := srcLen; i < size; i++ {
			tempBuf[i] = 0
		}

		// Perform FFT
		if len(f.lut) > 0 {
			totalFactors := len(f.lut) / 2
			c64.FFTWithLUT(tempBuf, f.lut, totalFactors)
		} else {
			c64.FFT(tempBuf)
		}

		// Copy result
		copyLen := size
		if copyLen > len(dstData) {
			copyLen = len(dstData)
		}
		copy(dstData, tempBuf[:copyLen])
		return
	}

	// For 2D: delegate to FFT2DC
	if f.rank == 2 {
		rows := f.shape[0]
		cols := f.shape[1]
		fft2dc := NewFFT2DC(rows, cols, f.lut)
		fft2dc.Forward(dst, src)
		return
	}

	panic("FFT.Forward: rank > 2 not yet implemented")
}

// backwardC64 performs inverse FFT on C64 tensor (complex-valued)
func (f *FFT) backwardC64(dst, src types.Tensor) {
	if src.Rank() != f.rank {
		panic("FFT.Backward: source rank must match FFT rank")
	}
	if dst.Rank() != f.rank {
		panic("FFT.Backward: destination rank must match FFT rank")
	}

	srcData := types.GetTensorData[[]complex64](src)
	dstData := types.GetTensorData[[]complex64](dst)

	// For 1D
	if f.rank == 1 {
		size := f.shape[0]
		// Allocate buffer only when needed
		if len(f.tempBufC) < size {
			f.tempBufC = make([]complex64, size)
		}
		tempBuf := f.tempBufC[:size]
		srcLen := len(srcData)
		if srcLen > size {
			srcLen = size
		}

		// Copy and zero-pad
		copy(tempBuf, srcData[:srcLen])
		for i := srcLen; i < size; i++ {
			tempBuf[i] = 0
		}

		// Perform inverse FFT
		// Note: IFFTWithLUT could be added later for optimization
		c64.IFFT(tempBuf)

		// Copy result
		copyLen := size
		if copyLen > len(dstData) {
			copyLen = len(dstData)
		}
		copy(dstData, tempBuf[:copyLen])
		return
	}

	// For 2D: delegate to FFT2DC
	if f.rank == 2 {
		rows := f.shape[0]
		cols := f.shape[1]
		fft2dc := NewFFT2DC(rows, cols, f.lut)
		fft2dc.Backward(dst, src)
		return
	}

	panic("FFT.Backward: rank > 2 not yet implemented")
}
