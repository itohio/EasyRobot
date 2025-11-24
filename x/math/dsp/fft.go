// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/itohio/EasyRobot/x/math/primitive/c64"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
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
	shape shape.Shape // FFT dimensions (next power of 2 for each dimension)
	rank  int         // Tensor rank (1D, 2D, etc.)

	dtype   types.DataType
	withLUT bool

	temp types.Tensor
	lut  types.Tensor

	lutFor     int
	lutFactors int
}

// Release releases any pooled tensors owned by FFT.
func (f *FFT) Release() {
	f.releaseTemp()
	f.releaseLUT()
}

func (f *FFT) releaseTemp() {
	if tensor.IsNil(f.temp) {
		return
	}
	f.temp.Release()
	f.temp = nil
}

func (f *FFT) releaseLUT() {
	if tensor.IsNil(f.lut) {
		return
	}
	f.lut.Release()
	f.lut = nil
	f.lutFor = 0
	f.lutFactors = 0
}

func (f *FFT) ensureType(dt types.DataType) {
	if dt != types.FP32 && dt != types.C64 {
		panic("FFT: unsupported tensor data type, must be FP32 or C64")
	}
	if f.dtype == types.DT_UNKNOWN {
		f.dtype = dt
		return
	}
	if f.dtype != dt {
		panic("FFT: data type mismatch with previously locked type")
	}
}

func (f *FFT) ensureTemp(size int) types.Tensor {
	if f.dtype == types.DT_UNKNOWN {
		panic("FFT: tensor data type not initialized")
	}

	var needed int
	switch f.dtype {
	case types.FP32:
		needed = 2 * size
	case types.C64:
		needed = size
	default:
		panic("FFT: unsupported tensor data type")
	}

	if tensor.IsNil(f.temp) || f.temp.Size() < needed {
		f.releaseTemp()
		f.temp = tensor.New(f.dtype, shape.NewShape(needed))
	}
	return f.temp
}

func (f *FFT) ensureLUT(size int) (any, int) {
	if !f.withLUT {
		return nil, 0
	}
	if f.dtype == types.DT_UNKNOWN {
		panic("FFT: tensor data type not initialized")
	}

	switch f.dtype {
	case types.FP32:
		required := fp32.ComputeFFTLUT(size)
		if tensor.IsNil(f.lut) || f.lut.Size() < required {
			f.releaseLUT()
			f.lut = tensor.New(types.FP32, shape.NewShape(required))
		}
		if f.lutFor != size {
			data := types.GetTensorData[[]float32](f.lut)
			total := fp32.FillFFTLUT(data[:required], size)
			f.lutFor = size
			f.lutFactors = total
		}
		if tensor.IsNil(f.lut) || f.lutFactors == 0 {
			return nil, 0
		}
		data := types.GetTensorData[[]float32](f.lut)
		return data[:fp32.ComputeFFTLUT(size)], f.lutFactors
	case types.C64:
		required := c64.ComputeFFTLUT(size)
		if tensor.IsNil(f.lut) || f.lut.Size() < required {
			f.releaseLUT()
			f.lut = tensor.New(types.C64, shape.NewShape(required))
		}
		if f.lutFor != size {
			data := types.GetTensorData[[]complex64](f.lut)
			total := c64.FillFFTLUT(data[:required], size)
			f.lutFor = size
			f.lutFactors = total
		}
		if tensor.IsNil(f.lut) || f.lutFactors == 0 {
			return nil, 0
		}
		data := types.GetTensorData[[]complex64](f.lut)
		return data[:c64.ComputeFFTLUT(size)], f.lutFactors
	default:
		panic("FFT: unsupported tensor data type")
	}
}

// NewFFT creates an FFT processor for tensors of the specified shape.
// The shape will be rounded up to the next power of 2 for each dimension.
// When withLUT is true, FFT lazily creates twiddle factor LUTs sized to the
// transform on first use and reuses them across calls.
// Returns a value (not a pointer).
func NewFFT(shape shape.Shape, withLUT bool) FFT {
	if shape.Rank() == 0 {
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

	return FFT{
		shape:   nextPow2Shape,
		rank:    len(shape),
		withLUT: withLUT,
	}
}

// Forward performs a forward FFT transform on a tensor.
// Automatically detects tensor type (F32 or C64) and calls appropriate primitive function.
func (f *FFT) Forward(dst, src types.Tensor) {
	srcType := src.DataType()
	dstType := dst.DataType()

	if srcType != dstType {
		panic("FFT.Forward: source and destination must have same data type")
	}

	f.ensureType(srcType)

	switch f.dtype {
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

	f.ensureType(srcType)

	switch f.dtype {
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
		tempTensor := f.ensureTemp(size)
		tempBuf := types.GetTensorData[[]float32](tempTensor)
		tempBuf = tempBuf[:2*size]
		realPart := tempBuf[:size]
		imagPart := tempBuf[size:]
		srcLen := len(srcData)
		if srcLen > size {
			srcLen = size
		}

		// Copy real input to real part of packed format, zero imaginary parts
		copy(realPart[:srcLen], srcData[:srcLen])
		for i := srcLen; i < size; i++ {
			realPart[i] = 0
		}
		for i := range size {
			imagPart[i] = 0
		}

		// Perform FFT
		if lut, totalFactors := f.ensureLUT(size); totalFactors > 0 {
			fp32.FFTWithLUT(tempBuf, lut.([]float32), totalFactors)
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
		tempTensor := f.ensureTemp(size)
		tempBuf := types.GetTensorData[[]float32](tempTensor)
		tempBuf = tempBuf[:2*size]
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
		if lut, totalFactors := f.ensureLUT(size); totalFactors > 0 {
			fp32.IFFTWithLUT(tempBuf, lut.([]float32), totalFactors)
		} else {
			fp32.IFFT(tempBuf)
		}

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
		tempTensor := f.ensureTemp(size)
		tempBuf := types.GetTensorData[[]complex64](tempTensor)
		tempBuf = tempBuf[:size]
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
		if lut, totalFactors := f.ensureLUT(size); totalFactors > 0 {
			c64.FFTWithLUT(tempBuf, lut.([]complex64), totalFactors)
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
		var fft2dc FFT2DC
		maxDim := rows
		if cols > maxDim {
			maxDim = cols
		}
		if lut, totalFactors := f.ensureLUT(maxDim); totalFactors > 0 {
			lutC := lut.([]complex64)
			fft2dc = NewFFT2DC(rows, cols, lutC[:totalFactors])
		} else {
			fft2dc = NewFFT2DC(rows, cols)
		}
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
		tempTensor := f.ensureTemp(size)
		tempBuf := types.GetTensorData[[]complex64](tempTensor)
		tempBuf = tempBuf[:size]
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
		if lut, totalFactors := f.ensureLUT(size); totalFactors > 0 {
			c64.IFFTWithLUT(tempBuf, lut.([]complex64), totalFactors)
		} else {
			c64.IFFT(tempBuf)
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
		maxDim := rows
		if cols > maxDim {
			maxDim = cols
		}
		var fft2dc FFT2DC
		if lut, totalFactors := f.ensureLUT(maxDim); totalFactors > 0 {
			lutC := lut.([]complex64)
			fft2dc = NewFFT2DC(rows, cols, lutC[:totalFactors])
		} else {
			fft2dc = NewFFT2DC(rows, cols)
		}
		fft2dc.Backward(dst, src)
		return
	}

	panic("FFT.Backward: rank > 2 not yet implemented")
}
