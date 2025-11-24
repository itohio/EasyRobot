// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/itohio/EasyRobot/x/math/primitive/c64"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// FFT2DC performs forward and inverse Fast Fourier Transforms on complex-valued 2D tensors.
//
// Data Layout:
//   - Input (src): Complex-valued Tensor with C64 data type and rank 2. Each element is a
//     complex64 value representing a complex sample (real + imaginary parts).
//   - Output (dst): Complex-valued Tensor with C64 data type. For forward FFT, the output
//     contains the full complex 2D frequency domain representation. Unlike real FFT, complex
//     FFT preserves both real and imaginary components throughout the transform.
//
// The tensor must be 2D (rank 2) and have C64 data type. The FFT dimensions are determined
// by the tensor shape, which will be padded to the next power of 2 if needed.
type FFT2DC struct {
	rows, cols int         // matrix dimensions (next power of 2)
	tempBuf    []complex64 // temporary buffer for optimizations
	lut        []complex64 // precomputed LUT for twiddle factors (optional)
}

// NewFFT2DC creates a 2D complex FFT processor for tensors of the specified dimensions.
// The dimensions will be rounded up to the next power of 2.
// If lut is provided (vararg), it will be used for optimized FFT operations.
// The LUT can be created using c64.NewFFTLUT(max(rows, cols)).
// Returns a value (not a pointer).
func NewFFT2DC(rows, cols int, lut ...[]complex64) FFT2DC {
	if rows <= 0 || cols <= 0 {
		panic("FFT dimensions must be positive")
	}
	nextPow2Rows := fp32.NextPowerOfTwo(rows)
	nextPow2Cols := fp32.NextPowerOfTwo(cols)

	fft := FFT2DC{
		rows:    nextPow2Rows,
		cols:    nextPow2Cols,
		tempBuf: make([]complex64, nextPow2Rows*nextPow2Cols),
	}

	// Use LUT if provided
	if len(lut) > 0 && len(lut[0]) > 0 {
		fft.lut = lut[0]
	}

	return fft
}

// Forward performs a forward FFT transform on a complex-valued 2D tensor.
//
// Input (src): Complex-valued Tensor with C64 data type and rank 2.
//
//	The tensor contains complex64 values representing complex samples.
//
// Output (dst): Complex-valued Tensor with C64 data type. The 2D FFT result contains
//
//	the full complex frequency domain representation. The output dimensions
//	match the FFT size (next power of 2 >= input dimensions).
//
// Panics if src or dst are not C64 type, not rank 2, or if dst is too small.
// Uses destination-based API with no allocations.
func (f *FFT2DC) Forward(dst, src types.Tensor) {
	f.forward(src, dst)
}

// Backward performs an inverse FFT transform on a complex-valued 2D tensor.
//
// Input (src): Complex-valued Tensor with C64 data type (2D frequency domain representation).
// Output (dst): Complex-valued Tensor with C64 data type (2D time domain signal).
//
//	The result is scaled by 1/(rows*cols) where rows and cols are the FFT dimensions.
//
// Panics if src or dst are not C64 type, not rank 2, or if dst is too small.
// Uses destination-based API with no allocations.
func (f FFT2DC) Backward(dst, src types.Tensor) {
	f.backward(src, dst)
}

// forward performs 2D forward complex FFT with destination buffer.
// Optimized with BCE techniques: reslicing and for range loops
func (f *FFT2DC) forward(src, dst types.Tensor) {
	// Validate input
	if src.DataType() != types.C64 {
		panic("FFT2DC.Forward: source must be C64 type")
	}
	if dst.DataType() != types.C64 {
		panic("FFT2DC.Forward: destination must be C64 type")
	}
	if src.Rank() != 2 {
		panic("FFT2DC.Forward: source must be rank 2")
	}
	if dst.Rank() != 2 {
		panic("FFT2DC.Forward: destination must be rank 2")
	}

	srcShape := src.Shape()
	srcRows, srcCols := srcShape[0], srcShape[1]
	srcData := types.GetTensorData[[]complex64](src)
	dstData := types.GetTensorData[[]complex64](dst)

	// Copy source to destination (handles padding)
	// BCE optimization: use row slices with reslicing
	for i := range srcRows {
		if i >= f.rows {
			break
		}
		srcRowStart := i * srcCols
		dstRowStart := i * f.cols

		// Copy row data
		copyLen := srcCols
		if copyLen > f.cols {
			copyLen = f.cols
		}
		copy(dstData[dstRowStart:dstRowStart+copyLen], srcData[srcRowStart:srcRowStart+copyLen])
		// Zero-pad remaining columns
		for j := copyLen; j < f.cols; j++ {
			dstData[dstRowStart+j] = 0
		}
	}
	// Zero-pad remaining rows
	for i := srcRows; i < f.rows; i++ {
		dstRowStart := i * f.cols
		for j := range f.cols {
			dstData[dstRowStart+j] = 0
		}
	}

	// FFT rows
	// BCE optimization: use for range
	tempRow := make([]complex64, f.cols)
	for i := range f.rows {
		rowStart := i * f.cols
		row := dstData[rowStart : rowStart+f.cols]
		row = row[:f.cols] // BCE hint
		copy(tempRow, row)
		if len(f.lut) > 0 {
			totalFactors := len(f.lut)
			c64.FFTWithLUT(tempRow, f.lut, totalFactors)
		} else {
			c64.FFT(tempRow)
		}
		copy(row, tempRow)
	}

	// FFT columns using temp buffer
	// BCE optimization: reslice temp buffer and use for range
	tempBuf := f.tempBuf[:f.rows]
	for j := range f.cols {
		// Extract column to temp buffer
		for i := range f.rows {
			tempBuf[i] = dstData[i*f.cols+j]
		}
		// FFT the column
		if len(f.lut) > 0 {
			totalFactors := len(f.lut)
			c64.FFTWithLUT(tempBuf, f.lut, totalFactors)
		} else {
			c64.FFT(tempBuf)
		}
		// Put back
		for i := range f.rows {
			dstData[i*f.cols+j] = tempBuf[i]
		}
	}
}

// backward performs 2D inverse complex FFT with destination buffer.
// Optimized with BCE techniques: reslicing and for range loops
func (f *FFT2DC) backward(src, dst types.Tensor) {
	// Validate input
	if src.DataType() != types.C64 {
		panic("FFT2DC.Backward: source must be C64 type")
	}
	if dst.DataType() != types.C64 {
		panic("FFT2DC.Backward: destination must be C64 type")
	}
	if src.Rank() != 2 {
		panic("FFT2DC.Backward: source must be rank 2")
	}
	if dst.Rank() != 2 {
		panic("FFT2DC.Backward: destination must be rank 2")
	}

	srcShape := src.Shape()
	srcRows, srcCols := srcShape[0], srcShape[1]
	srcData := types.GetTensorData[[]complex64](src)
	dstData := types.GetTensorData[[]complex64](dst)

	// Copy source to destination
	// BCE optimization: use row slices with reslicing
	for i := range srcRows {
		if i >= f.rows {
			break
		}
		srcRowStart := i * srcCols
		dstRowStart := i * f.cols

		copyLen := srcCols
		if copyLen > f.cols {
			copyLen = f.cols
		}
		copy(dstData[dstRowStart:dstRowStart+copyLen], srcData[srcRowStart:srcRowStart+copyLen])
	}

	// IFFT columns first using temp buffer
	// BCE optimization: reslice temp buffer and use for range
	tempBuf := f.tempBuf[:f.rows]
	tempRow := make([]complex64, f.cols)
	for j := range f.cols {
		// Extract column to temp buffer
		for i := range f.rows {
			tempBuf[i] = dstData[i*f.cols+j]
		}
		// IFFT the column
		if len(f.lut) > 0 {
			totalFactors := len(f.lut)
			c64.IFFTWithLUT(tempBuf, f.lut, totalFactors)
		} else {
			c64.IFFT(tempBuf)
		}
		// Put back
		for i := range f.rows {
			dstData[i*f.cols+j] = tempBuf[i]
		}
	}

	// IFFT rows
	// BCE optimization: use for range
	for i := range f.rows {
		rowStart := i * f.cols
		row := dstData[rowStart : rowStart+f.cols]
		row = row[:f.cols] // BCE hint
		copy(tempRow, row)
		if len(f.lut) > 0 {
			totalFactors := len(f.lut)
			c64.IFFTWithLUT(tempRow, f.lut, totalFactors)
		} else {
			c64.IFFT(tempRow)
		}
		copy(row, tempRow)
	}

}

// Convolve performs 2D convolution of signal matrix with kernel matrix using complex FFT.
// Uses destination-based API with no allocations.
func (f FFT2DC) Convolve(dst, signal, kernel types.Tensor) {
	sigShape := signal.Shape()
	sigRows, sigCols := sigShape[0], sigShape[1]
	kernShape := kernel.Shape()
	kernRows, kernCols := kernShape[0], kernShape[1]
	resultRows := sigRows + kernRows - 1
	resultCols := sigCols + kernCols - 1

	// Ensure destination is large enough
	dstShape := dst.Shape()
	if dstShape[0] < resultRows || dstShape[1] < resultCols {
		panic("Convolve: destination buffer too small")
	}

	// Reuse f if sizes match (benefits from LUT if present), otherwise create new instances
	sigRowsPow2 := fp32.NextPowerOfTwo(sigRows)
	sigColsPow2 := fp32.NextPowerOfTwo(sigCols)
	kernRowsPow2 := fp32.NextPowerOfTwo(kernRows)
	kernColsPow2 := fp32.NextPowerOfTwo(kernCols)

	var sigFFT, kernFFT FFT2DC
	if sigRowsPow2 == f.rows && sigColsPow2 == f.cols {
		sigFFT = f
	} else {
		sigFFT = NewFFT2DC(sigRows, sigCols)
	}
	if kernRowsPow2 == f.rows && kernColsPow2 == f.cols {
		kernFFT = f
	} else {
		kernFFT = NewFFT2DC(kernRows, kernCols)
	}

	// Temporary buffers
	sigFreq := tensor.New(types.C64, shape.NewShape(f.rows, f.cols))
	kernFreq := tensor.New(types.C64, shape.NewShape(f.rows, f.cols))
	defer sigFreq.Release()

	// Forward FFTs
	sigFFT.Forward(sigFreq, signal)
	kernFFT.Forward(kernFreq, kernel)

	// Multiply in frequency domain
	// BCE optimization: reslice to exact size
	sigFreqData := types.GetTensorData[[]complex64](sigFreq)
	kernFreqData := types.GetTensorData[[]complex64](kernFreq)
	size := f.rows * f.cols
	if len(sigFreqData) < size {
		size = len(sigFreqData)
	}
	if len(kernFreqData) < size {
		size = len(kernFreqData)
	}
	sigFreqData = sigFreqData[:size]
	kernFreqData = kernFreqData[:size]
	for i := range size {
		sigFreqData[i] *= kernFreqData[i]
	}

	// Release kernFreq as it's no longer needed
	kernFreq.Release()

	// Inverse FFT to destination
	f.Backward(dst, sigFreq)

	dstData := types.GetTensorData[[]complex64](dst)
	resultSize := resultRows * resultCols
	if resultSize > len(dstData) {
		resultSize = len(dstData)
	}
	for i := resultSize; i < len(dstData); i++ {
		dstData[i] = 0
	}
}

// Correlate performs 2D cross-correlation of signal1 with signal2 using complex FFT.
// Cross-correlation is similar to convolution but with one signal conjugated.
// Uses destination-based API with no allocations.
func (f *FFT2DC) Correlate(dst, signal1, signal2 types.Tensor) {
	sig1Shape := signal1.Shape()
	sig1Rows, sig1Cols := sig1Shape[0], sig1Shape[1]
	sig2Shape := signal2.Shape()
	sig2Rows, sig2Cols := sig2Shape[0], sig2Shape[1]
	resultRows := sig1Rows + sig2Rows - 1
	resultCols := sig1Cols + sig2Cols - 1

	// Ensure destination is large enough
	dstShape := dst.Shape()
	if dstShape[0] < resultRows || dstShape[1] < resultCols {
		panic("Correlate: destination buffer too small")
	}

	// Create temporary FFTs for signals
	sig1FFT := NewFFT2DC(sig1Rows, sig1Cols)
	sig2FFT := NewFFT2DC(sig2Rows, sig2Cols)

	// Temporary buffers
	sig1Freq := tensor.New(types.C64, shape.NewShape(f.rows, f.cols))
	sig2Freq := tensor.New(types.C64, shape.NewShape(f.rows, f.cols))

	// Forward FFTs
	sig1FFT.Forward(sig1Freq, signal1)
	sig2FFT.Forward(sig2Freq, signal2)

	// For correlation, multiply sig1 with conjugate of sig2 (for complex signals)
	// BCE optimization: reslice to exact size
	sig1FreqData := types.GetTensorData[[]complex64](sig1Freq)
	sig2FreqData := types.GetTensorData[[]complex64](sig2Freq)
	size := f.rows * f.cols
	if len(sig1FreqData) < size {
		size = len(sig1FreqData)
	}
	if len(sig2FreqData) < size {
		size = len(sig2FreqData)
	}
	sig1FreqData = sig1FreqData[:size]
	sig2FreqData = sig2FreqData[:size]
	for i := range size {
		// Correlation: multiply by conjugate
		sig1FreqData[i] *= complex64(complex(real(sig2FreqData[i]), -imag(sig2FreqData[i])))
	}

	// Inverse FFT to destination
	f.Backward(dst, sig1Freq)

	dstData := types.GetTensorData[[]complex64](dst)
	resultSize := resultRows * resultCols
	if resultSize > len(dstData) {
		resultSize = len(dstData)
	}
	for i := resultSize; i < len(dstData); i++ {
		dstData[i] = 0
	}
}
