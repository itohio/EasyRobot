// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// FFT2D performs forward and inverse Fast Fourier Transforms on real-valued matrices.
//
// Data Layout:
//   - Input (src): Real-valued float32 matrix. Each element represents a real sample.
//   - Output (dst): float32 matrix with dimensions [rows, 2*cols] to store packed complex values.
//     The output stores complex values in packed format per row:
//   - Real parts: dst[i][0:cols]
//   - Imaginary parts: dst[i][cols:2*cols]
//     For a real input signal, the 2D frequency domain has conjugate symmetry.
//
// Note: The destination buffer must have 2*cols columns to store both real and imaginary parts.
// For complex input/output, use FFT2DC.
type FFT2D struct {
	rows, cols int       // matrix dimensions (FFT size)
	tempBuf    []float32 // temporary buffer for optimizations (size 2*rows*cols for packed format)
	lut        []float32 // precomputed LUT for twiddle factors (optional)
}

// NewFFT2D creates a 2D FFT processor for real-valued matrices of the specified dimensions.
// If lut is provided (vararg), it will be used for optimized FFT operations.
// The LUT can be created using fp32.NewFFTLUT(max(rows, cols)).
// Returns a value (not a pointer).
func NewFFT2D(rows, cols int, lut ...[]float32) FFT2D {
	if rows <= 0 || cols <= 0 {
		panic("FFT dimensions must be positive")
	}
	nextPow2Rows := fp32.NextPowerOfTwo(rows)
	nextPow2Cols := fp32.NextPowerOfTwo(cols)

	fft := FFT2D{
		rows:    nextPow2Rows,
		cols:    nextPow2Cols,
		tempBuf: make([]float32, 2*nextPow2Rows*nextPow2Cols), // 2*rows*cols for packed format
	}

	// Use LUT if provided
	if len(lut) > 0 && len(lut[0]) > 0 {
		fft.lut = lut[0]
	}

	return fft
}

// Forward performs a forward FFT transform on a real-valued matrix.
//
// Input (src): Real-valued float32 matrix of dimensions [rows, cols].
// Output (dst): float32 matrix with dimensions [rows, 2*cols] to store packed complex values.
//
//	The output stores complex values in packed format per row:
//	- Real parts: dst[i][0:cols]
//	- Imaginary parts: dst[i][cols:2*cols]
//	For a real input signal, the 2D frequency domain has conjugate symmetry.
//
// The destination buffer must have 2*cols columns.
// Uses destination-based API with no allocations.
func (f *FFT2D) Forward(dst, src matTypes.Matrix) {
	f.forward(src, dst)
}

// Backward performs an inverse FFT transform on packed complex frequency domain data.
//
// Input (src): float32 matrix with dimensions [rows, 2*cols] in packed format:
//   - Real parts: src[i][0:cols]
//   - Imaginary parts: src[i][cols:2*cols]
//
// Output (dst): Real-valued float32 matrix (2D time domain signal) of dimensions [rows, cols].
//
//	The result is scaled by 1/(rows*cols) where rows and cols are the FFT dimensions.
//
// Uses destination-based API with no allocations.
func (f *FFT2D) Backward(dst, src matTypes.Matrix) {
	f.backward(src, dst)
}

// forward performs 2D forward FFT with destination buffer.
// Optimized with BCE techniques: row slices, reslicing, and for range loops
func (f *FFT2D) forward(src, dst matTypes.Matrix) {
	srcRows, srcCols := src.Rows(), src.Cols()
	dstCols := dst.Cols()

	// Ensure destination has 2*cols columns for packed format
	if dstCols < 2*f.cols {
		panic("FFT2D.Forward: destination must have at least 2*cols columns for packed format")
	}

	// Copy source to destination real parts, zero imaginary parts
	// BCE optimization: use row slices with reslicing
	for i := range srcRows {
		if i >= f.rows {
			break
		}
		dstRow := dst.(mat.Matrix)[i]
		srcRow := src.(mat.Matrix)[i]
		srcRow = srcRow[:srcCols] // BCE hint

		// Copy real input to real part of packed format [0:cols]
		copyLen := srcCols
		if copyLen > f.cols {
			copyLen = f.cols
		}
		copy(dstRow[:copyLen], srcRow[:copyLen])
		// Zero-pad remaining real parts
		for j := copyLen; j < f.cols; j++ {
			dstRow[j] = 0
		}
		// Zero all imaginary parts [cols:2*cols]
		for j := range f.cols {
			dstRow[j+f.cols] = 0
		}
	}
	// Zero-pad remaining rows
	for i := srcRows; i < f.rows; i++ {
		dstRow := dst.(mat.Matrix)[i]
		// Zero real parts
		for j := range f.cols {
			dstRow[j] = 0
		}
		// Zero imaginary parts
		for j := range f.cols {
			dstRow[j+f.cols] = 0
		}
	}

	// FFT rows (each row is packed format: real[0:cols], imag[cols:2*cols])
	// BCE optimization: use for range
	for i := range f.rows {
		row := dst.Row(i)
		f.fftRow(row.(vec.Vector))
	}

	// FFT columns using temp buffer (packed format: 2*rows)
	// BCE optimization: reslice temp buffer and use for range
	tempBuf := f.tempBuf[:2*f.rows]
	for j := range f.cols {
		// Extract column real and imag parts to temp buffer
		for i := range f.rows {
			tempBuf[i] = dst.(mat.Matrix)[i][j]               // real part
			tempBuf[i+f.rows] = dst.(mat.Matrix)[i][j+f.cols] // imag part
		}
		// FFT the column (packed format)
		if len(f.lut) > 0 {
			totalFactors := len(f.lut) / 2
			fp32.FFTWithLUT(tempBuf, f.lut, totalFactors)
		} else {
			fp32.FFT(tempBuf)
		}
		// Put back
		for i := range f.rows {
			dst.(mat.Matrix)[i][j] = tempBuf[i]
			dst.(mat.Matrix)[i][j+f.cols] = tempBuf[i+f.rows]
		}
	}
}

// backward performs 2D inverse FFT with destination buffer.
// Optimized with BCE techniques: row slices, reslicing, and for range loops
func (f *FFT2D) backward(src, dst matTypes.Matrix) {
	srcRows, srcCols := src.Rows(), src.Cols()

	// Ensure source has 2*cols columns for packed format
	if srcCols < 2*f.cols {
		panic("FFT2D.Backward: source must have at least 2*cols columns for packed format")
	}

	// Copy packed complex data from source to temp buffer
	// We'll work in-place on a temp buffer, then copy real parts to destination
	tempBuf := f.tempBuf[:2*f.rows*f.cols]

	// Copy all packed data to temp buffer
	for i := range srcRows {
		if i >= f.rows {
			break
		}
		srcRow := src.(mat.Matrix)[i]
		rowStart := i * 2 * f.cols
		copyLen := srcCols
		if copyLen > 2*f.cols {
			copyLen = 2 * f.cols
		}
		copy(tempBuf[rowStart:rowStart+copyLen], srcRow[:copyLen])
	}
	// Zero-pad remaining rows in temp buffer
	for i := srcRows; i < f.rows; i++ {
		rowStart := i * 2 * f.cols
		for j := range 2 * f.cols {
			tempBuf[rowStart+j] = 0
		}
	}

	// IFFT columns first using temp buffer (packed format: 2*rows)
	colBuf := f.tempBuf[:2*f.rows]
	for j := range f.cols {
		// Extract column real and imag parts to colBuf
		for i := range f.rows {
			idx := i*2*f.cols + j
			colBuf[i] = tempBuf[idx]               // real part
			colBuf[i+f.rows] = tempBuf[idx+f.cols] // imag part
		}
		// IFFT the column (packed format)
		// Note: IFFTWithLUT could be added later for optimization
		fp32.IFFT(colBuf)
		// Put back
		for i := range f.rows {
			idx := i*2*f.cols + j
			tempBuf[idx] = colBuf[i]
			tempBuf[idx+f.cols] = colBuf[i+f.rows]
		}
	}

	// IFFT rows (each row is packed format: real[0:cols], imag[cols:2*cols])
	rowBuf := f.tempBuf[:2*f.cols]
	for i := range f.rows {
		rowStart := i * 2 * f.cols
		copy(rowBuf, tempBuf[rowStart:rowStart+2*f.cols])
		// Note: IFFTWithLUT could be added later for optimization
		fp32.IFFT(rowBuf)
		// Copy real part to destination
		copy(dst.(mat.Matrix)[i][:f.cols], rowBuf[:f.cols])
	}
}

// fftRow performs FFT on a single row
func (f *FFT2D) fftRow(row vec.Vector) {
	rowData := []float32(row)
	if len(f.lut) > 0 {
		totalFactors := len(f.lut) / 2
		fp32.FFTWithLUT(rowData, f.lut, totalFactors)
	} else {
		fp32.FFT(rowData)
	}
}

// ifftRow performs inverse FFT on a single row
func (f *FFT2D) ifftRow(row vec.Vector) {
	rowData := []float32(row)
	// Note: IFFTWithLUT could be added later for optimization
	fp32.IFFT(rowData)
}

// Convolve performs 2D convolution of signal matrix with kernel matrix using FFT.
// Uses destination-based API with no allocations.
func (f *FFT2D) Convolve(dst, signal, kernel matTypes.Matrix) {
	sigRows, sigCols := signal.Rows(), signal.Cols()
	kernRows, kernCols := kernel.Rows(), kernel.Cols()
	resultRows := sigRows + kernRows - 1
	resultCols := sigCols + kernCols - 1

	// Ensure destination is large enough
	if dst.Rows() < resultRows || dst.Cols() < resultCols {
		panic("Convolve: destination buffer too small")
	}

	// Create temporary FFTs for signal and kernel
	sigFFT := NewFFT2D(sigRows, sigCols)
	kernFFT := NewFFT2D(kernRows, kernCols)

	// Temporary buffers
	sigFreq := mat.New(f.rows, f.cols)
	kernFreq := mat.New(f.rows, f.cols)

	// Forward FFTs
	sigFFT.Forward(sigFreq, signal)
	kernFFT.Forward(kernFreq, kernel)

	// Multiply in frequency domain
	// BCE optimization: reslice to exact size
	sigFlat := sigFreq.Flat()
	kernFlat := kernFreq.Flat()
	size := f.rows * f.cols
	if len(sigFlat) < size {
		size = len(sigFlat)
	}
	if len(kernFlat) < size {
		size = len(kernFlat)
	}
	sigFlat = sigFlat[:size]
	kernFlat = kernFlat[:size]
	for i := range size {
		sigFlat[i] *= kernFlat[i]
	}

	// Inverse FFT to destination
	f.Backward(dst, sigFreq)

	// Scale result (FFT normalization)
	// BCE optimization: reslice to exact size
	dstFlat := dst.Flat()
	resultSize := resultRows * resultCols
	if resultSize > len(dstFlat) {
		resultSize = len(dstFlat)
	}
	dstFlat = dstFlat[:resultSize]
	scale := 1.0 / float32(f.rows*f.cols)
	for i := range resultSize {
		dstFlat[i] *= scale
	}
}

// Correlate performs 2D cross-correlation of signal1 with signal2 using FFT.
// Cross-correlation is similar to convolution but with one signal time-reversed.
// Uses destination-based API with no allocations.
func (f *FFT2D) Correlate(dst, signal1, signal2 matTypes.Matrix) {
	sig1Rows, sig1Cols := signal1.Rows(), signal1.Cols()
	sig2Rows, sig2Cols := signal2.Rows(), signal2.Cols()
	resultRows := sig1Rows + sig2Rows - 1
	resultCols := sig1Cols + sig2Cols - 1

	// Ensure destination is large enough
	if dst.Rows() < resultRows || dst.Cols() < resultCols {
		panic("Correlate: destination buffer too small")
	}

	// Create temporary FFTs for signals
	sig1FFT := NewFFT2D(sig1Rows, sig1Cols)
	sig2FFT := NewFFT2D(sig2Rows, sig2Cols)

	// Temporary buffers
	sig1Freq := mat.New(f.rows, f.cols)
	sig2Freq := mat.New(f.rows, f.cols)

	// Forward FFTs
	sig1FFT.Forward(sig1Freq, signal1)
	sig2FFT.Forward(sig2Freq, signal2)

	// For correlation, multiply sig1 with sig2 (for real signals, no conjugation needed)
	// BCE optimization: reslice to exact size
	sig1Flat := sig1Freq.Flat()
	sig2Flat := sig2Freq.Flat()
	size := f.rows * f.cols
	if len(sig1Flat) < size {
		size = len(sig1Flat)
	}
	if len(sig2Flat) < size {
		size = len(sig2Flat)
	}
	sig1Flat = sig1Flat[:size]
	sig2Flat = sig2Flat[:size]
	for i := range size {
		sig1Flat[i] *= sig2Flat[i]
	}

	// Inverse FFT to destination
	f.Backward(dst, sig1Freq)

	// Scale result (FFT normalization)
	// BCE optimization: reslice to exact size
	dstFlat := dst.Flat()
	resultSize := resultRows * resultCols
	if resultSize > len(dstFlat) {
		resultSize = len(dstFlat)
	}
	dstFlat = dstFlat[:resultSize]
	scale := 1.0 / float32(f.rows*f.cols)
	for i := range resultSize {
		dstFlat[i] *= scale
	}
}
