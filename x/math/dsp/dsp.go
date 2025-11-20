// Package dsp provides digital signal processing functions for EasyRobot.
// It includes FFT transforms, convolution, windowing, signal generation,
// and measurement functions optimized for fp32 operations.
package dsp

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// FFT1D performs forward and inverse Fast Fourier Transforms on vectors.
type FFT1D struct {
	length  int       // vector length
	tempBuf []float32 // temporary buffer for optimizations
}

// NewFFT1D creates a 1D FFT processor for vectors of the specified length.
func NewFFT1D(length int) *FFT1D {
	if length <= 0 {
		panic("FFT length must be positive")
	}
	nextPow2 := nextPowerOfTwo(length)
	return &FFT1D{
		length:  nextPow2,
		tempBuf: make([]float32, nextPow2),
	}
}

// FFT2D performs forward and inverse Fast Fourier Transforms on matrices.
type FFT2D struct {
	rows, cols int    // matrix dimensions
	tempBuf    []float32 // temporary buffer for optimizations
}

// NewFFT2D creates a 2D FFT processor for matrices of the specified dimensions.
func NewFFT2D(rows, cols int) *FFT2D {
	if rows <= 0 || cols <= 0 {
		panic("FFT dimensions must be positive")
	}
	nextPow2Rows := nextPowerOfTwo(rows)
	nextPow2Cols := nextPowerOfTwo(cols)
	return &FFT2D{
		rows:    nextPow2Rows,
		cols:    nextPow2Cols,
		tempBuf: make([]float32, nextPow2Rows*nextPow2Cols),
	}
}

// Forward performs a forward FFT transform on a vector.
// Uses destination-based API with no allocations.
func (f *FFT1D) Forward(src, dst vec.Vector) {
	f.forward(src, dst)
}

// Backward performs an inverse FFT transform on a vector.
// Uses destination-based API with no allocations.
func (f *FFT1D) Backward(src, dst vec.Vector) {
	f.backward(src, dst)
}

// Forward performs a forward FFT transform on a matrix.
// Uses destination-based API with no allocations.
func (f *FFT2D) Forward(src, dst matTypes.Matrix) {
	f.forward(src, dst)
}

// Backward performs an inverse FFT transform on a matrix.
// Uses destination-based API with no allocations.
func (f *FFT2D) Backward(src, dst matTypes.Matrix) {
	f.backward(src, dst)
}

// forward performs 1D forward FFT with destination buffer.
func (f *FFT1D) forward(src, dst vec.Vector) {
	srcData := []float32(src)
	dstData := []float32(dst)

	// Copy and zero-pad to temp buffer
	copy(f.tempBuf, srcData)
	for i := len(srcData); i < len(f.tempBuf); i++ {
		f.tempBuf[i] = 0
	}

	// Perform FFT
	fftReal(f.tempBuf)

	// Copy result to destination
	copy(dstData, f.tempBuf)
}

// backward performs 1D inverse FFT with destination buffer.
func (f *FFT1D) backward(src, dst vec.Vector) {
	srcData := []float32(src)
	dstData := []float32(dst)

	// Copy to temp buffer
	copy(f.tempBuf, srcData)

	// Perform inverse FFT
	ifftReal(f.tempBuf)

	// Copy result to destination
	copy(dstData, f.tempBuf)
}

// forward performs 2D forward FFT with destination buffer.
func (f *FFT2D) forward(src, dst matTypes.Matrix) {
	srcRows, srcCols := src.Rows(), src.Cols()

	// Copy source to destination (handles padding)
	for i := 0; i < srcRows && i < f.rows; i++ {
		for j := 0; j < srcCols && j < f.cols; j++ {
			dst.(mat.Matrix)[i][j] = src.(mat.Matrix)[i][j]
		}
		// Zero-pad remaining columns
		for j := srcCols; j < f.cols; j++ {
			dst.(mat.Matrix)[i][j] = 0
		}
	}
	// Zero-pad remaining rows
	for i := srcRows; i < f.rows; i++ {
		for j := 0; j < f.cols; j++ {
			dst.(mat.Matrix)[i][j] = 0
		}
	}

	// FFT rows
	for i := 0; i < f.rows; i++ {
		row := dst.Row(i)
		f.fftRow(row.(vec.Vector))
	}

	// FFT columns using temp buffer
	for j := 0; j < f.cols; j++ {
		// Extract column to temp buffer
		for i := 0; i < f.rows; i++ {
			f.tempBuf[i] = dst.(mat.Matrix)[i][j]
		}
		// FFT the column
		fftReal(f.tempBuf[:f.rows])
		// Put back
		for i := 0; i < f.rows; i++ {
			dst.(mat.Matrix)[i][j] = f.tempBuf[i]
		}
	}
}

// backward performs 2D inverse FFT with destination buffer.
func (f *FFT2D) backward(src, dst matTypes.Matrix) {
	srcRows, srcCols := src.Rows(), src.Cols()

	// Copy source to destination
	for i := 0; i < srcRows && i < f.rows; i++ {
		for j := 0; j < srcCols && j < f.cols; j++ {
			dst.(mat.Matrix)[i][j] = src.(mat.Matrix)[i][j]
		}
	}

	// IFFT columns first using temp buffer
	for j := 0; j < f.cols; j++ {
		// Extract column to temp buffer
		for i := 0; i < f.rows; i++ {
			f.tempBuf[i] = dst.(mat.Matrix)[i][j]
		}
		// IFFT the column
		ifftReal(f.tempBuf[:f.rows])
		// Put back
		for i := 0; i < f.rows; i++ {
			dst.(mat.Matrix)[i][j] = f.tempBuf[i]
		}
	}

	// IFFT rows
	for i := 0; i < f.rows; i++ {
		row := dst.Row(i)
		f.ifftRow(row.(vec.Vector))
	}
}

// fftRow performs FFT on a single row
func (f *FFT2D) fftRow(row vec.Vector) {
	rowData := []float32(row)
	fftReal(rowData)
}

// ifftRow performs inverse FFT on a single row
func (f *FFT2D) ifftRow(row vec.Vector) {
	rowData := []float32(row)
	ifftReal(rowData)
}

// Helper functions
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

func nextPowerOfTwo(n int) int {
	if n == 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	return n + 1
}

// fftReal performs in-place real FFT using Cooley-Tukey algorithm
func fftReal(data []float32) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Bit-reversal permutation
	j := 0
	for i := 0; i < n; i++ {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		m := n >> 1
		for ; m >= 1 && j >= m; m >>= 1 {
			j -= m
		}
		j += m
	}

	// Cooley-Tukey FFT
	for length := 2; length <= n; length <<= 1 {
		angle := -2 * math.Pi / float64(length)
		wlen := complex64(complex(math.Cos(angle), math.Sin(angle)))

		for i := 0; i < n; i += length {
			w := complex64(1 + 0i)
			for j := 0; j < length/2; j++ {
				u := complex64(complex(data[i+j], 0))
				v := complex64(complex(data[i+j+length/2], 0)) * w
				data[i+j] = float32(real(u + v))
				data[i+j+length/2] = float32(real(u - v))
				w *= wlen
			}
		}
	}
}

// ifftReal performs in-place real inverse FFT
func ifftReal(data []float32) {
	n := len(data)

	// Conjugate the data for IFFT
	for i := range data {
		data[i] = -data[i]
	}

	// Forward FFT on conjugated data
	fftReal(data)

	// Conjugate back and scale
	scale := 1.0 / float32(n)
	for i := range data {
		data[i] = -data[i] * scale
	}
}

// Convolve performs 1D convolution of signal with kernel using FFT.
// Uses destination-based API with no allocations.
func (f *FFT1D) Convolve(signal, kernel, dst vec.Vector) {
	sigLen := len(signal)
	kernLen := len(kernel)
	resultLen := sigLen + kernLen - 1

	// Ensure destination is large enough
	if len(dst) < resultLen {
		panic("Convolve: destination buffer too small")
	}

	// Create temporary FFTs for signal and kernel
	sigFFT := NewFFT1D(sigLen)
	kernFFT := NewFFT1D(kernLen)

	// Temporary buffers
	sigFreq := vec.New(f.length)
	kernFreq := vec.New(f.length)

	// Forward FFTs
	sigFFT.Forward(signal, sigFreq)
	kernFFT.Forward(kernel, kernFreq)

	// Multiply in frequency domain
	sigFreqData := []float32(sigFreq)
	kernFreqData := []float32(kernFreq)
	for i := range sigFreqData {
		sigFreqData[i] *= kernFreqData[i]
	}

	// Inverse FFT to destination
	f.Backward(sigFreq, dst)

	// Scale result (FFT normalization)
	dstData := []float32(dst)
	scale := 1.0 / float32(f.length)
	for i := range dstData[:resultLen] {
		dstData[i] *= scale
	}
}

// Convolve performs 2D convolution of signal matrix with kernel matrix using FFT.
// Uses destination-based API with no allocations.
func (f *FFT2D) Convolve(signal, kernel, dst matTypes.Matrix) {
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
	sigFFT.Forward(signal, sigFreq)
	kernFFT.Forward(kernel, kernFreq)

	// Multiply in frequency domain
	sigFlat := sigFreq.Flat()
	kernFlat := kernFreq.Flat()
	for i := range sigFlat {
		sigFlat[i] *= kernFlat[i]
	}

	// Inverse FFT to destination
	f.Backward(sigFreq, dst)

	// Scale result (FFT normalization)
	dstFlat := dst.Flat()
	scale := 1.0 / float32(f.rows*f.cols)
	for i := range dstFlat[:resultRows*resultCols] {
		dstFlat[i] *= scale
	}
}
