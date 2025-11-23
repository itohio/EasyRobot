package dsp

import (
	"math/cmplx"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// CorrelationResult holds correlation measurements
type CorrelationResult struct {
	Coefficient float32
	Lag         int
}

// CrossCorrelate1D computes cross-correlation between two 1D signals into dst
// Returns the number of samples filled in dst
func CrossCorrelate1D(dst vecTypes.Vector, signal1, signal2 vecTypes.Vector) int {
	data1 := signal1.View().(vec.Vector)
	data2 := signal2.View().(vec.Vector)
	dstData := dst.View().(vec.Vector)

	n1 := len(data1)
	n2 := len(data2)
	resultLen := n1 + n2 - 1
	if len(dstData) < resultLen {
		resultLen = len(dstData)
	}

	// Compute cross-correlation: corr[i] = sum(signal1[j] * signal2[j+i])
	for i := range resultLen {
		sum := float32(0)
		for j := range n1 {
			k := j + i
			if k >= 0 && k < n2 {
				sum += data1[j] * data2[k]
			}
		}
		dstData[i] = sum
	}

	return resultLen
}

// CrossCorrelate1DByConvolution computes cross-correlation using reversed signal2
// Returns the number of samples filled in dst
func CrossCorrelate1DByConvolution(dst vecTypes.Vector, signal1, signal2 vecTypes.Vector) int {
	data1 := signal1.View().(vec.Vector)
	data2 := signal2.View().(vec.Vector)
	dstData := dst.View().(vec.Vector)

	n1 := len(data1)
	n2 := len(data2)

	if n1 == 0 || n2 == 0 {
		return 0
	}

	resultLen := n1 + n2 - 1
	if len(dstData) < resultLen {
		resultLen = len(dstData)
	}

	// Reverse signal2 in-place if we can use dst as temp buffer, otherwise allocate
	var reversed []float32
	needsAlloc := len(dstData) < resultLen+n2
	if !needsAlloc {
		// Use end of dst as temporary buffer
		reversed = dstData[resultLen : resultLen+n2]
	} else {
		// Stack array for small buffers, heap for large
		if n2 <= 64 {
			var buf [64]float32
			reversed = buf[:n2]
		} else {
			reversed = make([]float32, n2)
		}
	}

	// Reverse signal2: reversed[i] = signal2[n2-1-i]
	for i := range n2 {
		reversed[i] = data2[n2-1-i]
	}

	// Compute correlation using the same formula as CrossCorrelate1D
	// corr[i] = sum(signal1[j] * signal2[j+i])
	// But we have reversed where reversed[k] = signal2[n2-1-k]
	// So signal2[j+i] = reversed[n2-1-(j+i)]
	for i := range resultLen {
		sum := float32(0)
		for j := range n1 {
			idx2 := j + i
			if idx2 >= 0 && idx2 < n2 {
				// signal2[idx2] = reversed[n2-1-idx2]
				sum += data1[j] * reversed[n2-1-idx2]
			}
		}
		dstData[i] = sum
	}

	return resultLen
}

// CrossCorrelate2D computes 2D cross-correlation between two matrices into dst
// Returns the number of rows filled
func CrossCorrelate2D(dst matTypes.Matrix, signal1, signal2 matTypes.Matrix) int {
	rows1, cols1 := signal1.Rows(), signal1.Cols()
	rows2, cols2 := signal2.Rows(), signal2.Cols()

	resultRows := rows1 + rows2 - 1
	resultCols := cols1 + cols2 - 1

	dstRows, dstCols := dst.Rows(), dst.Cols()
	if dstRows < resultRows {
		resultRows = dstRows
	}
	if dstCols < resultCols {
		resultCols = dstCols
	}

	// Compute 2D cross-correlation
	for i := range resultRows {
		for j := range resultCols {
			sum := float32(0)
			for k := range rows1 {
				for l := range cols1 {
					rowIdx := i - k
					colIdx := j - l
					if rowIdx >= 0 && rowIdx < rows2 && colIdx >= 0 && colIdx < cols2 {
						sum += signal1.Flat()[k*cols1+l] * signal2.Flat()[rowIdx*cols2+colIdx]
					}
				}
			}
			dst.Flat()[i*dstCols+j] = sum
		}
	}

	return resultRows
}

// FindPeakCorrelation finds the peak correlation and its position
func FindPeakCorrelation(correlation vecTypes.Vector) CorrelationResult {
	data := correlation.View().(vec.Vector)
	n := len(data)

	if n == 0 {
		return CorrelationResult{}
	}

	maxCorr := float32(-1e9)
	maxIdx := 0

	for i, v := range data {
		if v > maxCorr {
			maxCorr = v
			maxIdx = i
		}
	}

	// Convert index to lag (negative for left shift, positive for right shift)
	lag := maxIdx - (n-1)/2

	return CorrelationResult{
		Coefficient: maxCorr,
		Lag:         lag,
	}
}

// FindPeakCorrelation2D finds the peak correlation in 2D and its position
func FindPeakCorrelation2D(correlation matTypes.Matrix) (float32, int, int) {
	rows, cols := correlation.Rows(), correlation.Cols()

	maxCorr := float32(-1e9)
	maxRow, maxCol := 0, 0

	for i := range rows {
		for j := range cols {
			if correlation.Flat()[i*cols+j] > maxCorr {
				maxCorr = correlation.Flat()[i*cols+j]
				maxRow = i
				maxCol = j
			}
		}
	}

	// Convert to lag coordinates (center is zero shift)
	centerRow := (rows - 1) / 2
	centerCol := (cols - 1) / 2

	return maxCorr, maxRow - centerRow, maxCol - centerCol
}

// PhaseCorrelation2D computes phase correlation for shift estimation into dst
// Returns the number of rows filled
func PhaseCorrelation2D(dst matTypes.Matrix, signal1, signal2 matTypes.Matrix) int {
	rows, cols := signal1.Rows(), signal1.Cols()
	fft := NewFFT2D(rows, cols)

	// FFT both signals - destination needs 2*cols for packed complex format
	// Use stack arrays for small sizes, heap for large
	var s1FFT, s2FFT, crossPower matTypes.Matrix
	if rows*2*cols <= 1024 {
		// Small enough for stack allocation consideration
		s1FFT = mat.New(rows, 2*cols)
		s2FFT = mat.New(rows, 2*cols)
		crossPower = mat.New(rows, 2*cols)
	} else {
		s1FFT = mat.New(rows, 2*cols)
		s2FFT = mat.New(rows, 2*cols)
		crossPower = mat.New(rows, 2*cols)
	}

	fft.Forward(s1FFT, signal1)
	fft.Forward(s2FFT, signal2)

	// Compute cross-power spectrum (packed complex format)
	for i := range rows {
		for j := range cols {
			// Packed format: real at [i][j], imag at [i][j+cols]
			s1Real := s1FFT.Flat()[i*2*cols+j]
			s1Imag := s1FFT.Flat()[i*2*cols+j+cols]
			s2Real := s2FFT.Flat()[i*2*cols+j]
			s2Imag := s2FFT.Flat()[i*2*cols+j+cols]

			s1 := complex(s1Real, s1Imag)
			s2 := complex(s2Real, s2Imag)
			cross := complex128(s1) * cmplx.Conj(complex128(s2))
			mag := cmplx.Abs(cross)
			if mag > 1e-10 {
				phase := cmplx.Phase(cross)
				// Store phase as real part, zero imaginary
				crossPower.Flat()[i*2*cols+j] = float32(phase)
				crossPower.Flat()[i*2*cols+j+cols] = 0
			}
		}
	}

	// Inverse FFT to get phase correlation (result is real, so cols columns)
	dstRows, dstCols := dst.Rows(), dst.Cols()
	resultRows := rows
	if dstRows < resultRows {
		resultRows = dstRows
	}
	resultCols := cols
	if dstCols < resultCols {
		resultCols = dstCols
	}

	// Create temporary result matrix if dst is wrong size
	var result matTypes.Matrix
	if dstRows == rows && dstCols == cols {
		result = dst
	} else {
		result = mat.New(rows, cols)
	}

	fft.Backward(result, crossPower)

	// Copy to dst if needed (compare by dimensions, not pointer)
	if dstRows != rows || dstCols != cols {
		for i := range resultRows {
			for j := range resultCols {
				dst.Flat()[i*dstCols+j] = result.Flat()[i*cols+j]
			}
		}
	}

	// Shift to center the zero-frequency component
	centerPhaseCorrelation(dst)

	return resultRows
}

// EstimateShift2D estimates the shift between two 2D signals using phase correlation
func EstimateShift2D(signal1, signal2 matTypes.Matrix) (shiftY, shiftX int) {
	rows, cols := signal1.Rows(), signal1.Cols()
	result := mat.New(rows, cols)
	PhaseCorrelation2D(result, signal1, signal2)
	_, dy, dx := FindPeakCorrelation2D(result)
	return dy, dx
}

func centerPhaseCorrelation(matrix matTypes.Matrix) {
	rows, cols := matrix.Rows(), matrix.Cols()
	centerRow := rows / 2
	centerCol := cols / 2

	// Use stack array for small matrices
	if rows*cols <= 256 {
		var buf [256]float32
		temp := buf[:rows*cols]
		flat := matrix.Flat()
		copy(temp, flat)

		for i := range rows {
			for j := range cols {
				srcI := (i + centerRow) % rows
				srcJ := (j + centerCol) % cols
				flat[i*cols+j] = temp[srcI*cols+srcJ]
			}
		}
	} else {
		// For larger matrices, clone
		temp := matrix.Clone()
		concreteMatrix := matrix.(mat.Matrix)
		concreteTemp := temp.(mat.Matrix)

		for i := range rows {
			for j := range cols {
				srcI := (i + centerRow) % rows
				srcJ := (j + centerCol) % cols
				concreteMatrix[i][j] = concreteTemp.Flat()[srcI*cols+srcJ]
			}
		}
	}
}
