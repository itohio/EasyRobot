package dsp

import (
	"math"
	"math/cmplx"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// MeasurementResult holds various signal measurements
type MeasurementResult struct {
	RMS      float32
	Peak     float32
	Mean     float32
	Variance float32
	SNR      float32 // Signal-to-Noise Ratio
	THD      float32 // Total Harmonic Distortion (placeholder)
}

// FrequencyMeasurement holds frequency-domain measurements
type FrequencyMeasurement struct {
	Amplitude float32
	Phase     float32
	Frequency float32
}

// CorrelationResult holds correlation measurements
type CorrelationResult struct {
	Coefficient float32
	Lag         int
}

// Measurements provides signal measurement functions
type Measurements struct{}

// NewMeasurements creates a new measurement processor
func NewMeasurements() *Measurements {
	return &Measurements{}
}

// Measure1D performs comprehensive measurements on a 1D signal
func (m *Measurements) Measure1D(signal vecTypes.Vector) MeasurementResult {
	data := []float32(signal.View().(vec.Vector))
	n := len(data)

	if n == 0 {
		return MeasurementResult{}
	}

	// RMS
	rms := float32(0)
	sum := float32(0)
	sumSq := float32(0)
	peak := float32(0)

	for _, v := range data {
		sum += v
		sumSq += v * v
		if abs := float32(math.Abs(float64(v))); abs > peak {
			peak = abs
		}
	}

	rms = float32(math.Sqrt(float64(sumSq / float32(n))))
	mean := sum / float32(n)
	variance := (sumSq / float32(n)) - (mean * mean)

	return MeasurementResult{
		RMS:      rms,
		Peak:     peak,
		Mean:     mean,
		Variance: variance,
		SNR:      m.calculateSNR(data),
	}
}

// Measure2D performs comprehensive measurements on a 2D signal
func (m *Measurements) Measure2D(signal matTypes.Matrix) MeasurementResult {
	flat := signal.Flat()
	n := len(flat)

	if n == 0 {
		return MeasurementResult{}
	}

	// RMS
	rms := float32(0)
	sum := float32(0)
	sumSq := float32(0)
	peak := float32(0)

	for _, v := range flat {
		sum += v
		sumSq += v * v
		if abs := float32(math.Abs(float64(v))); abs > peak {
			peak = abs
		}
	}

	rms = float32(math.Sqrt(float64(sumSq / float32(n))))
	mean := sum / float32(n)
	variance := (sumSq / float32(n)) - (mean * mean)

	return MeasurementResult{
		RMS:      rms,
		Peak:     peak,
		Mean:     mean,
		Variance: variance,
		SNR:      m.calculateSNR(flat),
	}
}

// Goertzel performs Goertzel algorithm for single-frequency detection
func (m *Measurements) Goertzel(signal vecTypes.Vector, targetFreq float32, sampleRate float32) FrequencyMeasurement {
	data := []float32(signal.View().(vec.Vector))
	n := len(data)

	// Normalize frequency
	k := int(0.5 + float32(n)*targetFreq/sampleRate)
	if k >= n {
		k = n - 1
	}

	omega := 2 * math.Pi * float64(k) / float64(n)
	coeff := 2 * float32(math.Cos(omega))

	// Goertzel algorithm
	q1, q2 := float32(0), float32(0)
	for _, sample := range data {
		q0 := sample + coeff*q1 - q2
		q2 = q1
		q1 = q0
	}

	// Calculate magnitude and phase
	real := q1 - q2*float32(math.Cos(omega))
	imag := q2 * float32(math.Sin(omega))

	magnitude := float32(math.Sqrt(float64(real*real + imag*imag)))
	phase := float32(math.Atan2(float64(imag), float64(real)))

	// Normalize magnitude
	magnitude *= 2.0 / float32(n)

	return FrequencyMeasurement{
		Amplitude: magnitude,
		Phase:     phase,
		Frequency: targetFreq,
	}
}

// CrossCorrelate1D computes cross-correlation between two 1D signals
func (m *Measurements) CrossCorrelate1D(signal1, signal2 vecTypes.Vector) vecTypes.Vector {
	data1 := []float32(signal1.View().(vec.Vector))
	data2 := []float32(signal2.View().(vec.Vector))

	n1 := len(data1)
	n2 := len(data2)

	// Result length is n1 + n2 - 1
	resultLen := n1 + n2 - 1
	result := make([]float32, resultLen)

	// Compute cross-correlation
	for i := 0; i < resultLen; i++ {
		sum := float32(0)
		for j := 0; j < n1; j++ {
			k := i - j
			if k >= 0 && k < n2 {
				sum += data1[j] * data2[k]
			}
		}
		result[i] = sum
	}

	return vec.NewFrom(result...)
}

// CrossCorrelate2D computes 2D cross-correlation between two matrices
func (m *Measurements) CrossCorrelate2D(signal1, signal2 matTypes.Matrix) matTypes.Matrix {
	rows1, cols1 := signal1.Rows(), signal1.Cols()
	rows2, cols2 := signal2.Rows(), signal2.Cols()

	resultRows := rows1 + rows2 - 1
	resultCols := cols1 + cols2 - 1

	result := mat.New(resultRows, resultCols)

	// Compute 2D cross-correlation
	for i := 0; i < resultRows; i++ {
		for j := 0; j < resultCols; j++ {
			sum := float32(0)
			for k := 0; k < rows1; k++ {
				for l := 0; l < cols1; l++ {
					rowIdx := i - k
					colIdx := j - l
					if rowIdx >= 0 && rowIdx < rows2 && colIdx >= 0 && colIdx < cols2 {
						sum += signal1.Flat()[k*cols1+l] * signal2.Flat()[rowIdx*cols2+colIdx]
					}
				}
			}
			result[i][j] = sum
		}
	}

	return result
}

// FindPeakCorrelation finds the peak correlation and its position
func (m *Measurements) FindPeakCorrelation(correlation vecTypes.Vector) CorrelationResult {
	data := []float32(correlation.View().(vec.Vector))
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
func (m *Measurements) FindPeakCorrelation2D(correlation matTypes.Matrix) (float32, int, int) {
	rows, cols := correlation.Rows(), correlation.Cols()

	maxCorr := float32(-1e9)
	maxRow, maxCol := 0, 0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
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

// PhaseCorrelation2D computes phase correlation for shift estimation
func (m *Measurements) PhaseCorrelation2D(signal1, signal2 matTypes.Matrix) matTypes.Matrix {
	rows, cols := signal1.Rows(), signal1.Cols()
	fft := NewFFT2D(rows, cols)

	// FFT both signals
	s1FFT := mat.New(rows, cols)
	s2FFT := mat.New(rows, cols)
	fft.Forward(signal1, s1FFT)
	fft.Forward(signal2, s2FFT)

	// Compute cross-power spectrum
	crossPower := mat.New(rows, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			s1 := complex(s1FFT.Flat()[idx], 0) // Real FFT result
			s2 := complex(s2FFT.Flat()[idx], 0)
			cross := complex128(s1) * cmplx.Conj(complex128(s2))
			mag := cmplx.Abs(cross)
			if mag > 1e-10 {
				crossPower[i][j] = float32(cmplx.Phase(cross))
			}
		}
	}

	// Inverse FFT to get phase correlation
	result := mat.New(rows, cols)
	fft.Backward(crossPower, result)

	// Shift to center the zero-frequency component
	m.centerPhaseCorrelation(result)

	return result
}

// EstimateShift2D estimates the shift between two 2D signals using phase correlation
func (m *Measurements) EstimateShift2D(signal1, signal2 matTypes.Matrix) (shiftY, shiftX int) {
	phaseCorr := m.PhaseCorrelation2D(signal1, signal2)
	_, dy, dx := m.FindPeakCorrelation2D(phaseCorr)
	return dy, dx
}

// Helper functions

func (m *Measurements) calculateSNR(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}

	// Calculate signal power (variance)
	mean := float32(0)
	for _, v := range data {
		mean += v
	}
	mean /= float32(len(data))

	signalPower := float32(0)
	for _, v := range data {
		diff := v - mean
		signalPower += diff * diff
	}
	signalPower /= float32(len(data))

	// For SNR calculation, we assume noise is the high-frequency components
	// This is a simplified approach - in practice, you'd need a noise reference
	noisePower := signalPower * 0.01 // Assume 1% noise floor

	if noisePower > 0 {
		return 10 * float32(math.Log10(float64(signalPower/noisePower)))
	}
	return 0
}

func (m *Measurements) centerPhaseCorrelation(matrix matTypes.Matrix) {
	rows, cols := matrix.Rows(), matrix.Cols()
	centerRow := rows / 2
	centerCol := cols / 2

	// Create a copy for swapping
	temp := matrix.Clone()

	// Cast to concrete matrix type for indexing
	concreteMatrix := matrix.(mat.Matrix)
	concreteTemp := temp.(mat.Matrix)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			srcI := (i + centerRow) % rows
			srcJ := (j + centerCol) % cols
			concreteMatrix[i][j] = concreteTemp.Flat()[srcI*cols+srcJ]
		}
	}
}
