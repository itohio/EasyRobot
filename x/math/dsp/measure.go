package dsp

import (
	"math"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
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

// Measure1D performs comprehensive measurements on a 1D signal
func Measure1D(signal vecTypes.Vector) MeasurementResult {
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
		if abs := math32.Abs(v); abs > peak {
			peak = abs
		}
	}

	rms = math32.Sqrt(sumSq / float32(n))
	mean := sum / float32(n)
	variance := (sumSq / float32(n)) - (mean * mean)

	return MeasurementResult{
		RMS:      rms,
		Peak:     peak,
		Mean:     mean,
		Variance: variance,
		SNR:      calculateSNR(data),
	}
}

// Measure2D performs comprehensive measurements on a 2D signal
func Measure2D(signal matTypes.Matrix) MeasurementResult {
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
		if abs := math32.Abs(v); abs > peak {
			peak = abs
		}
	}

	rms = math32.Sqrt(sumSq / float32(n))
	mean := sum / float32(n)
	variance := (sumSq / float32(n)) - (mean * mean)

	return MeasurementResult{
		RMS:      rms,
		Peak:     peak,
		Mean:     mean,
		Variance: variance,
		SNR:      calculateSNR(flat),
	}
}

// Goertzel performs Goertzel algorithm for single-frequency detection
func Goertzel(signal vecTypes.Vector, targetFreq float32, sampleRate float32) FrequencyMeasurement {
	data := []float32(signal.View().(vec.Vector))
	n := len(data)

	// Normalize frequency
	k := int(0.5 + float32(n)*targetFreq/sampleRate)
	if k >= n {
		k = n - 1
	}

	omega := 2 * math32.Pi * float32(k) / float32(n)
	coeff := 2 * math32.Cos(omega)

	// Goertzel algorithm - standard implementation
	// s_prev and s_prev2 track the filter state
	sPrev, sPrev2 := float32(0), float32(0)
	for _, sample := range data {
		s := sample + coeff*sPrev - sPrev2
		sPrev2 = sPrev
		sPrev = s
	}

	// Final step: X[k] = s_prev*W^(-k) - s_prev2
	// Where W = e^(-j*2*pi*k/N), so W^(-k) = e^(j*2*pi*k/N) = cos(omega) + j*sin(omega)
	// X[k] = s_prev*(cos(omega) + j*sin(omega)) - s_prev2
	//      = (s_prev*cos(omega) - s_prev2) + j*(s_prev*sin(omega))
	real := sPrev*math32.Cos(omega) - sPrev2
	imag := sPrev * math32.Sin(omega)

	magnitude := math32.Sqrt(real*real + imag*imag)
	phase := math32.Atan2(imag, real)

	// Normalize magnitude: for a pure tone with amplitude A, DFT magnitude is A*N/2
	// So to recover A, we multiply by 2/N
	magnitude *= 2.0 / float32(n)

	return FrequencyMeasurement{
		Amplitude: magnitude,
		Phase:     phase,
		Frequency: targetFreq,
	}
}

// Helper functions

func calculateSNR(data []float32) float32 {
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
		return 10 * math32.Log10(signalPower/noisePower)
	}
	return 0
}

// RMSStatistics holds MIN/MAX/AVG RMS values in dB
type RMSStatistics struct {
	MinRMSDb float64
	MaxRMSDb float64
	AvgRMSDb float64
}

// CalculateRMSStatistics calculates MIN/MAX/AVG RMS statistics over windows
// Returns statistics in dB. Works on vectors with zero-copy extraction.
func CalculateRMSStatistics(signal vecTypes.Vector) RMSStatistics {
	data := signal.View().(vec.Vector)
	n := len(data)

	if n == 0 {
		return RMSStatistics{
			MinRMSDb: -math.Inf(1),
			MaxRMSDb: -math.Inf(1),
			AvgRMSDb: -math.Inf(1),
		}
	}

	// Calculate overall RMS for AVG
	var sumSquares float64
	for _, v := range data {
		sumSquares += float64(v * v)
	}
	avgRMS := math32.Sqrt(float32(sumSquares / float64(n)))

	// Calculate RMS over windows for MIN/MAX
	windowSize := n / 100 // 100 windows
	if windowSize < 1 {
		windowSize = 1
	}
	var minRMS, maxRMS float32 = avgRMS, avgRMS
	for i := 0; i < n; i += windowSize {
		end := i + windowSize
		if end > n {
			end = n
		}
		var windowSumSquares float64
		for j := i; j < end; j++ {
			windowSumSquares += float64(data[j] * data[j])
		}
		windowRMS := math32.Sqrt(float32(windowSumSquares / float64(end-i)))
		if windowRMS < minRMS {
			minRMS = windowRMS
		}
		if windowRMS > maxRMS {
			maxRMS = windowRMS
		}
	}

	// Convert to dB (handle zero case)
	avgRMSDb := -math.Inf(1)
	if avgRMS > 0 {
		avgRMSDb = 20 * math.Log10(float64(avgRMS)) // Keep math for float64
	}
	minRMSDb := -math.Inf(1)
	if minRMS > 0 {
		minRMSDb = 20 * math.Log10(float64(minRMS))
	}
	maxRMSDb := -math.Inf(1)
	if maxRMS > 0 {
		maxRMSDb = 20 * math.Log10(float64(maxRMS))
	}

	return RMSStatistics{
		MinRMSDb: minRMSDb,
		MaxRMSDb: maxRMSDb,
		AvgRMSDb: avgRMSDb,
	}
}

// Measure1DC performs comprehensive measurements on a 1D complex signal
func Measure1DC(signal types.Tensor) MeasurementResult {
	if signal.DataType() != types.C64 {
		panic("Measure1DC: signal must be C64 type")
	}
	if signal.Rank() != 1 {
		panic("Measure1DC: signal must be rank 1")
	}

	data := types.GetTensorData[[]complex64](signal)
	n := len(data)

	if n == 0 {
		return MeasurementResult{}
	}

	// For complex signals, use magnitude for measurements
	var sumMag, sumMagSq float32
	peak := float32(0)

	for _, v := range data {
		realVal := real(v)
		imagVal := imag(v)
		mag := math32.Sqrt(realVal*realVal + imagVal*imagVal)
		sumMag += mag
		sumMagSq += mag * mag
		if mag > peak {
			peak = mag
		}
	}

	rms := math32.Sqrt(sumMagSq / float32(n))
	mean := sumMag / float32(n)
	variance := (sumMagSq / float32(n)) - (mean * mean)

	// Convert to real array for SNR calculation
	realData := make([]float32, n)
	for i, v := range data {
		realData[i] = math32.Sqrt(real(v)*real(v) + imag(v)*imag(v))
	}

	return MeasurementResult{
		RMS:      rms,
		Peak:     peak,
		Mean:     mean,
		Variance: variance,
		SNR:      calculateSNR(realData),
	}
}

// Measure2DC performs comprehensive measurements on a 2D complex signal
func Measure2DC(signal types.Tensor) MeasurementResult {
	if signal.DataType() != types.C64 {
		panic("Measure2DC: signal must be C64 type")
	}
	if signal.Rank() != 2 {
		panic("Measure2DC: signal must be rank 2")
	}

	data := types.GetTensorData[[]complex64](signal)
	n := len(data)

	if n == 0 {
		return MeasurementResult{}
	}

	// For complex signals, use magnitude for measurements
	var sumMag, sumMagSq float32
	peak := float32(0)

	for _, v := range data {
		realVal := real(v)
		imagVal := imag(v)
		mag := math32.Sqrt(realVal*realVal + imagVal*imagVal)
		sumMag += mag
		sumMagSq += mag * mag
		if mag > peak {
			peak = mag
		}
	}

	rms := math32.Sqrt(sumMagSq / float32(n))
	mean := sumMag / float32(n)
	variance := (sumMagSq / float32(n)) - (mean * mean)

	// Convert to real array for SNR calculation
	realData := make([]float32, n)
	for i, v := range data {
		realData[i] = math32.Sqrt(real(v)*real(v) + imag(v)*imag(v))
	}

	return MeasurementResult{
		RMS:      rms,
		Peak:     peak,
		Mean:     mean,
		Variance: variance,
		SNR:      calculateSNR(realData),
	}
}

// GoertzelC performs Goertzel algorithm for single-frequency detection on complex signal
func GoertzelC(signal types.Tensor, targetFreq float32, sampleRate float32) FrequencyMeasurement {
	if signal.DataType() != types.C64 {
		panic("GoertzelC: signal must be C64 type")
	}
	if signal.Rank() != 1 {
		panic("GoertzelC: signal must be rank 1")
	}

	data := types.GetTensorData[[]complex64](signal)
	n := len(data)

	// Normalize frequency
	k := int(0.5 + float32(n)*targetFreq/sampleRate)
	if k >= n {
		k = n - 1
	}

	omega := 2 * math32.Pi * float32(k) / float32(n)
	coeff := 2 * math32.Cos(omega)

	// Goertzel algorithm for complex signals
	// s_prev and s_prev2 track the filter state (complex)
	sPrev, sPrev2 := complex64(0), complex64(0)
	for _, sample := range data {
		coeffComplex := complex64(complex(float32(coeff), 0))
		s := sample + coeffComplex*sPrev - sPrev2
		sPrev2 = sPrev
		sPrev = s
	}

	// Final step: X[k] = s_prev*W^(-k) - s_prev2
	// Where W = e^(-j*2*pi*k/N), so W^(-k) = e^(j*2*pi*k/N) = cos(omega) + j*sin(omega)
	w := complex64(complex(math32.Cos(omega), math32.Sin(omega)))
	result := sPrev*w - sPrev2

	magnitude := math32.Sqrt(real(result)*real(result) + imag(result)*imag(result))
	phase := math32.Atan2(imag(result), real(result))

	// Normalize magnitude: for a pure tone with amplitude A, DFT magnitude is A*N/2
	// So to recover A, we multiply by 2/N
	magnitude *= 2.0 / float32(n)

	return FrequencyMeasurement{
		Amplitude: magnitude,
		Phase:     phase,
		Frequency: targetFreq,
	}
}

// CalculateRMSStatisticsC calculates MIN/MAX/AVG RMS statistics over windows for complex signals
// Returns statistics in dB. Works on complex tensors.
func CalculateRMSStatisticsC(signal types.Tensor) RMSStatistics {
	if signal.DataType() != types.C64 {
		panic("CalculateRMSStatisticsC: signal must be C64 type")
	}
	if signal.Rank() != 1 {
		panic("CalculateRMSStatisticsC: signal must be rank 1")
	}

	data := types.GetTensorData[[]complex64](signal)
	n := len(data)

	if n == 0 {
		return RMSStatistics{
			MinRMSDb: -math.Inf(1),
			MaxRMSDb: -math.Inf(1),
			AvgRMSDb: -math.Inf(1),
		}
	}

	// Calculate overall RMS for AVG (using magnitude)
	var sumSquares float64
	for _, v := range data {
		realVal := real(v)
		imagVal := imag(v)
		mag := float64(realVal*realVal + imagVal*imagVal)
		sumSquares += mag
	}
	avgRMS := math32.Sqrt(float32(sumSquares / float64(n)))

	// Calculate RMS over windows for MIN/MAX
	windowSize := n / 100 // 100 windows
	if windowSize < 1 {
		windowSize = 1
	}
	var minRMS, maxRMS float32 = avgRMS, avgRMS
	for i := 0; i < n; i += windowSize {
		end := i + windowSize
		if end > n {
			end = n
		}
		var windowSumSquares float64
		for j := i; j < end; j++ {
			realVal := real(data[j])
			imagVal := imag(data[j])
			mag := float64(realVal*realVal + imagVal*imagVal)
			windowSumSquares += mag
		}
		windowRMS := math32.Sqrt(float32(windowSumSquares / float64(end-i)))
		if windowRMS < minRMS {
			minRMS = windowRMS
		}
		if windowRMS > maxRMS {
			maxRMS = windowRMS
		}
	}

	// Convert to dB (handle zero case)
	avgRMSDb := -math.Inf(1)
	if avgRMS > 0 {
		avgRMSDb = 20 * math.Log10(float64(avgRMS))
	}
	minRMSDb := -math.Inf(1)
	if minRMS > 0 {
		minRMSDb = 20 * math.Log10(float64(minRMS))
	}
	maxRMSDb := -math.Inf(1)
	if maxRMS > 0 {
		maxRMSDb = 20 * math.Log10(float64(maxRMS))
	}

	return RMSStatistics{
		MinRMSDb: minRMSDb,
		MaxRMSDb: maxRMSDb,
		AvgRMSDb: avgRMSDb,
	}
}
