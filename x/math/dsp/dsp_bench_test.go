package dsp

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func BenchmarkFFT_Transform1D_1024(b *testing.B) {
	fft := NewFFT1D(1024)
	signal := vec.New(1024)
	spectrum := vec.New(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Forward(signal, spectrum)
	}
}

func BenchmarkFFT_Transform1D_8192(b *testing.B) {
	fft := NewFFT1D(8192)
	signal := vec.New(8192)
	spectrum := vec.New(8192)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Forward(signal, spectrum)
	}
}

func BenchmarkFFT_InverseTransform1D_1024(b *testing.B) {
	fft := NewFFT1D(1024)
	signal := vec.New(1024)
	spectrum := vec.New(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Backward(spectrum, signal)
	}
}

func BenchmarkFFT_Transform2D_64x64(b *testing.B) {
	fft := NewFFT2D(64, 64)
	matrix := mat.New(64, 64)
	spectrum := mat.New(64, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Forward(matrix, spectrum)
	}
}

func BenchmarkFFT_Transform2D_256x256(b *testing.B) {
	fft := NewFFT2D(256, 256)
	matrix := mat.New(256, 256)
	spectrum := mat.New(256, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Forward(matrix, spectrum)
	}
}

func BenchmarkFFT_InverseTransform2D_64x64(b *testing.B) {
	fft := NewFFT2D(64, 64)
	matrix := mat.New(64, 64)
	spectrum := mat.New(64, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Backward(spectrum, matrix)
	}
}

func BenchmarkFFT1D_Convolve_1024(b *testing.B) {
	fft := NewFFT1D(1024)
	signal := vec.New(1024)
	kernel := vec.New(128)
	result := vec.New(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Convolve(signal, kernel, result)
	}
}

func BenchmarkFFT2D_Convolve_64x64(b *testing.B) {
	fft := NewFFT2D(64, 64)
	signal := mat.New(64, 64)
	kernel := mat.New(16, 16)
	result := mat.New(64, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fft.Convolve(signal, kernel, result)
	}
}

func BenchmarkWindows_ApplyToVector_Hann_1024(b *testing.B) {
	windows := NewWindows()
	signal := vec.New(1024)
	params := WindowParams{Type: Hann}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		windows.ApplyToVector(signal, params)
	}
}

func BenchmarkWindows_ApplyToMatrix_Hann_64x64(b *testing.B) {
	windows := NewWindows()
	matrix := mat.New(64, 64)
	params := WindowParams{Type: Hann}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		windows.ApplyToMatrix(matrix, params)
	}
}

func BenchmarkSignalGenerator1D_Sinusoid_1024(b *testing.B) {
	gen := NewSignalGenerator1D()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.Sinusoid(1024, 10.0, 1.0, 0.0, 1000.0)
	}
}

func BenchmarkSignalGenerator2D_Gaussian_256x256(b *testing.B) {
	gen := NewSignalGenerator2D()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.Gaussian(256, 256, 50.0)
	}
}

func BenchmarkMeasurements_Measure1D_1024(b *testing.B) {
	measurements := NewMeasurements()
	signal := vec.New(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		measurements.Measure1D(signal)
	}
}

func BenchmarkMeasurements_Measure2D_64x64(b *testing.B) {
	measurements := NewMeasurements()
	matrix := mat.New(64, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		measurements.Measure2D(matrix)
	}
}

func BenchmarkMeasurements_Goertzel_1024(b *testing.B) {
	measurements := NewMeasurements()
	signal := vec.New(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		measurements.Goertzel(signal, 100.0, 1000.0)
	}
}

func BenchmarkMeasurements_CrossCorrelate1D_1024(b *testing.B) {
	measurements := NewMeasurements()
	signal1 := vec.New(1024)
	signal2 := vec.New(512)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		measurements.CrossCorrelate1D(signal1, signal2)
	}
}

func BenchmarkMeasurements_CrossCorrelate2D_64x64(b *testing.B) {
	measurements := NewMeasurements()
	matrix1 := mat.New(64, 64)
	matrix2 := mat.New(32, 32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		measurements.CrossCorrelate2D(matrix1, matrix2)
	}
}

func BenchmarkMeasurements_EstimateShift2D_64x64(b *testing.B) {
	measurements := NewMeasurements()
	matrix1 := mat.New(64, 64)
	matrix2 := mat.New(64, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		measurements.EstimateShift2D(matrix1, matrix2)
	}
}
