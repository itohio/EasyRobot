package dsp

import (
	"math"
	"math/rand"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// SignalGenerator1D provides functions to generate various types of 1D signals
type SignalGenerator1D struct{}

// NewSignalGenerator1D creates a new 1D signal generator
func NewSignalGenerator1D() *SignalGenerator1D {
	return &SignalGenerator1D{}
}

// SignalGenerator2D provides functions to generate various types of 2D signals
type SignalGenerator2D struct{}

// NewSignalGenerator2D creates a new 2D signal generator
func NewSignalGenerator2D() *SignalGenerator2D {
	return &SignalGenerator2D{}
}

// Zeros creates a vector filled with zeros
func (sg *SignalGenerator1D) Zeros(length int) vecTypes.Vector {
	data := make([]float32, length)
	return vec.NewFrom(data...)
}

// Ones creates a vector filled with ones
func (sg *SignalGenerator1D) Ones(length int) vecTypes.Vector {
	data := make([]float32, length)
	for i := range data {
		data[i] = 1.0
	}
	return vec.NewFrom(data...)
}

// Constant creates a vector filled with a constant value
func (sg *SignalGenerator1D) Constant(length int, value float32) vecTypes.Vector {
	data := make([]float32, length)
	for i := range data {
		data[i] = value
	}
	return vec.NewFrom(data...)
}

// Ramp creates a linear ramp signal: [0, 1, 2, ..., length-1]
func (sg *SignalGenerator1D) Ramp(length int) vecTypes.Vector {
	data := make([]float32, length)
	for i := range data {
		data[i] = float32(i)
	}
	return vec.NewFrom(data...)
}

// RampNormalized creates a normalized linear ramp: [0, 1/length, 2/length, ..., 1]
func (sg *SignalGenerator1D) RampNormalized(length int) vecTypes.Vector {
	data := make([]float32, length)
	scale := 1.0 / float32(length-1)
	for i := range data {
		data[i] = float32(i) * scale
	}
	return vec.NewFrom(data...)
}

// Step creates a step function: [0, 0, ..., 0, 1, 1, ..., 1]
func (sg *SignalGenerator1D) Step(length int, stepPoint int) vecTypes.Vector {
	if stepPoint < 0 {
		stepPoint = 0
	}
	if stepPoint > length {
		stepPoint = length
	}

	data := make([]float32, length)
	for i := stepPoint; i < length; i++ {
		data[i] = 1.0
	}
	return vec.NewFrom(data...)
}

// Sinusoid creates a sinusoidal signal
func (sg *SignalGenerator1D) Sinusoid(length int, frequency, amplitude, phase float32, sampleRate float32) vecTypes.Vector {
	data := make([]float32, length)
	omega := 2 * math.Pi * float64(frequency) / float64(sampleRate)

	for i := range data {
		data[i] = amplitude * float32(math.Sin(omega*float64(i)+float64(phase)))
	}
	return vec.NewFrom(data...)
}

// Cosine creates a cosine signal
func (sg *SignalGenerator1D) Cosine(length int, frequency, amplitude, phase float32, sampleRate float32) vecTypes.Vector {
	data := make([]float32, length)
	omega := 2 * math.Pi * float64(frequency) / float64(sampleRate)

	for i := range data {
		data[i] = amplitude * float32(math.Cos(omega*float64(i)+float64(phase)))
	}
	return vec.NewFrom(data...)
}

// Square creates a square wave signal
func (sg *SignalGenerator1D) Square(length int, frequency, amplitude float32, sampleRate float32) vecTypes.Vector {
	data := make([]float32, length)
	period := int(sampleRate / frequency)

	for i := range data {
		if (i/period)%2 == 0 {
			data[i] = amplitude
		} else {
			data[i] = -amplitude
		}
	}
	return vec.NewFrom(data...)
}

// Triangle creates a triangle wave signal
func (sg *SignalGenerator1D) Triangle(length int, frequency, amplitude float32, sampleRate float32) vecTypes.Vector {
	data := make([]float32, length)
	period := int(sampleRate / frequency)

	for i := range data {
		pos := i % period
		halfPeriod := period / 2
		if pos < halfPeriod {
			data[i] = amplitude * (2*float32(pos)/float32(period) - 0.5)
		} else {
			data[i] = amplitude * (1.5 - 2*float32(pos)/float32(period))
		}
	}
	return vec.NewFrom(data...)
}

// NoiseUniform creates uniform random noise
func (sg *SignalGenerator1D) NoiseUniform(length int, amplitude float32) vecTypes.Vector {
	data := make([]float32, length)
	for i := range data {
		data[i] = amplitude * (rand.Float32()*2 - 1) // [-amplitude, amplitude]
	}
	return vec.NewFrom(data...)
}

// NoiseGaussian creates Gaussian random noise
func (sg *SignalGenerator1D) NoiseGaussian(length int, stddev float32) vecTypes.Vector {
	data := make([]float32, length)
	for i := range data {
		data[i] = stddev * float32(rand.NormFloat64())
	}
	return vec.NewFrom(data...)
}

// Impulse creates an impulse signal (1 at position, 0 elsewhere)
func (sg *SignalGenerator1D) Impulse(length int, position int) vecTypes.Vector {
	data := make([]float32, length)
	if position >= 0 && position < length {
		data[position] = 1.0
	}
	return vec.NewFrom(data...)
}

// Exponential creates an exponential decay signal
func (sg *SignalGenerator1D) Exponential(length int, decayRate float32) vecTypes.Vector {
	data := make([]float32, length)
	for i := range data {
		data[i] = float32(math.Exp(-float64(decayRate) * float64(i)))
	}
	return vec.NewFrom(data...)
}

// Chirp creates a linear chirp signal (frequency sweep)
func (sg *SignalGenerator1D) Chirp(length int, f0, f1, amplitude float32, sampleRate float32) vecTypes.Vector {
	data := make([]float32, length)
	t := 1.0 / sampleRate
	k := (f1 - f0) / float32(length-1)

	for i := range data {
		time := float32(i) * t
		freq := f0 + k*float32(i)
		phase := 2 * math.Pi * freq * time
		data[i] = amplitude * float32(math.Sin(float64(phase)))
	}
	return vec.NewFrom(data...)
}

// Zeros creates a matrix filled with zeros
func (sg *SignalGenerator2D) Zeros(rows, cols int) matTypes.Matrix {
	return mat.New(rows, cols)
}

// Ones creates a matrix filled with ones
func (sg *SignalGenerator2D) Ones(rows, cols int) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	flat := matrix.Flat()
	for i := range flat {
		flat[i] = 1.0
	}
	return matrix
}

// Constant creates a matrix filled with a constant value
func (sg *SignalGenerator2D) Constant(rows, cols int, value float32) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	flat := matrix.Flat()
	for i := range flat {
		flat[i] = value
	}
	return matrix
}

// Ramp creates a 2D ramp in row-major order
func (sg *SignalGenerator2D) Ramp(rows, cols int) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	flat := matrix.Flat()
	for i := range flat {
		flat[i] = float32(i)
	}
	return matrix
}

// Checkerboard creates a checkerboard pattern
func (sg *SignalGenerator2D) Checkerboard(rows, cols int, blockSize int) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if ((i/blockSize)+(j/blockSize))%2 == 0 {
				matrix[i][j] = 1.0
			} else {
				matrix[i][j] = -1.0
			}
		}
	}
	return matrix
}

// Gaussian creates a 2D Gaussian distribution
func (sg *SignalGenerator2D) Gaussian(rows, cols int, sigma float32) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	centerX := float32(cols-1) / 2
	centerY := float32(rows-1) / 2

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			x := float32(j) - centerX
			y := float32(i) - centerY
			matrix[i][j] = float32(math.Exp(float64(-0.5 * (x*x+y*y) / (sigma * sigma))))
		}
	}
	return matrix
}

// Sinusoid creates a 2D sinusoidal pattern
func (sg *SignalGenerator2D) Sinusoid(rows, cols int, freqX, freqY, amplitude float32) matTypes.Matrix {
	matrix := mat.New(rows, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			phase := 2 * math.Pi * (freqX*float32(j)/float32(cols) + freqY*float32(i)/float32(rows))
			matrix[i][j] = amplitude * float32(math.Sin(float64(phase)))
		}
	}
	return matrix
}

// NoiseUniform creates 2D uniform random noise
func (sg *SignalGenerator2D) NoiseUniform(rows, cols int, amplitude float32) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	flat := matrix.Flat()
	for i := range flat {
		flat[i] = amplitude * (rand.Float32()*2 - 1)
	}
	return matrix
}

// NoiseGaussian creates 2D Gaussian random noise
func (sg *SignalGenerator2D) NoiseGaussian(rows, cols int, stddev float32) matTypes.Matrix {
	matrix := mat.New(rows, cols)
	flat := matrix.Flat()
	for i := range flat {
		flat[i] = stddev * float32(rand.NormFloat64())
	}
	return matrix
}
