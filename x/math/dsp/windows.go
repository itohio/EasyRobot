package dsp

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// WindowType represents different window function types
type WindowType int

const (
	Rectangular WindowType = iota
	Sine
	Lanczos
	Triangular
	Hann
	BartlettHann
	Hamming
	Blackman
	BlackmanHarris
	Nuttall
	BlackmanNuttall
	FlatTop
	Gaussian
	Tukey
)

// WindowParams holds window function parameters
type WindowParams struct {
	Type   WindowType
	Param  float32 // Used for Gaussian (sigma) and Tukey (alpha) windows
}

// Windows provides windowing functions for vectors and matrices
type Windows struct{}

// NewWindows creates a new window processor
func NewWindows() *Windows {
	return &Windows{}
}

// ApplyToVector applies a window function to a vector in-place
func (w *Windows) ApplyToVector(v vecTypes.Vector, params WindowParams) {
	data := []float32(v.View().(vec.Vector))
	w.applyWindow(data, params)
}

// ApplyToMatrix applies a window function to each row of a matrix
func (w *Windows) ApplyToMatrix(m matTypes.Matrix, params WindowParams) {
	rows := m.Rows()
	for i := 0; i < rows; i++ {
		row := m.Row(i)
		w.ApplyToVector(row, params)
	}
}

// CreateWindowVector creates a new vector filled with the specified window function
func (w *Windows) CreateWindowVector(length int, params WindowParams) vecTypes.Vector {
	data := make([]float32, length)
	w.applyWindow(data, params)
	return vec.NewFrom(data...)
}

// CreateWindowMatrix creates a 2D window matrix (separable window applied to both dimensions)
func (w *Windows) CreateWindowMatrix(rows, cols int, params WindowParams) matTypes.Matrix {
	matrix := mat.New(rows, cols)

	// Create 1D windows for rows and columns
	rowWindow := make([]float32, rows)
	colWindow := make([]float32, cols)

	w.applyWindow(rowWindow, params)
	w.applyWindow(colWindow, params)

	// Apply separable window
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = rowWindow[i] * colWindow[j]
		}
	}

	return matrix
}

// applyWindow applies the window function to the data slice in-place
func (w *Windows) applyWindow(data []float32, params WindowParams) {
	n := len(data)
	if n == 0 {
		return
	}

	switch params.Type {
	case Rectangular:
		w.rectangular(data)
	case Sine:
		w.sine(data)
	case Lanczos:
		w.lanczos(data)
	case Triangular:
		w.triangular(data)
	case Hann:
		w.hann(data)
	case BartlettHann:
		w.bartlettHann(data)
	case Hamming:
		w.hamming(data)
	case Blackman:
		w.blackman(data)
	case BlackmanHarris:
		w.blackmanHarris(data)
	case Nuttall:
		w.nuttall(data)
	case BlackmanNuttall:
		w.blackmanNuttall(data)
	case FlatTop:
		w.flatTop(data)
	case Gaussian:
		w.gaussian(data, params.Param)
	case Tukey:
		w.tukey(data, params.Param)
	}
}

// Window function implementations
func (w *Windows) rectangular(data []float32) {
	// Rectangular window is just 1.0 for all values
	for i := range data {
		data[i] = 1.0
	}
}

func (w *Windows) sine(data []float32) {
	k := math.Pi / float64(len(data)-1)
	for i := range data {
		data[i] *= float32(math.Sin(k * float64(i)))
	}
}

func (w *Windows) lanczos(data []float32) {
	k := 2 / float64(len(data)-1)
	for i := range data {
		x := math.Pi * (k*float64(i) - 1)
		if x == 0 {
			continue // Avoid NaN
		}
		data[i] *= float32(math.Sin(x) / x)
	}
}

func (w *Windows) triangular(data []float32) {
	a := float64(len(data)-1) / 2
	for i := range data {
		data[i] *= float32(1 - math.Abs(float64(i)/a-1))
	}
}

func (w *Windows) hann(data []float32) {
	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		data[i] *= float32(0.5 * (1 - math.Cos(k*float64(i))))
	}
}

func (w *Windows) bartlettHann(data []float32) {
	const (
		a0 = 0.62
		a1 = 0.48
		a2 = 0.38
	)

	n := len(data)
	k := 2 * math.Pi / float64(n-1)
	for i := range data {
		data[i] *= float32(a0 - a1*math.Abs(float64(i)/float64(n-1)-0.5) - a2*math.Cos(k*float64(i)))
	}
}

func (w *Windows) hamming(data []float32) {
	const (
		a0 = 0.54
		a1 = 0.46
	)

	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		data[i] *= float32(a0 - a1*math.Cos(k*float64(i)))
	}
}

func (w *Windows) blackman(data []float32) {
	const (
		a0 = 0.42
		a1 = 0.5
		a2 = 0.08
	)

	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		x := k * float64(i)
		data[i] *= float32(a0 - a1*math.Cos(x) + a2*math.Cos(2*x))
	}
}

func (w *Windows) blackmanHarris(data []float32) {
	const (
		a0 = 0.35875
		a1 = 0.48829
		a2 = 0.14128
		a3 = 0.01168
	)

	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		x := k * float64(i)
		data[i] *= float32(a0 - a1*math.Cos(x) + a2*math.Cos(2*x) - a3*math.Cos(3*x))
	}
}

func (w *Windows) nuttall(data []float32) {
	const (
		a0 = 0.355768
		a1 = 0.487396
		a2 = 0.144232
		a3 = 0.012604
	)

	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		x := k * float64(i)
		data[i] *= float32(a0 - a1*math.Cos(x) + a2*math.Cos(2*x) - a3*math.Cos(3*x))
	}
}

func (w *Windows) blackmanNuttall(data []float32) {
	const (
		a0 = 0.3635819
		a1 = 0.4891775
		a2 = 0.1365995
		a3 = 0.0106411
	)

	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		x := k * float64(i)
		data[i] *= float32(a0 - a1*math.Cos(x) + a2*math.Cos(2*x) - a3*math.Cos(3*x))
	}
}

func (w *Windows) flatTop(data []float32) {
	const (
		a0 = 0.21557895
		a1 = 0.41663158
		a2 = 0.277263158
		a3 = 0.083578947
		a4 = 0.006947368
	)

	k := 2 * math.Pi / float64(len(data)-1)
	for i := range data {
		x := k * float64(i)
		data[i] *= float32(a0 - a1*math.Cos(x) + a2*math.Cos(2*x) - a3*math.Cos(3*x) + a4*math.Cos(4*x))
	}
}

func (w *Windows) gaussian(data []float32, sigma float32) {
	if sigma <= 0 {
		sigma = 0.5 // Default sigma
	}

	a := float64(len(data)-1) / 2
	for i := range data {
		x := -0.5 * math.Pow((float64(i)-a)/(float64(sigma)*a), 2)
		data[i] *= float32(math.Exp(x))
	}
}

func (w *Windows) tukey(data []float32, alpha float32) {
	if alpha <= 0 {
		// Rectangular window
		return
	}
	if alpha >= 1 {
		// Hann window
		w.hann(data)
		return
	}

	n := len(data)
	alphaL := float64(alpha) * float64(n-1)
	width := int(0.5*alphaL) + 1

	// Apply Hann window to edges
	j := 0.0
	for i := 0; i < width; i++ {
		w := 0.5 * (1 - math.Cos(2*math.Pi*j/alphaL))
		data[i] *= float32(w)
		data[n-1-i] *= float32(w)
		j += 1
	}
}
