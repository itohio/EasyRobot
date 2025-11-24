package dsp

import (
	"github.com/chewxy/math32"
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
	Type  WindowType
	Param float32 // Used for Gaussian (sigma) and Tukey (alpha) windows
}

// ApplyWindowVector applies a window function to a vector in-place
func ApplyWindowVector(v vecTypes.Vector, params WindowParams) {
	data := []float32(v.View().(vec.Vector))
	applyWindow(data, params)
}

// ApplyWindowMatrix applies a window function to each row of a matrix
func ApplyWindowMatrix(m matTypes.Matrix, params WindowParams) {
	rows := m.Rows()
	for i := 0; i < rows; i++ {
		row := m.Row(i)
		ApplyWindowVector(row, params)
	}
}

// CreateWindowVector fills dst with the specified window and returns it as vec.Vector.
// If dst has insufficient capacity, a new slice is allocated.
func CreateWindowVector(length int, params WindowParams, dst []float32) vecTypes.Vector {
	data := ensureBuffer(dst, length)
	setOnes(data)
	applyWindow(data, params)
	return vec.Vector(data)
}

// CreateWindowMatrix fills dst with a separable 2D window. If dst is nil or has
// mismatched dimensions, a new matrix is allocated.
func CreateWindowMatrix(rows, cols int, params WindowParams, dst matTypes.Matrix) matTypes.Matrix {
	matrix := ensureMatrix(dst, rows, cols)

	rowWindow := ensureBuffer(nil, rows)
	colWindow := ensureBuffer(nil, cols)

	setOnes(rowWindow)
	setOnes(colWindow)
	applyWindow(rowWindow, params)
	applyWindow(colWindow, params)

	for i := 0; i < rows; i++ {
		row := matrix.Row(i).(vec.Vector)
		rowFactor := rowWindow[i]
		for j := 0; j < cols; j++ {
			row[j] = rowFactor * colWindow[j]
		}
	}

	return matrix
}

// applyWindow applies the window function to the data slice in-place
func applyWindow(data []float32, params WindowParams) {
	n := len(data)
	if n == 0 {
		return
	}

	switch params.Type {
	case Rectangular:
		rectangular(data)
	case Sine:
		sine(data)
	case Lanczos:
		lanczos(data)
	case Triangular:
		triangular(data)
	case Hann:
		hann(data)
	case BartlettHann:
		bartlettHann(data)
	case Hamming:
		hamming(data)
	case Blackman:
		blackman(data)
	case BlackmanHarris:
		blackmanHarris(data)
	case Nuttall:
		nuttall(data)
	case BlackmanNuttall:
		blackmanNuttall(data)
	case FlatTop:
		flatTop(data)
	case Gaussian:
		gaussian(data, params.Param)
	case Tukey:
		tukey(data, params.Param)
	}
}

func setOnes(dst []float32) {
	for i := range dst {
		dst[i] = 1
	}
}

func ensureBuffer(dst []float32, length int) []float32 {
	if cap(dst) < length {
		return make([]float32, length)
	}
	return dst[:length]
}

func ensureMatrix(dst matTypes.Matrix, rows, cols int) matTypes.Matrix {
	if dst == nil || dst.Rows() != rows || dst.Cols() != cols {
		return mat.New(rows, cols)
	}
	return dst
}

// Window function implementations
func rectangular(data []float32) {
	// Rectangular window is just 1.0 for all values
	for i := range data {
		data[i] = 1.0
	}
}

func sine(data []float32) {
	k := math32.Pi / float32(len(data)-1)
	for i := range data {
		data[i] *= math32.Sin(k * float32(i))
	}
}

func lanczos(data []float32) {
	k := 2 / float32(len(data)-1)
	for i := range data {
		x := math32.Pi * (k*float32(i) - 1)
		if x == 0 {
			continue // Avoid NaN
		}
		data[i] *= math32.Sin(x) / x
	}
}

func triangular(data []float32) {
	a := float32(len(data)-1) / 2
	for i := range data {
		data[i] *= 1 - math32.Abs(float32(i)/a-1)
	}
}

func hann(data []float32) {
	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		data[i] *= 0.5 * (1 - math32.Cos(k*float32(i)))
	}
}

func bartlettHann(data []float32) {
	const (
		a0 = 0.62
		a1 = 0.48
		a2 = 0.38
	)

	n := len(data)
	k := 2 * math32.Pi / float32(n-1)
	for i := range data {
		data[i] *= a0 - a1*math32.Abs(float32(i)/float32(n-1)-0.5) - a2*math32.Cos(k*float32(i))
	}
}

func hamming(data []float32) {
	const (
		a0 = 0.54
		a1 = 0.46
	)

	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		data[i] *= a0 - a1*math32.Cos(k*float32(i))
	}
}

func blackman(data []float32) {
	const (
		a0 = 0.42
		a1 = 0.5
		a2 = 0.08
	)

	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		x := k * float32(i)
		data[i] *= a0 - a1*math32.Cos(x) + a2*math32.Cos(2*x)
	}
}

func blackmanHarris(data []float32) {
	const (
		a0 = 0.35875
		a1 = 0.48829
		a2 = 0.14128
		a3 = 0.01168
	)

	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		x := k * float32(i)
		data[i] *= a0 - a1*math32.Cos(x) + a2*math32.Cos(2*x) - a3*math32.Cos(3*x)
	}
}

func nuttall(data []float32) {
	const (
		a0 = 0.355768
		a1 = 0.487396
		a2 = 0.144232
		a3 = 0.012604
	)

	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		x := k * float32(i)
		data[i] *= a0 - a1*math32.Cos(x) + a2*math32.Cos(2*x) - a3*math32.Cos(3*x)
	}
}

func blackmanNuttall(data []float32) {
	const (
		a0 = 0.3635819
		a1 = 0.4891775
		a2 = 0.1365995
		a3 = 0.0106411
	)

	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		x := k * float32(i)
		data[i] *= a0 - a1*math32.Cos(x) + a2*math32.Cos(2*x) - a3*math32.Cos(3*x)
	}
}

func flatTop(data []float32) {
	const (
		a0 = 0.21557895
		a1 = 0.41663158
		a2 = 0.277263158
		a3 = 0.083578947
		a4 = 0.006947368
	)

	k := 2 * math32.Pi / float32(len(data)-1)
	for i := range data {
		x := k * float32(i)
		data[i] *= a0 - a1*math32.Cos(x) + a2*math32.Cos(2*x) - a3*math32.Cos(3*x) + a4*math32.Cos(4*x)
	}
}

func gaussian(data []float32, sigma float32) {
	if sigma <= 0 {
		sigma = 0.5 // Default sigma
	}

	a := float32(len(data)-1) / 2
	for i := range data {
		x := -0.5 * math32.Pow((float32(i)-a)/(sigma*a), 2)
		data[i] *= math32.Exp(x)
	}
}

func tukey(data []float32, alpha float32) {
	if alpha <= 0 {
		// Rectangular window
		return
	}
	if alpha >= 1 {
		// Hann window
		hann(data)
		return
	}

	n := len(data)
	alphaL := alpha * float32(n-1)
	width := int(0.5*alphaL) + 1

	// Apply Hann window to edges
	j := float32(0)
	for i := 0; i < width; i++ {
		coeff := 0.5 * (1 - math32.Cos(2*math32.Pi*j/alphaL))
		data[i] *= coeff
		data[n-1-i] *= coeff
		j += 1
	}
}
