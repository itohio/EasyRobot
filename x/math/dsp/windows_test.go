package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestWindows_ApplyToVector(t *testing.T) {
	tests := []struct {
		name   string
		params WindowParams
		check  func(t *testing.T, data vec.Vector)
	}{
		{
			name:   "Rectangular",
			params: WindowParams{Type: Rectangular},
			check: func(t *testing.T, data vec.Vector) {
				for i, v := range data {
					if math.Abs(float64(v-1.0)) > 1e-6 {
						t.Errorf("Rectangular window should be 1.0 at index %d, got %f", i, v)
					}
				}
			},
		},
		{
			name:   "Hann",
			params: WindowParams{Type: Hann},
			check: func(t *testing.T, data vec.Vector) {
				// Edges should be near zero
				if math.Abs(float64(data[0])) > 1e-5 {
					t.Errorf("Hann window edge should be near 0, got %f", data[0])
				}
				if math.Abs(float64(data[len(data)-1])) > 1e-5 {
					t.Errorf("Hann window edge should be near 0, got %f", data[len(data)-1])
				}
				// Center should be higher
				center := data[len(data)/2]
				if center <= data[0] {
					t.Errorf("Hann window center should be higher than edge")
				}
			},
		},
		{
			name:   "Hamming",
			params: WindowParams{Type: Hamming},
			check: func(t *testing.T, data vec.Vector) {
				// Edges should be small but not zero
				if data[0] < 0 || data[0] > 1.0 {
					t.Errorf("Hamming window edge out of range: %f", data[0])
				}
			},
		},
		{
			name:   "Blackman",
			params: WindowParams{Type: Blackman},
			check: func(t *testing.T, data vec.Vector) {
				// Edges should be near zero
				if math.Abs(float64(data[0])) > 1e-4 {
					t.Errorf("Blackman window edge should be near 0, got %f", data[0])
				}
			},
		},
		{
			name:   "Gaussian",
			params: WindowParams{Type: Gaussian, Param: 0.5},
			check: func(t *testing.T, data vec.Vector) {
				// Center should be highest
				center := data[len(data)/2]
				edge := data[0]
				if center <= edge {
					t.Errorf("Gaussian window center should be higher than edge")
				}
			},
		},
		{
			name:   "Tukey",
			params: WindowParams{Type: Tukey, Param: 0.5},
			check: func(t *testing.T, data vec.Vector) {
				// Should have some tapering
				if len(data) == 0 {
					t.Errorf("Expected non-empty window")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signal := vec.NewFrom(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
			ApplyWindowVector(signal, tt.params)
			tt.check(t, signal)
		})
	}
}

func TestWindows_CreateWindowVector(t *testing.T) {
	tests := []struct {
		name   string
		length int
		params WindowParams
		check  func(t *testing.T, window vec.Vector)
	}{
		{
			name:   "Rectangular",
			length: 8,
			params: WindowParams{Type: Rectangular},
			check: func(t *testing.T, window vec.Vector) {
				for i, v := range window {
					if math.Abs(float64(v-1.0)) > 1e-6 {
						t.Errorf("Expected 1.0 at index %d, got %f", i, v)
					}
				}
			},
		},
		{
			name:   "Hann",
			length: 8,
			params: WindowParams{Type: Hann},
			check: func(t *testing.T, window vec.Vector) {
				// Edges should be near zero
				if math.Abs(float64(window[0])) > 1e-5 {
					t.Errorf("Hann edge should be near 0, got %f", window[0])
				}
				if math.Abs(float64(window[len(window)-1])) > 1e-5 {
					t.Errorf("Hann edge should be near 0, got %f", window[len(window)-1])
				}
			},
		},
		{
			name:   "Gaussian",
			length: 10,
			params: WindowParams{Type: Gaussian, Param: 0.5},
			check: func(t *testing.T, window vec.Vector) {
				// Center should be highest
				center := window[5]
				edge := window[0]
				if center <= edge {
					t.Errorf("Gaussian center should be higher than edge")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			window := CreateWindowVector(tt.length, tt.params, nil)
			data := window.View().(vec.Vector)
			if len(data) != tt.length {
				t.Errorf("Expected length %d, got %d", tt.length, len(data))
			}
			tt.check(t, data)
		})
	}
}

func TestWindows_CreateWindowVectorReuse(t *testing.T) {
	buf := make([]float32, 8)
	window := CreateWindowVector(8, WindowParams{Type: Rectangular}, buf)
	data := window.View().(vec.Vector)

	if len(data) == 0 {
		t.Fatalf("expected non-empty window")
	}

	if &data[0] != &buf[0] {
		t.Fatalf("expected destination buffer to be reused")
	}
}

func TestWindows_ApplyToMatrix(t *testing.T) {
	matrix := mat.New(4, 4)
	// Fill with ones
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			matrix[i][j] = 1.0
		}
	}

	params := WindowParams{Type: Hann}
	ApplyWindowMatrix(matrix, params)

	// Check that windowing was applied (values should be less than 1.0)
	flat := matrix.Flat()
	hasReduced := false
	for _, v := range flat {
		if v < 1.0 {
			hasReduced = true
			break
		}
	}
	if !hasReduced {
		t.Errorf("Window should reduce values")
	}
}

func TestWindows_CreateWindowMatrix(t *testing.T) {
	params := WindowParams{Type: Hann}
	window2D := CreateWindowMatrix(4, 4, params, nil)
	matrix := window2D.(mat.Matrix)

	// Should be separable (row window * col window)
	// Check that corners are near zero (both row and col edges)
	if math.Abs(float64(matrix[0][0])) > 1e-4 {
		t.Errorf("Corner should be near 0, got %f", matrix[0][0])
	}
	if math.Abs(float64(matrix[3][3])) > 1e-4 {
		t.Errorf("Corner should be near 0, got %f", matrix[3][3])
	}

	// Center should be higher
	center := matrix[2][2]
	if center <= matrix[0][0] {
		t.Errorf("Center should be higher than corner")
	}
}

func TestWindows_CreateWindowMatrixReuse(t *testing.T) {
	dst := mat.New(4, 4)
	window := CreateWindowMatrix(4, 4, WindowParams{Type: Rectangular}, dst)

	dstFlat := dst.Flat()
	windowFlat := window.Flat()

	if len(dstFlat) == 0 || len(windowFlat) == 0 {
		t.Fatalf("expected non-empty matrix")
	}

	if &dstFlat[0] != &windowFlat[0] {
		t.Fatalf("expected destination matrix buffer to be reused")
	}
}

func TestWindows_AllWindowTypes(t *testing.T) {
	length := 16

	windowTypes := []WindowType{
		Rectangular,
		Sine,
		Lanczos,
		Triangular,
		Hann,
		BartlettHann,
		Hamming,
		Blackman,
		BlackmanHarris,
		Nuttall,
		BlackmanNuttall,
		FlatTop,
		Gaussian,
		Tukey,
	}

	for _, wt := range windowTypes {
		wt := wt
		t.Run(wt.String(), func(t *testing.T) {
			params := WindowParams{Type: wt}
			if wt == Gaussian {
				params.Param = 0.5
			} else if wt == Tukey {
				params.Param = 0.5
			}

			window := CreateWindowVector(length, params, nil)
			data := window.View().(vec.Vector)

			if len(data) != length {
				t.Errorf("Expected length %d, got %d", length, len(data))
			}

			if wt == FlatTop {
				return
			}

			const minValue = -1e-6
			for i, v := range data {
				if v < minValue {
					t.Errorf("Window value at index %d is below allowed threshold: %f", i, v)
				}
			}
		})
	}
}

func TestWindows_EmptySignal(t *testing.T) {
	// Test with empty vector
	signal := vec.NewFrom([]float32{}...)
	params := WindowParams{Type: Hann}
	ApplyWindowVector(signal, params)

	// Should not panic
	if len(signal) != 0 {
		t.Errorf("Expected empty signal to remain empty")
	}
}

func TestWindows_ParametricWindows(t *testing.T) {
	length := 16

	tests := []struct {
		name  string
		wt    WindowType
		param float32
	}{
		{
			name:  "Gaussian small sigma",
			wt:    Gaussian,
			param: 0.3,
		},
		{
			name:  "Gaussian large sigma",
			wt:    Gaussian,
			param: 0.8,
		},
		{
			name:  "Tukey small alpha",
			wt:    Tukey,
			param: 0.2,
		},
		{
			name:  "Tukey large alpha",
			wt:    Tukey,
			param: 0.8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := WindowParams{Type: tt.wt, Param: tt.param}
			window := CreateWindowVector(length, params, nil)
			data := window.View().(vec.Vector)

			if len(data) != length {
				t.Errorf("Expected length %d, got %d", length, len(data))
			}
		})
	}
}

// Helper to get window type name for testing
func (wt WindowType) String() string {
	names := []string{
		"Rectangular",
		"Sine",
		"Lanczos",
		"Triangular",
		"Hann",
		"BartlettHann",
		"Hamming",
		"Blackman",
		"BlackmanHarris",
		"Nuttall",
		"BlackmanNuttall",
		"FlatTop",
		"Gaussian",
		"Tukey",
	}
	if int(wt) < len(names) {
		return names[wt]
	}
	return "Unknown"
}
