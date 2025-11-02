package mat

import (
	"testing"

	"github.com/chewxy/math32"
)

func TestPytag(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected float32
		epsilon  float32
	}{
		{
			name:     "zero both",
			a:        0,
			b:        0,
			expected: 0,
			epsilon:  1e-6,
		},
		{
			name:     "zero a",
			a:        0,
			b:        3,
			expected: 3,
			epsilon:  1e-6,
		},
		{
			name:     "zero b",
			a:        4,
			b:        0,
			expected: 4,
			epsilon:  1e-6,
		},
		{
			name:     "3-4-5 triangle",
			a:        3,
			b:        4,
			expected: 5,
			epsilon:  1e-6,
		},
		{
			name:     "a > b",
			a:        10,
			b:        6,
			expected: math32.Sqrt(136),
			epsilon:  1e-5,
		},
		{
			name:     "b > a",
			a:        6,
			b:        10,
			expected: math32.Sqrt(136),
			epsilon:  1e-5,
		},
		{
			name:     "large values (overflow protection)",
			a:        1e10,
			b:        1e10,
			expected: math32.Sqrt(2) * 1e10,
			epsilon:  1e-3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pytag(tt.a, tt.b)
			if math32.Abs(result-tt.expected) > tt.epsilon {
				t.Errorf("pytag(%v, %v) = %v, want %v (error: %v)",
					tt.a, tt.b, result, tt.expected, math32.Abs(result-tt.expected))
			}
		})
	}
}

func TestSIGN(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected float32
	}{
		{
			name:     "positive b",
			a:        5,
			b:        3,
			expected: 5,
		},
		{
			name:     "negative b",
			a:        5,
			b:        -3,
			expected: -5,
		},
		{
			name:     "zero b",
			a:        5,
			b:        0,
			expected: 5,
		},
		{
			name:     "negative a, positive b",
			a:        -5,
			b:        3,
			expected: 5,
		},
		{
			name:     "negative a, negative b",
			a:        -5,
			b:        -3,
			expected: -5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SIGN(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("SIGN(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestFMAX(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected float32
	}{
		{
			name:     "a > b",
			a:        10,
			b:        5,
			expected: 10,
		},
		{
			name:     "b > a",
			a:        5,
			b:        10,
			expected: 10,
		},
		{
			name:     "equal",
			a:        5,
			b:        5,
			expected: 5,
		},
		{
			name:     "negative values",
			a:        -5,
			b:        -10,
			expected: -5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FMAX(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("FMAX(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestIMIN(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{
			name:     "a < b",
			a:        5,
			b:        10,
			expected: 5,
		},
		{
			name:     "b < a",
			a:        10,
			b:        5,
			expected: 5,
		},
		{
			name:     "equal",
			a:        5,
			b:        5,
			expected: 5,
		},
		{
			name:     "negative values",
			a:        -5,
			b:        -10,
			expected: -10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IMIN(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("IMIN(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

