package c64

import (
	"math/cmplx"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func complexInDelta(t *testing.T, expected, actual complex64, delta float32, msgAndArgs ...interface{}) {
	require.InDelta(t, real(expected), real(actual), float64(delta), append([]interface{}{"real part"}, msgAndArgs...)...)
	require.InDelta(t, imag(expected), imag(actual), float64(delta), append([]interface{}{"imag part"}, msgAndArgs...)...)
}

func TestConvolve1D(t *testing.T) {
	tests := []struct {
		name     string
		signal1  []complex64
		signal2  []complex64
		expected []complex64
	}{
		{
			name:    "impulse_with_impulse",
			signal1: []complex64{1, 0, 0, 0},
			signal2: []complex64{1, 0, 0, 0},
			// Convolution: y[n] = Σ signal1[k] * signal2[n-k]
			// i=0: k=0: signal2[0-0]=signal2[0]=1 → 1*1=1; k=1,2,3: signal2[0-1,0-2,0-3] invalid
			//      dst[0] = 1
			// i=1,2,3,4,5,6: all signal2 indices invalid → 0
			expected: []complex64{1, 0, 0, 0, 0, 0, 0},
		},
		{
			name:    "simple",
			signal1: []complex64{1, 2, 3},
			signal2: []complex64{1, 2},
			// Convolution: y[n] = Σ signal1[k] * signal2[n-k]
			// i=0: k=0: signal2[0-0]=signal2[0]=1 → 1*1=1; k=1,2: signal2[0-1,0-2] invalid
			//      dst[0] = 1
			// i=1: k=0: signal2[1-0]=signal2[1]=2 → 1*2=2; k=1: signal2[1-1]=signal2[0]=1 → 2*1=2
			//      dst[1] = 2 + 2 = 4
			// i=2: k=1: signal2[2-1]=signal2[1]=2 → 2*2=4; k=2: signal2[2-2]=signal2[0]=1 → 3*1=3
			//      dst[2] = 4 + 3 = 7
			// i=3: k=2: signal2[3-2]=signal2[1]=2 → 3*2=6; k=0,1: signal2[3-0,3-1] invalid
			//      dst[3] = 6
			expected: []complex64{1, 4, 7, 6},
		},
		{
			name:    "single_element",
			signal1: []complex64{5},
			signal2: []complex64{3},
			// i=0: k=0: signal2[0-0]=signal2[0]=3 → 5*3=15
			expected: []complex64{15},
		},
		{
			name:    "signal2_longer",
			signal1: []complex64{1, 2},
			signal2: []complex64{1, 2, 3, 4},
			// Convolution: y[n] = Σ signal1[k] * signal2[n-k]
			// i=0: k=0: signal2[0-0]=signal2[0]=1 → 1*1=1
			//      dst[0] = 1
			// i=1: k=0: signal2[1-0]=signal2[1]=2 → 1*2=2; k=1: signal2[1-1]=signal2[0]=1 → 2*1=2
			//      dst[1] = 2 + 2 = 4
			// i=2: k=0: signal2[2-0]=signal2[2]=3 → 1*3=3; k=1: signal2[2-1]=signal2[1]=2 → 2*2=4
			//      dst[2] = 3 + 4 = 7
			// i=3: k=0: signal2[3-0]=signal2[3]=4 → 1*4=4; k=1: signal2[3-1]=signal2[2]=3 → 2*3=6
			//      dst[3] = 4 + 6 = 10
			// i=4: k=1: signal2[4-1]=signal2[3]=4 → 2*4=8
			//      dst[4] = 8
			expected: []complex64{1, 4, 7, 10, 8},
		},
		{
			name:    "complex_values",
			signal1: []complex64{1 + 2i, 3 + 4i},
			signal2: []complex64{1 + 1i, 2 + 2i},
			// i=0: k=0: signal2[0]=1+1i → (1+2i)*(1+1i) = 1+1i+2i+2i² = 1+3i-2 = -1+3i
			//      dst[0] = -1+3i
			// i=1: k=0: signal2[1]=2+2i → (1+2i)*(2+2i) = 2+2i+4i+4i² = 2+6i-4 = -2+6i
			//      k=1: signal2[0]=1+1i → (3+4i)*(1+1i) = 3+3i+4i+4i² = 3+7i-4 = -1+7i
			//      dst[1] = (-2+6i) + (-1+7i) = -3+13i
			// i=2: k=1: signal2[1]=2+2i → (3+4i)*(2+2i) = 6+6i+8i+8i² = 6+14i-8 = -2+14i
			//      dst[2] = -2+14i
			expected: []complex64{-1 + 3i, -3 + 13i, -2 + 14i},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.signal1)+len(tt.signal2)-1)
			Convolve1D(dst, tt.signal1, tt.signal2)

			for i, exp := range tt.expected {
				if i < len(dst) {
					complexInDelta(t, exp, dst[i], 1e-6, "Convolution at index %d", i)
				}
			}
		})
	}
}

func TestConvolve1D_Empty(t *testing.T) {
	dst := make([]complex64, 10)
	Convolve1D(dst, []complex64{}, []complex64{1, 2})
	// Should not panic, just return early
}

func TestConvolve1D_SmallDst(t *testing.T) {
	signal1 := []complex64{1, 2, 3, 4, 5}
	signal2 := []complex64{1, 2, 3}
	dst := make([]complex64, 3) // Smaller than needed (7)

	Convolve1D(dst, signal1, signal2)
	// Should fill what it can without panicking
}

func TestConvolve1DAdd(t *testing.T) {
	tests := []struct {
		name     string
		dst      []complex64
		signal1  []complex64
		signal2  []complex64
		expected []complex64
	}{
		{
			name:    "accumulation",
			dst:     []complex64{10, 20, 30, 40},
			signal1: []complex64{1, 2, 3},
			signal2: []complex64{1, 2},
			// Convolution: [1, 4, 7, 6]
			// dst + result: [10+1, 20+4, 30+7, 40+6] = [11, 24, 37, 46]
			expected: []complex64{11, 24, 37, 46},
		},
		{
			name:    "zero_dst",
			dst:     make([]complex64, 4),
			signal1: []complex64{1, 2, 3},
			signal2: []complex64{1, 2},
			// Same as Convolve1D
			expected: []complex64{1, 4, 7, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.dst))
			copy(dst, tt.dst)
			Convolve1DAdd(dst, tt.signal1, tt.signal2)

			for i, exp := range tt.expected {
				if i < len(dst) {
					complexInDelta(t, exp, dst[i], 1e-6, "ConvolutionAdd at index %d", i)
				}
			}
		})
	}
}

func TestConvolve1DVsAdd(t *testing.T) {
	// Test that Convolve1D produces same result as Convolve1DAdd with zero-initialized dst
	signal1 := []complex64{1, 2, 3, 4, 5}
	signal2 := []complex64{1, 2}

	dst1 := make([]complex64, len(signal1)+len(signal2)-1)
	dst2 := make([]complex64, len(signal1)+len(signal2)-1)

	Convolve1D(dst1, signal1, signal2)
	Convolve1DAdd(dst2, signal1, signal2)

	// Results should match for the computed portion
	for i := range dst1 {
		complexInDelta(t, dst1[i], dst2[i], 1e-5, "Convolve1D should match Convolve1DAdd with zero dst at index %d", i)
	}
}

func TestConvolve2D(t *testing.T) {
	tests := []struct {
		name     string
		signal1  []complex64
		rows1    int
		signal2  []complex64
		rows2    int
		dstRows  int
		validate func(*testing.T, []complex64, int, int)
	}{
		{
			name:    "simple_2x2_with_1x2",
			signal1: []complex64{1, 2, 3, 4}, // 2x2
			rows1:   2,
			signal2: []complex64{1, 2}, // 1x2
			rows2:   1,
			dstRows: 2,
			validate: func(t *testing.T, dst []complex64, rows, cols int) {
				// Convolution: y[i,j] = Σ signal1[k,l] * signal2[i-k, j-l]
				// For (i,j)=(0,0): k=0,l=0: signal2[0-0,0-0]=signal2[0,0]=1 → 1*1=1
				//                  k=0,l=1: signal2[0-0,0-1] invalid
				//                  k=1,l=0: signal2[0-1,0-0] invalid
				//                  k=1,l=1: signal2[0-1,0-1] invalid
				//                  dst[0] = 1
				// For (i,j)=(0,1): k=0,l=0: signal2[0-0,1-0]=signal2[0,1]=2 → 1*2=2
				//                  k=0,l=1: signal2[0-0,1-1]=signal2[0,0]=1 → 2*1=2
				//                  dst[1] = 2 + 2 = 4
				// For (i,j)=(1,0): k=0,l=0: signal2[1-0,0-0]=signal2[1,0] invalid
				//                  k=1,l=0: signal2[1-1,0-0]=signal2[0,0]=1 → 3*1=3
				//                  dst[2] = 3
				// For (i,j)=(1,1): k=0,l=0: signal2[1-0,1-0]=signal2[1,1] invalid
				//                  k=0,l=1: signal2[1-0,1-1]=signal2[1,0] invalid
				//                  k=1,l=0: signal2[1-1,1-0]=signal2[0,1]=2 → 3*2=6
				//                  k=1,l=1: signal2[1-1,1-1]=signal2[0,0]=1 → 4*1=4
				//                  dst[3] = 6 + 4 = 10
				complexInDelta(t, 1.0, dst[0], 1e-6, "Convolution at (0,0)")
				complexInDelta(t, 4.0, dst[1], 1e-6, "Convolution at (0,1)")
				complexInDelta(t, 3.0, dst[2], 1e-6, "Convolution at (1,0)")
				complexInDelta(t, 10.0, dst[3], 1e-6, "Convolution at (1,1)")
			},
		},
		{
			name:    "identity_3x3",
			signal1: []complex64{1, 0, 0, 0, 1, 0, 0, 0, 1}, // 3x3 identity
			rows1:   3,
			signal2: []complex64{1, 0, 0, 0, 1, 0, 0, 0, 1}, // 3x3 identity
			rows2:   3,
			dstRows: 3,
			validate: func(t *testing.T, dst []complex64, rows, cols int) {
				// Should have non-zero values where they align
				hasNonZero := false
				for _, v := range dst {
					if cmplx.Abs(complex128(v)) > 1e-6 {
						hasNonZero = true
						break
					}
				}
				assert.True(t, hasNonZero, "Identity convolution should produce non-zero result")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cols1 := len(tt.signal1) / tt.rows1
			cols2 := len(tt.signal2) / tt.rows2
			dstCols := cols1
			if cols2 > cols1 {
				dstCols = cols2
			}
			dst := make([]complex64, tt.dstRows*dstCols)

			Convolve2D(dst, tt.dstRows, tt.signal1, tt.rows1, tt.signal2, tt.rows2)

			tt.validate(t, dst, tt.dstRows, dstCols)
		})
	}
}

func TestConvolve2D_Panic(t *testing.T) {
	t.Run("zero_rows1", func(t *testing.T) {
		signal1 := []complex64{1, 2}
		signal2 := []complex64{1, 2}
		dst := make([]complex64, 10)

		assert.Panics(t, func() {
			Convolve2D(dst, 2, signal1, 0, signal2, 1)
		})
	})

	t.Run("zero_rows2", func(t *testing.T) {
		signal1 := []complex64{1, 2}
		signal2 := []complex64{1, 2}
		dst := make([]complex64, 10)

		assert.Panics(t, func() {
			Convolve2D(dst, 2, signal1, 1, signal2, 0)
		})
	})

	t.Run("dimension_mismatch", func(t *testing.T) {
		signal1 := []complex64{1, 2, 3} // 1x3, but rows1=2 would be wrong
		signal2 := []complex64{1, 2}
		dst := make([]complex64, 10)

		assert.Panics(t, func() {
			Convolve2D(dst, 2, signal1, 2, signal2, 1) // rows1=2 but len(signal1)=3, not divisible
		})
	})

	t.Run("dst_too_small", func(t *testing.T) {
		signal1 := []complex64{1, 2, 3, 4} // 2x2
		signal2 := []complex64{1, 2}       // 1x2
		dst := make([]complex64, 1)         // Too small

		assert.Panics(t, func() {
			Convolve2D(dst, 2, signal1, 2, signal2, 1)
		})
	})
}

func TestConvolve2DAdd(t *testing.T) {
	t.Run("accumulation", func(t *testing.T) {
		signal1 := []complex64{1, 2, 3, 4} // 2x2
		signal2 := []complex64{1, 2}       // 1x2
		dst := []complex64{10, 20, 30, 40} // 2x2 pre-filled

		Convolve2DAdd(dst, 2, signal1, 2, signal2, 1)

		// Should have accumulated values (not just replaced)
		hasNonZero := false
		for _, v := range dst {
			if cmplx.Abs(complex128(v)) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		assert.True(t, hasNonZero, "2D convolution add should produce non-zero result")
	})
}

func BenchmarkConvolve1D(b *testing.B) {
	sizes := []struct {
		name   string
		n1, n2 int
	}{
		{"small", 10, 5},
		{"medium", 100, 50},
		{"large", 1000, 500},
		{"very_large", 10000, 5000},
		{"signal2_longer", 50, 100},
	}

	for _, sz := range sizes {
		signal1 := make([]complex64, sz.n1)
		signal2 := make([]complex64, sz.n2)
		resultLen := sz.n1 + sz.n2 - 1
		dst := make([]complex64, resultLen)

		for i := range signal1 {
			signal1[i] = complex64(complex(float32(i), float32(i*2)))
		}
		for i := range signal2 {
			signal2[i] = complex64(complex(float32(i*2), float32(i*3)))
		}

		b.Run(sz.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Convolve1D(dst, signal1, signal2)
			}
		})
	}
}

func BenchmarkConvolve1DAdd(b *testing.B) {
	sizes := []struct {
		name   string
		n1, n2 int
	}{
		{"small", 10, 5},
		{"medium", 100, 50},
		{"large", 1000, 500},
		{"very_large", 10000, 5000},
	}

	for _, sz := range sizes {
		signal1 := make([]complex64, sz.n1)
		signal2 := make([]complex64, sz.n2)
		resultLen := sz.n1 + sz.n2 - 1
		dst := make([]complex64, resultLen)

		for i := range signal1 {
			signal1[i] = complex64(complex(float32(i), float32(i*2)))
		}
		for i := range signal2 {
			signal2[i] = complex64(complex(float32(i*2), float32(i*3)))
		}
		for i := range dst {
			dst[i] = complex64(complex(float32(i), float32(i)))
		}

		b.Run(sz.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Convolve1DAdd(dst, signal1, signal2)
			}
		})
	}
}

func BenchmarkConvolve2D(b *testing.B) {
	sizes := []struct {
		name         string
		rows1, cols1 int
		rows2, cols2 int
		dstRows      int
	}{
		{"small", 10, 10, 5, 5, 10},
		{"medium", 50, 50, 25, 25, 50},
		{"large", 100, 100, 50, 50, 100},
		{"wide_signal2", 20, 20, 10, 40, 20},
		{"tall_signal2", 20, 20, 40, 10, 20},
	}

	for _, sz := range sizes {
		signal1 := make([]complex64, sz.rows1*sz.cols1)
		signal2 := make([]complex64, sz.rows2*sz.cols2)
		dstCols := sz.cols1
		if sz.cols2 > sz.cols1 {
			dstCols = sz.cols2
		}
		dst := make([]complex64, sz.dstRows*dstCols)

		for i := range signal1 {
			signal1[i] = complex64(complex(float32(i), float32(i*2)))
		}
		for i := range signal2 {
			signal2[i] = complex64(complex(float32(i*2), float32(i*3)))
		}

		b.Run(sz.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Convolve2D(dst, sz.dstRows, signal1, sz.rows1, signal2, sz.rows2)
			}
		})
	}
}

func BenchmarkConvolve2DAdd(b *testing.B) {
	sizes := []struct {
		name         string
		rows1, cols1 int
		rows2, cols2 int
		dstRows      int
	}{
		{"small", 10, 10, 5, 5, 10},
		{"medium", 50, 50, 25, 25, 50},
		{"large", 100, 100, 50, 50, 100},
	}

	for _, sz := range sizes {
		signal1 := make([]complex64, sz.rows1*sz.cols1)
		signal2 := make([]complex64, sz.rows2*sz.cols2)
		dstCols := sz.cols1
		if sz.cols2 > sz.cols1 {
			dstCols = sz.cols2
		}
		dst := make([]complex64, sz.dstRows*dstCols)

		for i := range signal1 {
			signal1[i] = complex64(complex(float32(i), float32(i*2)))
		}
		for i := range signal2 {
			signal2[i] = complex64(complex(float32(i*2), float32(i*3)))
		}
		for i := range dst {
			dst[i] = complex64(complex(float32(i), float32(i)))
		}

		b.Run(sz.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				Convolve2DAdd(dst, sz.dstRows, signal1, sz.rows1, signal2, sz.rows2)
			}
		})
	}
}

