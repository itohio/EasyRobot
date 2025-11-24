package c64

import (
	"math/cmplx"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCrossCorrelate1D(t *testing.T) {
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
			// R_xy(τ) = Σ signal1[n] * signal2[τ+n]
			// i=0: n=0: signal2[0]=1 → 1*1=1; n=1,2,3: signal2[1,2,3]=0 → 0
			// i=1,2,3: all signal2 indices invalid → 0
			expected: []complex64{1, 0, 0, 0, 0, 0, 0},
		},
		{
			name:    "simple",
			signal1: []complex64{1, 2, 3},
			signal2: []complex64{1, 2},
			// R_xy(τ) = Σ signal1[n] * signal2[τ+n]
			// i=0: n=0: signal2[0]=1 → 1*1=1; n=1: signal2[1]=2 → 2*2=4; n=2: signal2[2] invalid
			//      dst[0] = 1 + 4 = 5
			// i=1: n=0: signal2[1]=2 → 1*2=2; n=1,2: signal2[2,3] invalid
			//      dst[1] = 2
			// i=2,3: all signal2 indices invalid → 0
			expected: []complex64{5, 2, 0, 0},
		},
		{
			name:    "single_element",
			signal1: []complex64{5},
			signal2: []complex64{3},
			// i=0: n=0: signal2[0]=3 → 5*3=15
			expected: []complex64{15},
		},
		{
			name:    "signal2_longer",
			signal1: []complex64{1, 2},
			signal2: []complex64{1, 2, 3, 4},
			// R_xy(τ) = Σ signal1[n] * signal2[τ+n]
			// i=0: n=0: signal2[0]=1 → 1*1=1; n=1: signal2[1]=2 → 2*2=4
			//      dst[0] = 1 + 4 = 5
			// i=1: n=0: signal2[1]=2 → 1*2=2; n=1: signal2[2]=3 → 2*3=6
			//      dst[1] = 2 + 6 = 8
			// i=2: n=0: signal2[2]=3 → 1*3=3; n=1: signal2[3]=4 → 2*4=8
			//      dst[2] = 3 + 8 = 11
			// i=3: n=0: signal2[3]=4 → 1*4=4; n=1: signal2[4] invalid
			//      dst[3] = 4
			// i=4: n=0: signal2[4] invalid
			//      dst[4] = 0
			expected: []complex64{5, 8, 11, 4, 0},
		},
		{
			name:    "complex_values",
			signal1: []complex64{1 + 2i, 3 + 4i},
			signal2: []complex64{1 + 1i, 2 + 2i},
			// i=0: n=0: signal2[0]=1+1i → (1+2i)*(1+1i) = 1+1i+2i+2i² = 1+3i-2 = -1+3i
			//      n=1: signal2[1]=2+2i → (3+4i)*(2+2i) = 6+6i+8i+8i² = 6+14i-8 = -2+14i
			//      dst[0] = (-1+3i) + (-2+14i) = -3+17i
			// i=1: n=0: signal2[1]=2+2i → (1+2i)*(2+2i) = 2+2i+4i+4i² = 2+6i-4 = -2+6i
			//      dst[1] = -2+6i
			// i=2: n=0: signal2[2] invalid
			//      dst[2] = 0
			expected: []complex64{-3 + 17i, -2 + 6i, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.signal1)+len(tt.signal2)-1)
			CrossCorrelate1D(dst, tt.signal1, tt.signal2)

			assert.Equal(t, len(tt.expected), len(dst))
			for i, exp := range tt.expected {
				complexInDelta(t, exp, dst[i], 1e-6, "Correlation at index %d", i)
			}
		})
	}
}

func TestCrossCorrelate1D_Empty(t *testing.T) {
	dst := make([]complex64, 10)
	CrossCorrelate1D(dst, []complex64{}, []complex64{1, 2})
	// Should not panic, just return early
}

func TestCrossCorrelate1D_SmallDst(t *testing.T) {
	signal1 := []complex64{1, 2, 3, 4, 5}
	signal2 := []complex64{1, 2, 3}
	dst := make([]complex64, 3) // Smaller than needed (7)

	CrossCorrelate1D(dst, signal1, signal2)
	// Should fill what it can without panicking
}

func TestCrossCorrelate1DAdd(t *testing.T) {
	tests := []struct {
		name     string
		dst      []complex64
		signal1  []complex64
		signal2  []complex64
		expected []complex64
	}{
		{
			name:    "accumulation",
			dst:     []complex64{10, 20, 30, 40, 50},
			signal1: []complex64{1, 2, 3},
			signal2: []complex64{1, 2},
			// Cross-correlation: [5, 2, 0, 0, 0]
			// dst + result: [10+5, 20+2, 30+0, 40+0, 50+0] = [15, 22, 30, 40, 50]
			expected: []complex64{15, 22, 30, 40, 50},
		},
		{
			name:    "zero_dst",
			dst:     make([]complex64, 5),
			signal1: []complex64{1, 2, 3},
			signal2: []complex64{1, 2},
			// Same as CrossCorrelate1D
			expected: []complex64{5, 2, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.dst))
			copy(dst, tt.dst)
			CrossCorrelate1DAdd(dst, tt.signal1, tt.signal2)

			for i, exp := range tt.expected {
				complexInDelta(t, exp, dst[i], 1e-6, "CorrelationAdd at index %d", i)
			}
		})
	}
}

func TestCrossCorrelate2D(t *testing.T) {
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
				// R_xy(τ_r, τ_c) = Σ signal1[k,l] * signal2[τ_r+k, τ_c+l]
				// For (i,j)=(0,0): k=0,l=0: signal2[0,0]=1 → 1*1=1; k=0,l=1: signal2[0,1]=2 → 2*2=4
				//                  k=1,l=0: signal2[1,0] invalid; k=1,l=1: signal2[1,1] invalid
				//                  dst[0] = 1 + 4 = 5
				// For (i,j)=(0,1): k=0,l=0: signal2[0,1]=2 → 1*2=2; k=0,l=1: signal2[0,2] invalid
				//                  dst[1] = 2
				// For (i,j)=(1,0): k=0,l=0: signal2[1,0] invalid
				//                  dst[2] = 0
				// For (i,j)=(1,1): k=0,l=0: signal2[1,1] invalid
				//                  dst[3] = 0
				complexInDelta(t, 5.0, dst[0], 1e-6, "Correlation at (0,0)")
				complexInDelta(t, 2.0, dst[1], 1e-6, "Correlation at (0,1)")
			},
		},
		{
			name:    "identity_3x3",
			signal1: []complex64{1, 0, 0, 0, 1, 0, 0, 0, 1}, // 3x3 identity
			rows1:   3,
			signal2: []complex64{1, 0, 0, 0, 1, 0, 0, 0, 1}, // 3x3 identity
			rows2:   3,
			dstRows: 5,
			validate: func(t *testing.T, dst []complex64, rows, cols int) {
				// Center should have high correlation (at position where they align)
				// For (i,j)=(2,2): all elements align → 1*1 + 1*1 + 1*1 = 3
				centerIdx := 2*cols + 2
				if centerIdx < len(dst) {
					assert.Greater(t, cmplx.Abs(complex128(dst[centerIdx])), 0.0, "Center should have positive correlation")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cols1 := len(tt.signal1) / tt.rows1
			cols2 := len(tt.signal2) / tt.rows2
			dstCols := cols1 + cols2 - 1
			dst := make([]complex64, tt.dstRows*dstCols)

			CrossCorrelate2D(dst, tt.dstRows, tt.signal1, tt.rows1, tt.signal2, tt.rows2)

			tt.validate(t, dst, tt.dstRows, dstCols)
		})
	}
}

func TestCrossCorrelate2D_Panic(t *testing.T) {
	t.Run("zero_rows1", func(t *testing.T) {
		signal1 := []complex64{1, 2}
		signal2 := []complex64{1, 2}
		dst := make([]complex64, 10)

		assert.Panics(t, func() {
			CrossCorrelate2D(dst, 2, signal1, 0, signal2, 1)
		})
	})

	t.Run("zero_rows2", func(t *testing.T) {
		signal1 := []complex64{1, 2}
		signal2 := []complex64{1, 2}
		dst := make([]complex64, 10)

		assert.Panics(t, func() {
			CrossCorrelate2D(dst, 2, signal1, 1, signal2, 0)
		})
	})

	t.Run("dimension_mismatch", func(t *testing.T) {
		signal1 := []complex64{1, 2, 3} // 1x3, but rows1=2 would be wrong
		signal2 := []complex64{1, 2}
		dst := make([]complex64, 10)

		assert.Panics(t, func() {
			CrossCorrelate2D(dst, 2, signal1, 2, signal2, 1) // rows1=2 but len(signal1)=3, not divisible
		})
	})

	t.Run("dst_too_small", func(t *testing.T) {
		signal1 := []complex64{1, 2, 3, 4} // 2x2
		signal2 := []complex64{1, 2}       // 1x2
		dst := make([]complex64, 1)        // Too small

		assert.Panics(t, func() {
			CrossCorrelate2D(dst, 2, signal1, 2, signal2, 1)
		})
	})
}

func TestCrossCorrelate2DAdd(t *testing.T) {
	t.Run("accumulation", func(t *testing.T) {
		signal1 := []complex64{1, 2, 3, 4}         // 2x2
		signal2 := []complex64{1, 2}               // 1x2
		dst := []complex64{10, 20, 30, 40, 50, 60} // 2x3 pre-filled
		// Cross-correlation result: [5, 2, 0, 0, 0, 0] (2x3)
		// dst + result: [10+5, 20+2, 30+0, 40+0, 50+0, 60+0] = [15, 22, 30, 40, 50, 60]
		expected := []complex64{15, 22, 30, 40, 50, 60}

		CrossCorrelate2DAdd(dst, 2, signal1, 2, signal2, 1)

		for i, exp := range expected {
			complexInDelta(t, exp, dst[i], 1e-6, "CorrelationAdd at index %d", i)
		}
	})
}

func BenchmarkCrossCorrelate1D(b *testing.B) {
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
				CrossCorrelate1D(dst, signal1, signal2)
			}
		})
	}
}

func BenchmarkCrossCorrelate1DAdd(b *testing.B) {
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
				CrossCorrelate1DAdd(dst, signal1, signal2)
			}
		})
	}
}

func BenchmarkCrossCorrelate2D(b *testing.B) {
	sizes := []struct {
		name         string
		rows1, cols1 int
		rows2, cols2 int
		dstRows      int
	}{
		{"small", 10, 10, 5, 5, 14},
		{"medium", 50, 50, 25, 25, 74},
		{"large", 100, 100, 50, 50, 149},
		{"wide_signal2", 20, 20, 10, 40, 29},
		{"tall_signal2", 20, 20, 40, 10, 59},
	}

	for _, sz := range sizes {
		signal1 := make([]complex64, sz.rows1*sz.cols1)
		signal2 := make([]complex64, sz.rows2*sz.cols2)
		dstCols := sz.cols1 + sz.cols2 - 1
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
				CrossCorrelate2D(dst, sz.dstRows, signal1, sz.rows1, signal2, sz.rows2)
			}
		})
	}
}

func BenchmarkCrossCorrelate2DAdd(b *testing.B) {
	sizes := []struct {
		name         string
		rows1, cols1 int
		rows2, cols2 int
		dstRows      int
	}{
		{"small", 10, 10, 5, 5, 14},
		{"medium", 50, 50, 25, 25, 74},
		{"large", 100, 100, 50, 50, 149},
	}

	for _, sz := range sizes {
		signal1 := make([]complex64, sz.rows1*sz.cols1)
		signal2 := make([]complex64, sz.rows2*sz.cols2)
		dstCols := sz.cols1 + sz.cols2 - 1
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
				CrossCorrelate2DAdd(dst, sz.dstRows, signal1, sz.rows1, signal2, sz.rows2)
			}
		})
	}
}
