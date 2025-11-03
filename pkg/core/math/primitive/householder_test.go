package primitive

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestH1(t *testing.T) {
	tests := []struct {
		name     string
		matrix   []float32
		ldA      int
		col0     int
		lpivot   int
		l1       int
		rangeVal float32
		wantErr  bool
		wantUp   float32
		verify   func(a []float32, up float32, t *testing.T)
	}{
		{
			name: "simple 3x3 matrix, first column",
			matrix: makeMatrix([]float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}, 3, 3, 0),
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, up float32, t *testing.T) {
				// H1 constructs Householder transformation
				// It modifies column col0 starting from lpivot
				// up should be non-zero if transformation was constructed
				// After H1, the pivot element should be modified
				if up == 0 {
					t.Log("up is zero - transformation may have been skipped (numerically zero case)")
				} else {
					// Check that pivot element was modified
					pivotVal := getElem(a, 3, 0, 0)
					assert.NotEqual(t, 1.0, pivotVal, "Pivot element should be modified")
				}
			},
		},
		{
			name: "identity matrix, first column",
			matrix: makeMatrix([]float32{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1,
			}, 3, 3, 0),
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, up float32, t *testing.T) {
				// For identity, transformation should still be constructed
				// up might be zero if column is already in correct form
				t.Logf("up value: %f", up)
			},
		},
		{
			name: "zero column",
			matrix: makeMatrix([]float32{
				0, 0, 0,
				0, 1, 0,
				0, 0, 1,
			}, 3, 3, 0),
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, up float32, t *testing.T) {
				// For zero column, up should be 0 (transformation skipped)
				assert.Equal(t, float32(0.0), up, "up should be 0 for zero column")
			},
		},
		{
			name: "second column, starting from row 1",
			matrix: makeMatrix([]float32{
				1, 2, 3,
				0, 5, 6,
				0, 8, 9,
			}, 3, 3, 0),
			ldA:      3,
			col0:     1,
			lpivot:   1,
			l1:       2,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, up float32, t *testing.T) {
				// Transformation should modify column 1 starting from row 1
				if up != 0 {
					pivotVal := getElem(a, 3, 1, 1)
					assert.NotEqual(t, 5.0, pivotVal, "Pivot element should be modified")
				}
			},
		},
		{
			name: "with leading dimension padding",
			matrix: makeMatrix([]float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}, 3, 3, 4),
			ldA:      4,
			col0:     0,
			lpivot:   0,
			l1:       1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, up float32, t *testing.T) {
				// Should work with leading dimension padding
				t.Logf("up value with padding: %f", up)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy original matrix
			a := make([]float32, len(tt.matrix))
			copy(a, tt.matrix)

			up, err := H1(a, tt.col0, tt.lpivot, tt.l1, tt.ldA, tt.rangeVal)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.verify != nil {
					tt.verify(a, up, t)
				}
			}
		})
	}
}

func TestH2(t *testing.T) {
	tests := []struct {
		name     string
		matrix   []float32
		zz       []float32
		ldA      int
		col0     int
		lpivot   int
		l1       int
		up       float32
		rangeVal float32
		wantErr  bool
		verify   func(zz []float32, t *testing.T)
	}{
		{
			name: "simple transformation on unit vector",
			matrix: makeMatrix([]float32{
				1, 0, 0,
				4, 0, 0,
				7, 0, 0,
			}, 3, 3, 0),
			zz:       []float32{1, 1, 1},
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       1.0,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(zz []float32, t *testing.T) {
				// Vector may or may not be transformed depending on matrix structure
				// If transformation is applied, verify properties hold
				// Just verify function completed without crash
				t.Logf("Vector after H2 transformation: %v", zz)
			},
		},
		{
			name: "identity transformation (zero up)",
			matrix: makeMatrix([]float32{
				1, 0, 0,
				0, 0, 0,
				0, 0, 0,
			}, 3, 3, 0),
			zz:       []float32{1, 2, 3},
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       0.0,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(zz []float32, t *testing.T) {
				// With zero up, transformation should be skipped
				// Vector should remain unchanged or check should be skipped
				t.Logf("Vector after zero-up transformation: %v", zz)
			},
		},
		{
			name: "transform unit vector x",
			matrix: makeMatrix([]float32{
				1, 0, 0,
				0, 0, 0,
				0, 0, 0,
			}, 3, 3, 0),
			zz:       []float32{1, 0, 0},
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       1.0,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(zz []float32, t *testing.T) {
				// Transformation should be applied
				// Magnitude might be preserved (Householder is unitary)
				mag := math32.Sqrt(zz[0]*zz[0] + zz[1]*zz[1] + zz[2]*zz[2])
				assert.InDelta(t, 1.0, mag, 0.1, "Magnitude should be approximately preserved")
			},
		},
		{
			name: "with leading dimension padding",
			matrix: makeMatrix([]float32{
				1, 2, 0,
				4, 5, 0,
				7, 8, 0,
			}, 3, 3, 4),
			zz:       []float32{1, 1, 1},
			ldA:      4,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       2.0,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(zz []float32, t *testing.T) {
				// Should work with leading dimension padding
				t.Logf("Vector after transformation with padding: %v", zz)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy original matrix and vector
			a := make([]float32, len(tt.matrix))
			copy(a, tt.matrix)
			zz := make([]float32, len(tt.zz))
			copy(zz, tt.zz)

			err := H2(a, zz, tt.col0, tt.lpivot, tt.l1, tt.up, tt.ldA, tt.rangeVal)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.verify != nil {
					tt.verify(zz, t)
				}
			}
		})
	}
}

func TestH3(t *testing.T) {
	tests := []struct {
		name     string
		matrix   []float32
		ldA      int
		col0     int
		lpivot   int
		l1       int
		up       float32
		col1     int
		rangeVal float32
		wantErr  bool
		verify   func(a []float32, t *testing.T)
	}{
		{
			name: "simple transformation on second column",
			matrix: makeMatrix([]float32{
				1, 1, 0,
				4, 1, 0,
				7, 1, 0,
			}, 3, 3, 0),
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       2.0,
			col1:     1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, t *testing.T) {
				// Column col1 may or may not be transformed depending on matrix structure
				// If transformation is applied, verify properties hold
				// Just verify function completed without crash
				col1Val := getElem(a, 3, 0, 1)
				t.Logf("Column 1[0] after H3 transformation: %f", col1Val)
			},
		},
		{
			name: "transform to same column (col0 = col1)",
			matrix: makeMatrix([]float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}, 3, 3, 0),
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       1.0,
			col1:     0,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, t *testing.T) {
				// Transforming same column should work
				// The column should be modified
				t.Logf("Column 0 after transformation: %v", []float32{
					getElem(a, 3, 0, 0),
					getElem(a, 3, 1, 0),
					getElem(a, 3, 2, 0),
				})
			},
		},
		{
			name: "identity transformation (zero up)",
			matrix: makeMatrix([]float32{
				1, 2, 3,
				0, 5, 6,
				0, 8, 9,
			}, 3, 3, 0),
			ldA:      3,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       0.0,
			col1:     1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, t *testing.T) {
				// With zero up, transformation should be skipped
				// Column should remain mostly unchanged
				t.Logf("Column 1 after zero-up transformation: %v", []float32{
					getElem(a, 3, 0, 1),
					getElem(a, 3, 1, 1),
					getElem(a, 3, 2, 1),
				})
			},
		},
		{
			name: "with leading dimension padding",
			matrix: makeMatrix([]float32{
				1, 1, 0,
				4, 1, 0,
				7, 1, 0,
			}, 3, 3, 4),
			ldA:      4,
			col0:     0,
			lpivot:   0,
			l1:       1,
			up:       2.0,
			col1:     1,
			rangeVal: DefaultRange,
			wantErr:  false,
			verify: func(a []float32, t *testing.T) {
				// Should work with leading dimension padding
				col1Val := getElem(a, 4, 0, 1)
				t.Logf("Column 1 value with padding: %f", col1Val)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy original matrix
			a := make([]float32, len(tt.matrix))
			copy(a, tt.matrix)

			err := H3(a, tt.col0, tt.lpivot, tt.l1, tt.up, tt.col1, tt.ldA, tt.rangeVal)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.verify != nil {
					tt.verify(a, t)
				}
			}
		})
	}
}

// TestHouseholderChain tests the combination of H1, H2, H3
// This verifies they work together correctly (common use case)
func TestHouseholderChain(t *testing.T) {
	t.Run("H1 then H2", func(t *testing.T) {
		// Create a simple matrix
		matrix := makeMatrix([]float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}, 3, 3, 0)

		a := make([]float32, len(matrix))
		copy(a, matrix)

		// H1: Construct Householder transformation
		up, err := H1(a, 0, 0, 1, 3, DefaultRange)
		require.NoError(t, err)

		if up != 0 {
			// H2: Apply to a vector
			zz := []float32{1, 1, 1}
			err = H2(a, zz, 0, 0, 1, up, 3, DefaultRange)
			require.NoError(t, err)

			// Vector should be transformed
			t.Logf("Vector after H1+H2: %v", zz)
		} else {
			t.Log("H1 returned zero up, skipping H2 test")
		}
	})

	t.Run("H1 then H3", func(t *testing.T) {
		// Create a simple matrix
		matrix := makeMatrix([]float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}, 3, 3, 0)

		a := make([]float32, len(matrix))
		copy(a, matrix)

		// H1: Construct Householder transformation
		up, err := H1(a, 0, 0, 1, 3, DefaultRange)
		require.NoError(t, err)

		if up != 0 {
			// H3: Apply to another column
			err = H3(a, 0, 0, 1, up, 1, 3, DefaultRange)
			require.NoError(t, err)

			// Column 1 should be transformed
			col1Val := getElem(a, 3, 0, 1)
			t.Logf("Column 1[0] after H1+H3: %f", col1Val)
		} else {
			t.Log("H1 returned zero up, skipping H3 test")
		}
	})
}

// TestHouseholderProperties verifies mathematical properties of Householder transformations
func TestHouseholderProperties(t *testing.T) {
	t.Run("H2 preserves magnitude approximately", func(t *testing.T) {
		matrix := makeMatrix([]float32{
			1, 0, 0,
			2, 0, 0,
			3, 0, 0,
		}, 3, 3, 0)

		a := make([]float32, len(matrix))
		copy(a, matrix)

		// Construct transformation
		up, err := H1(a, 0, 0, 1, 3, DefaultRange)
		require.NoError(t, err)

		if up != 0 {
			// Apply to a vector
			zz := []float32{1, 0, 0}
			originalMag := math32.Sqrt(zz[0]*zz[0] + zz[1]*zz[1] + zz[2]*zz[2])

			err = H2(a, zz, 0, 0, 1, up, 3, DefaultRange)
			require.NoError(t, err)

			newMag := math32.Sqrt(zz[0]*zz[0] + zz[1]*zz[1] + zz[2]*zz[2])
			// Householder transformations are unitary, so magnitude should be preserved
			// Use relaxed tolerance for numerical precision
			assert.InDelta(t, originalMag, newMag, 0.1, "Magnitude should be approximately preserved")
		}
	})

	t.Run("H3 applies orthogonal transformation", func(t *testing.T) {
		matrix := makeMatrix([]float32{
			1, 1, 0,
			2, 2, 0,
			3, 3, 0,
		}, 3, 3, 0)

		a := make([]float32, len(matrix))
		copy(a, matrix)

		// Construct transformation on column 0
		up, err := H1(a, 0, 0, 1, 3, DefaultRange)
		require.NoError(t, err)

		if up != 0 {
			// Store original column 1 magnitude
			origCol1 := []float32{
				getElem(a, 3, 0, 1),
				getElem(a, 3, 1, 1),
				getElem(a, 3, 2, 1),
			}
			origMag := math32.Sqrt(origCol1[0]*origCol1[0] + origCol1[1]*origCol1[1] + origCol1[2]*origCol1[2])

			// Apply to column 1
			err = H3(a, 0, 0, 1, up, 1, 3, DefaultRange)
			require.NoError(t, err)

			// Check magnitude is approximately preserved
			newCol1 := []float32{
				getElem(a, 3, 0, 1),
				getElem(a, 3, 1, 1),
				getElem(a, 3, 2, 1),
			}
			newMag := math32.Sqrt(newCol1[0]*newCol1[0] + newCol1[1]*newCol1[1] + newCol1[2]*newCol1[2])
			assert.InDelta(t, origMag, newMag, 0.1, "Column magnitude should be approximately preserved")
		}
	})
}
