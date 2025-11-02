package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatMulVec(t *testing.T) {
	vec := []float32{1, 2}
	mat := []float32{
		1, 2, // Row 0
		3, 4, // Row 1
		5, 6, // Row 2
	}
	dst := make([]float32, 3)

	// mat is 2x3 (N=2, M=3), vec is 2x1
	MatMulVec(dst, vec, mat, 2, 3, false, false)
	// Result should be: [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
	assert.Equal(t, []float32{5, 11, 17}, dst)
}

func TestMatMulVecWithBias(t *testing.T) {
	vec := []float32{1, 2}
	mat := []float32{
		1, 2, 1, // Row 0 with bias
		3, 4, 2, // Row 1 with bias
		5, 6, 3, // Row 2 with bias
	}
	dst := make([]float32, 3)

	MatMulVec(dst, vec, mat, 2, 3, false, true)
	// Result should be: [1*1+2*2+1, 1*3+2*4+2, 1*5+2*6+3] = [6, 13, 20]
	assert.Equal(t, []float32{6, 13, 20}, dst)
}

func TestMatMulVecTransposed(t *testing.T) {
	// From C++ reference: when TRANSPOSED=true
	// vec is size M (input vector)
	// mat is N rows x M columns (stored column-major)
	// dst is size N (output)
	// Computation: dst[j] = sum_i(vec[i] * mat[j + i * (N + bias)])

	// Example: vec=[1,2] (M=2), mat is 3x2 (N=3, M=2) stored column-major
	vec := []float32{1, 2} // M=2
	// 3x2 matrix stored column-major:
	// mat = [1, 4, 2, 5, 3, 6] where:
	//   Column 0: [1, 2, 3]
	//   Column 1: [4, 5, 6]
	// Access: mat[j + i * N] where j=0..2, i=0..1
	mat := []float32{
		1, 2, 3, // j=0: [1, 2, 3] (column 0)
		4, 5, 6, // j=1: [4, 5, 6] (column 1)
	}
	dst := make([]float32, 3) // N=3

	// N=3 (matrix rows), M=2 (vec size), transposed=true
	MatMulVec(dst, vec, mat, 3, 2, true, false)
	// Result:
	//   dst[0] = vec[0]*mat[0+0*3] + vec[1]*mat[0+1*3] = 1*1 + 2*4 = 9
	//   dst[1] = vec[0]*mat[1+0*3] + vec[1]*mat[1+1*3] = 1*2 + 2*5 = 12
	//   dst[2] = vec[0]*mat[2+0*3] + vec[1]*mat[2+1*3] = 1*3 + 2*6 = 15
	assert.Equal(t, []float32{9, 12, 15}, dst)
}

func TestMatTranspose(t *testing.T) {
	src := []float32{
		1, 2, // Row 0
		3, 4, // Row 1
		5, 6, // Row 2
	}
	dst := make([]float32, 6)

	MatTranspose(dst, src, 2, 3)
	// Transpose of 3x2 should be 2x3
	// dst should be: [1, 3, 5, 2, 4, 6]
	expected := []float32{1, 3, 5, 2, 4, 6}
	assert.Equal(t, expected, dst)
}
