package primitive

import (
	"github.com/chewxy/math32"
)

// MatMulVec computes matrix-vector product: dst = mat * vec
// vec is vector of size N
// mat is NxM matrix (row-major)
// dst is vector of size M
// transposed: if true, mat is MxN (column-major layout)
// bias: if true, mat has extra row/column for bias
func MatMulVec(dst, vec, mat []float32, N, M int, transposed, bias bool) {
	if N == 0 || M == 0 {
		return
	}

	if !transposed {
		// Row-major: mat[i + j * (N + bias_offset)]
		matRowSize := N
		if bias {
			matRowSize++
		}

		pd := 0
		pm := 0

		for j := 0; j < M; j++ {
			acc := float32(0.0)

			// Compute dot product of vec with row j of mat
			for i := 0; i < N; i++ {
				acc += vec[i] * mat[pm+i]
			}

			// Add bias if present
			if bias {
				acc += mat[pm+N]
			}

			dst[pd] = acc
			pd++
			pm += matRowSize
		}
	} else {
		// Transposed: mat is stored column-major
		// mat[j + i * (N + bias)] where:
		//   j is column index (0 to N-1)
		//   i is row index (0 to M-1)
		//   N is vector size (rows in transposed matrix)
		//   M is output size (columns in transposed matrix)
		matRowStride := N
		if bias {
			matRowStride++
		}

		for j := 0; j < N; j++ {
			acc := float32(0.0)
			pm := j // Start at column j

			// Compute dot product: vec[i] * mat[j + i * matRowStride]
			for i := 0; i < M; i++ {
				acc += vec[i] * mat[pm]
				pm += matRowStride // Move to next row in same column
			}

			// Add bias if present (at end of column)
			if bias {
				acc += mat[pm]
			}

			dst[j] = acc
		}
	}
}

// MatMulVecAdd computes matrix-vector product and add: dst += mat * vec
func MatMulVecAdd(dst, vec, mat []float32, N, M int, transposed, bias bool) {
	if N == 0 || M == 0 {
		return
	}

	if !transposed {
		// Row-major: mat[i + j * (N + bias_offset)]
		matRowSize := N
		if bias {
			matRowSize++
		}

		pd := 0
		pm := 0

		for j := 0; j < M; j++ {
			acc := float32(0.0)

			for i := 0; i < N; i++ {
				acc += vec[i] * mat[pm+i]
			}

			if bias {
				acc += mat[pm+N]
			}

			dst[pd] += acc
			pd++
			pm += matRowSize
		}
	} else {
		// Transposed: mat is stored column-major
		matRowStride := N
		if bias {
			matRowStride++
		}

		for j := 0; j < N; j++ {
			acc := float32(0.0)
			pm := j

			for i := 0; i < M; i++ {
				acc += vec[i] * mat[pm]
				pm += matRowStride
			}

			if bias {
				acc += mat[pm]
			}

			dst[j] += acc
		}
	}
}

// MatTranspose transposes matrix: dst = src^T
// src is width x height matrix (row-major)
// dst is height x width matrix (row-major)
func MatTranspose(dst, src []float32, width, height int) {
	if width == 0 || height == 0 {
		return
	}

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			dst[j*height+i] = src[i*width+j]
		}
	}
}

// MinMat finds minimum value in matrix, returns value and indices
func MinMat(x, y *int, a []float32, inWidth, width, height, stride int) float32 {
	if width == 0 || height == 0 {
		return 0
	}

	min := float32(math32.MaxFloat32)
	minX := 0
	minY := 0
	pa := 0

	for i := 0; i < height; i++ {
		rowStart := pa
		for j := 0; j < width; j++ {
			val := a[rowStart+j*stride]
			if val < min {
				min = val
				minX = j
				minY = i
			}
		}
		pa += inWidth
	}

	*x = minX
	*y = minY
	return min
}

// MaxMat finds maximum value in matrix, returns value and indices
func MaxMat(x, y *int, a []float32, inWidth, width, height, stride int) float32 {
	if width == 0 || height == 0 {
		return 0
	}

	max := float32(-math32.MaxFloat32)
	maxX := 0
	maxY := 0
	pa := 0

	for i := 0; i < height; i++ {
		rowStart := pa
		for j := 0; j < width; j++ {
			val := a[rowStart+j*stride]
			if val > max {
				max = val
				maxX = j
				maxY = i
			}
		}
		pa += inWidth
	}

	*x = maxX
	*y = maxY
	return max
}

// MeanMat computes mean of matrix elements
func MeanMat(a []float32, inWidth, width, height, stride int) float32 {
	if width == 0 || height == 0 {
		return 0
	}

	acc := float32(0.0)
	pa := 0
	count := 0

	for i := 0; i < height; i++ {
		rowStart := pa
		for j := 0; j < width; j++ {
			acc += a[rowStart+j*stride]
			count++
		}
		pa += inWidth
	}

	return acc / float32(count)
}
