package mat

import (
	"errors"

	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
)

var (
	// ErrPseudoInverseFailed is returned when pseudo-inverse computation fails
	ErrPseudoInverseFailed = errors.New("pseudo-inverse computation failed")
)

// PseudoInverse calculates Moore-Penrose pseudo-inverse of a rectangular matrix.
// Uses SVD-based implementation: A^+ = V * Σ^+ * U^T
// Destination matrix must be properly sized (cols × rows).
func (m Matrix) PseudoInverse(dst matTypes.Matrix) error {
	rows := len(m)
	if rows == 0 {
		return ErrPseudoInverseFailed
	}
	cols := len(m[0])
	if cols == 0 {
		return ErrPseudoInverseFailed
	}

	dstMat := dst.View().(Matrix)

	// Flatten matrices (zero-copy if contiguous)
	mFlat := m.Flat()
	dstFlat := dstMat.Flat()
	ldA := len(m[0])
	ldApinv := len(dstMat[0])

	// Use Gepseu for pseudo-inverse computation
	if err := fp32.Gepseu(dstFlat, mFlat, ldA, ldApinv, rows, cols); err != nil {
		return ErrPseudoInverseFailed
	}

	return nil
}

// DampedLeastSquares calculates damped least squares (Levenberg-Marquardt) pseudo-inverse.
// J+ = J^T * (J * J^T + λ^2 * I)^(-1)
// Lambda (λ) is damping factor for better singularity handling.
// Destination matrix must be properly sized (cols × rows).
func (m Matrix) DampedLeastSquares(lambda float32, dst matTypes.Matrix) error {
	rows := len(m)
	if rows == 0 {
		return ErrPseudoInverseFailed
	}
	cols := len(m[0])
	if cols == 0 {
		return ErrPseudoInverseFailed
	}

	dstMat := dst.View().(Matrix)

	// Transpose
	mT := New(cols, rows)
	mT.Transpose(m)

	// J * J^T
	JJT := New(rows, rows)
	JJT.Mul(m, mT)

	// Add lambda^2 * I
	lambda2 := lambda * lambda
	for i := 0; i < rows; i++ {
		JJT[i][i] += lambda2
	}

	// Invert (J * J^T + λ^2 * I)
	JJTInv := New(rows, rows)
	if err := JJT.Inverse(JJTInv); err != nil {
		return ErrPseudoInverseFailed
	}

	// J+ = J^T * (J * J^T + λ^2 * I)^(-1)
	dstMat.Mul(mT, JJTInv)

	return nil
}
