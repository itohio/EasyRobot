package mat

import (
	"errors"
)

var (
	// ErrPseudoInverseFailed is returned when pseudo-inverse computation fails
	ErrPseudoInverseFailed = errors.New("pseudo-inverse computation failed")
)

// PseudoInverse calculates Moore-Penrose pseudo-inverse of a rectangular matrix.
// For overdetermined (rows >= cols): J+ = (J^T * J)^(-1) * J^T
// For underdetermined (rows < cols): J+ = J^T * (J * J^T)^(-1)
// Destination matrix must be properly sized (cols × rows).
func (m Matrix) PseudoInverse(dst Matrix) error {
	rows := len(m)
	if rows == 0 {
		return ErrPseudoInverseFailed
	}
	cols := len(m[0])
	if cols == 0 {
		return ErrPseudoInverseFailed
	}

	// Transpose
	mT := New(cols, rows)
	mT.Transpose(m)

	if rows >= cols {
		// Overdetermined: J+ = (J^T * J)^(-1) * J^T
		JTJ := New(cols, cols)
		JTJ.Mul(mT, m)

		JTJInv := New(cols, cols)
		if err := JTJ.Inverse(JTJInv); err != nil {
			return ErrPseudoInverseFailed
		}

		dst.Mul(JTJInv, mT)
	} else {
		// Underdetermined: J+ = J^T * (J * J^T)^(-1)
		JJT := New(rows, rows)
		JJT.Mul(m, mT)

		JJTInv := New(rows, rows)
		if err := JJT.Inverse(JJTInv); err != nil {
			return ErrPseudoInverseFailed
		}

		dst.Mul(mT, JJTInv)
	}

	return nil
}

// DampedLeastSquares calculates damped least squares (Levenberg-Marquardt) pseudo-inverse.
// J+ = J^T * (J * J^T + λ^2 * I)^(-1)
// Lambda (λ) is damping factor for better singularity handling.
// Destination matrix must be properly sized (cols × rows).
func (m Matrix) DampedLeastSquares(lambda float32, dst Matrix) error {
	rows := len(m)
	if rows == 0 {
		return ErrPseudoInverseFailed
	}
	cols := len(m[0])
	if cols == 0 {
		return ErrPseudoInverseFailed
	}

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
	dst.Mul(mT, JJTInv)

	return nil
}
