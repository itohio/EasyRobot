// Package mat provides Householder transformation functions.
// Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'
// Split into three functions for efficiency.

package mat

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const (
	// DefaultRange is the default range parameter for numerical stability
	// Using a smaller value that fits in float32 (max ~3.4e38)
	DefaultRange = float32(1e30)
)

// H1 constructs Householder transformation.
// Returns transformation parameter 'up'.
// Used by NNLS algorithm.
// col0: column index of the pivot vector
// lpivot: pivot row index
// l1: starting row index for transformation
// rangeVal: regularization parameter (typically 1e306)
func (m Matrix) H1(col0, lpivot, l1 int, rangeVal float32) (float32, error) {
	const one = 1.0
	rangin := one / rangeVal
	var sm, cl, clinv, up float32

	if lpivot < 0 || lpivot >= l1 || l1 >= len(m) {
		return 0, nil
	}

	// Construct transformation
	cl = math32.Abs(m[lpivot][col0])
	for j := l1; j < len(m); j++ {
		cl = FMAX(math32.Abs(m[j][col0]), cl)
	}
	if cl < rangin {
		return 0, nil
	}

	clinv = one / cl
	sm = (m[lpivot][col0] * clinv) * (m[lpivot][col0] * clinv)
	for j := l1; j < len(m); j++ {
		sm += (m[j][col0] * clinv) * (m[j][col0] * clinv)
	}
	cl *= math32.Sqrt(sm)
	if m[lpivot][col0] > 0 {
		cl *= -1
	}
	up = m[lpivot][col0] - cl
	m[lpivot][col0] = cl
	return up, nil
}

// H2 applies Householder transformation to vector.
// Used by NNLS algorithm.
// col0: column index of the pivot vector
// lpivot: pivot row index
// l1: starting row index for transformation
// up: transformation parameter from H1
// zz: vector to transform (modified in place)
// rangeVal: regularization parameter (typically 1e306)
func (m Matrix) H2(col0, lpivot, l1 int, up float32, zz vec.Vector, rangeVal float32) error {
	const one = 1.0
	rangin := one / rangeVal
	var b, sm, cl float32

	if lpivot < 0 || lpivot >= l1 || l1 >= len(m) {
		return nil
	}

	// Apply transformation "I+UU^T/B" to Vector ZZ
	cl = math32.Abs(m[lpivot][col0])
	if cl <= rangin {
		return nil
	}

	b = up * m[lpivot][col0] // b must be nonpositive here
	if b > -rangin {
		return nil
	}
	b = 1 / b

	i2 := lpivot
	i3 := i2 + 1
	i4 := i3

	sm = zz[i2] * up
	for i := l1; i < len(m); i++ {
		sm += zz[i3] * m[i][col0]
		i3++
	}

	if sm == 0 {
		return nil
	}

	sm *= b
	zz[i2] += sm * up
	for i := l1; i < len(m); i++ {
		zz[i4] += sm * m[i][col0]
		i4++
	}
	return nil
}

// H3 applies Householder transformation to matrix column.
// Used by NNLS algorithm.
// col0: column index of the pivot vector
// lpivot: pivot row index
// l1: starting row index for transformation
// up: transformation parameter from H1
// col1: column index to transform
// rangeVal: regularization parameter (typically 1e306)
func (m Matrix) H3(col0, lpivot, l1 int, up float32, col1 int, rangeVal float32) error {
	const one = 1.0
	rangin := one / rangeVal
	var b, sm, cl float32

	if lpivot < 0 || lpivot >= l1 || l1 >= len(m) {
		return nil
	}

	cl = math32.Abs(m[lpivot][col0])
	if cl <= rangin {
		return nil
	}

	// Apply transformation "I+UU^T/B" to Column col1
	b = up * m[lpivot][col0] // b must be nonpositive here
	if b > -rangin {
		return nil
	}
	b = 1 / b

	i2 := lpivot
	i3 := i2 + 1
	i4 := i3

	sm = m[i2][col1] * up
	for i := l1; i < len(m); i++ {
		sm += m[i3][col1] * m[i][col0]
		i3++
	}

	if sm == 0 {
		return nil
	}

	sm *= b
	m[i2][col1] += sm * up
	for i := l1; i < len(m); i++ {
		m[i4][col1] += sm * m[i][col0]
		i4++
	}
	return nil
}

