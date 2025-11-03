package primitive

import (
	"errors"

	"github.com/chewxy/math32"
)

// Default range value for numerical stability in Householder transforms
const DefaultRange = float32(1e30)

var (
	// ErrSingularMatrix is returned when trying to invert a singular matrix
	ErrSingularMatrix = errors.New("matrix is singular")
	// ErrNotSquare is returned when trying to invert a non-square matrix
	ErrNotSquare = errors.New("matrix must be square for inverse")
	// ErrBadDimensions is returned when dimensions are invalid
	ErrBadDimensions = errors.New("bad matrix dimensions")
	// ErrMaxIterations is returned when maximum iterations exceeded
	ErrMaxIterations = errors.New("maximum iterations exceeded")
)

// Helper functions for numerical computations

// sign returns the sign of b times the absolute value of a.
// Used in Householder transformations and other matrix algorithms.
func sign(a, b float32) float32 {
	if b >= 0.0 {
		return math32.Abs(a)
	}
	return -math32.Abs(a)
}

// fmax returns the maximum of two float32 values.
func fmax(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// fmin returns the minimum of two float32 values.
func fmin(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// imin returns the minimum of two int values.
func imin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Get matrix element at row i, column j (row-major storage)
func getElem(a []float32, ldA, i, j int) float32 {
	return a[i*ldA+j]
}

// Set matrix element at row i, column j (row-major storage)
func setElem(a []float32, ldA, i, j int, val float32) {
	a[i*ldA+j] = val
}

// Swap rows i and j in matrix A (row-major storage)
func swapRows(a []float32, ldA, i, j, N int) {
	for k := 0; k < N; k++ {
		a[i*ldA+k], a[j*ldA+k] = a[j*ldA+k], a[i*ldA+k]
	}
}

// H1: Construct Householder transformation
// Computes transformation parameter 'up' for Householder reflector
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//
//	col0: column index of the pivot vector
//	lpivot: pivot row index
//	l1: starting row index for transformation
//	rangeVal: regularization parameter for numerical stability (typically 1e30)
//
// Output: up: transformation parameter (returned)
//
//	a: modified (pivot element contains transformation value)
func H1(a []float32, col0, lpivot, l1 int, ldA int, rangeVal float32) (up float32, err error) {
	const one = 1.0
	rangin := one / rangeVal
	var sm, cl, clinv float32

	if lpivot < 0 || lpivot >= l1 || l1*ldA+col0 >= len(a) {
		return 0, nil
	}

	M := len(a) / ldA
	if M == 0 {
		return 0, nil
	}

	if l1 > M {
		l1 = M
	}

	// Construct transformation
	cl = math32.Abs(getElem(a, ldA, lpivot, col0))
	for j := l1; j < M; j++ {
		if j*ldA+col0 < len(a) {
			cl = fmax(cl, math32.Abs(getElem(a, ldA, j, col0)))
		}
	}
	if cl < rangin {
		return 0, nil
	}

	clinv = one / cl
	sm = (getElem(a, ldA, lpivot, col0) * clinv) * (getElem(a, ldA, lpivot, col0) * clinv)
	for j := l1; j < M; j++ {
		if j*ldA+col0 < len(a) {
			val := getElem(a, ldA, j, col0) * clinv
			sm += val * val
		}
	}
	cl *= math32.Sqrt(sm)
	if getElem(a, ldA, lpivot, col0) > 0 {
		cl *= -1
	}
	up = getElem(a, ldA, lpivot, col0) - cl
	setElem(a, ldA, lpivot, col0, cl)
	return up, nil
}

// H2: Apply Householder transformation to vector
// Applies transformation I + u*u^T/b to vector zz
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//
//	zz: vector to transform (modified in-place)
//	col0: column index of the pivot vector in a
//	lpivot: pivot row index
//	l1: starting row index for transformation
//	up: transformation parameter from H1
//	rangeVal: regularization parameter
func H2(a, zz []float32, col0, lpivot, l1 int, up float32, ldA int, rangeVal float32) error {
	const one = 1.0
	rangin := one / rangeVal
	var b, sm, cl float32

	if lpivot < 0 || lpivot >= l1 || len(zz) == 0 || lpivot*ldA+col0 >= len(a) {
		return nil
	}

	M := len(a) / ldA
	if M == 0 {
		return nil
	}

	if l1 > M {
		l1 = M
	}
	if len(zz) < M {
		return ErrBadDimensions
	}

	// Apply transformation "I+UU^T/B" to Vector ZZ
	cl = math32.Abs(getElem(a, ldA, lpivot, col0))
	if cl <= rangin {
		return nil
	}

	b = up * getElem(a, ldA, lpivot, col0) // b must be nonpositive here
	if b > -rangin {
		return nil
	}
	b = 1 / b

	i2 := lpivot
	i3 := i2 + 1
	i4 := i3

	sm = zz[i2] * up
	for i := l1; i < M && i3 < len(zz); i++ {
		if i*ldA+col0 < len(a) {
			sm += zz[i3] * getElem(a, ldA, i, col0)
		}
		i3++
	}

	if sm == 0 {
		return nil
	}

	sm *= b
	zz[i2] += sm * up
	for i := l1; i < M && i4 < len(zz); i++ {
		if i*ldA+col0 < len(a) {
			zz[i4] += sm * getElem(a, ldA, i, col0)
		}
		i4++
	}
	return nil
}

// H3: Apply Householder transformation to matrix column
// Applies transformation I + u*u^T/b to column col1 of matrix a
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//
//	col0: column index of the pivot vector (contains Householder vector)
//	lpivot: pivot row index
//	l1: starting row index for transformation
//	up: transformation parameter from H1
//	col1: column index to transform
//	rangeVal: regularization parameter
//
// Output: a: column col1 is transformed in-place
func H3(a []float32, col0, lpivot, l1 int, up float32, col1 int, ldA int, rangeVal float32) error {
	const one = 1.0
	rangin := one / rangeVal
	var b, sm, cl float32

	if lpivot < 0 || lpivot >= l1 || lpivot*ldA+col0 >= len(a) || lpivot*ldA+col1 >= len(a) {
		return nil
	}

	M := len(a) / ldA
	if M == 0 {
		return nil
	}

	if l1 > M {
		l1 = M
	}

	cl = math32.Abs(getElem(a, ldA, lpivot, col0))
	if cl <= rangin {
		return nil
	}

	// Apply transformation "I+UU^T/B" to Column col1
	b = up * getElem(a, ldA, lpivot, col0) // b must be nonpositive here
	if b > -rangin {
		return nil
	}
	b = 1 / b

	i2 := lpivot
	i3 := i2 + 1
	i4 := i3

	sm = getElem(a, ldA, i2, col1) * up
	for i := l1; i < M; i++ {
		if i*ldA+col0 < len(a) && i*ldA+col1 < len(a) {
			sm += getElem(a, ldA, i3, col1) * getElem(a, ldA, i, col0)
		}
		i3++
	}

	if sm == 0 {
		return nil
	}

	sm *= b
	setElem(a, ldA, i2, col1, getElem(a, ldA, i2, col1)+sm*up)
	for i := l1; i < M; i++ {
		if i*ldA+col0 < len(a) && i*ldA+col1 < len(a) {
			val := getElem(a, ldA, i4, col1) + sm*getElem(a, ldA, i, col0)
			setElem(a, ldA, i4, col1, val)
		}
		i4++
	}
	return nil
}

// GETRF_IP: Compute LU decomposition with partial pivoting (in-place)
// On input: a contains M × N matrix
// On output: a contains L (below diagonal) and U (on/above diagonal)
//
//	ipiv contains pivot indices (length min(M,N))
func Getrf_IP(a []float32, ipiv []int, ldA, M, N int) error {
	if M <= 0 || N <= 0 {
		return ErrBadDimensions
	}
	if len(a) < M*ldA {
		return ErrBadDimensions
	}
	if len(ipiv) < imin(M, N) {
		return ErrBadDimensions
	}

	minMN := imin(M, N)
	var i, j, k, p int
	var maxVal, val float32

	// Initialize pivot array
	for i = 0; i < minMN; i++ {
		ipiv[i] = i
	}

	// LU decomposition with partial pivoting
	for k = 0; k < minMN; k++ {
		// Find pivot
		p = k
		maxVal = math32.Abs(getElem(a, ldA, k, k))
		for i = k + 1; i < M; i++ {
			val = math32.Abs(getElem(a, ldA, i, k))
			if val > maxVal {
				maxVal = val
				p = i
			}
		}
		ipiv[k] = p

		// Swap rows if needed
		if p != k {
			swapRows(a, ldA, k, p, N)
		}

		// Check for singularity
		akk := getElem(a, ldA, k, k)
		if math32.Abs(akk) < 1e-6 {
			return ErrSingularMatrix
		}

		// Compute multipliers and update matrix
		for i = k + 1; i < M; i++ {
			aik := getElem(a, ldA, i, k) / akk
			setElem(a, ldA, i, k, aik) // Store L factor

			// Update submatrix
			for j = k + 1; j < N; j++ {
				val = getElem(a, ldA, i, j) - aik*getElem(a, ldA, k, j)
				setElem(a, ldA, i, j, val)
			}
		}
	}

	return nil
}

// GETRF: Compute LU decomposition with partial pivoting
// A = P * L * U
// On input: a contains M × N matrix (row-major, ldA ≥ N)
// On output: l contains M × min(M,N) lower triangular matrix (unit diagonal)
//
//	u contains min(M,N) × N upper triangular matrix
//	ipiv contains pivot indices (length min(M,N))
func Getrf(a, l, u []float32, ipiv []int, ldA, ldL, ldU, M, N int) error {
	if M <= 0 || N <= 0 {
		return ErrBadDimensions
	}
	if len(a) < M*ldA {
		return ErrBadDimensions
	}
	if len(l) < M*ldL || len(u) < imin(M, N)*ldU {
		return ErrBadDimensions
	}
	if len(ipiv) < imin(M, N) {
		return ErrBadDimensions
	}

	minMN := imin(M, N)
	var i, j int

	// Copy a to work space (we'll modify it)
	work := make([]float32, M*ldA)
	copy(work, a)

	// Perform LU with pivoting on work
	if err := Getrf_IP(work, ipiv, ldA, M, N); err != nil {
		return err
	}

	// Extract L and U from work
	for i = 0; i < M; i++ {
		for j = 0; j < minMN; j++ {
			if i > j {
				// Lower triangular (L)
				setElem(l, ldL, i, j, getElem(work, ldA, i, j))
			} else if i == j {
				// Diagonal - L has ones, U has values
				setElem(l, ldL, i, j, 1.0)
				setElem(u, ldU, i, j, getElem(work, ldA, i, j))
			}
		}
	}

	// Extract U upper triangular part
	for i = 0; i < minMN; i++ {
		for j = i; j < N; j++ {
			setElem(u, ldU, i, j, getElem(work, ldA, i, j))
		}
	}

	return nil
}

// GETRI: Compute inverse using LU decomposition
// A^(-1) = U^(-1) * L^(-1) * P^T
// Input: a contains LU decomposition from GETRF_IP, ipiv contains pivots
// Output: aInv contains the inverse
func Getri(aInv, a []float32, ldA, ldInv, N int, ipiv []int) error {
	if N <= 0 {
		return ErrBadDimensions
	}
	if N != ldA || N != ldInv {
		return ErrNotSquare
	}
	if len(a) < N*ldA || len(aInv) < N*ldInv {
		return ErrBadDimensions
	}
	if len(ipiv) < N {
		return ErrBadDimensions
	}

	var i, j, k int
	var sum float32

	// Initialize inverse with identity matrix
	for i = 0; i < N; i++ {
		for j = 0; j < N; j++ {
			if i == j {
				setElem(aInv, ldInv, i, j, 1.0)
			} else {
				setElem(aInv, ldInv, i, j, 0.0)
			}
		}
	}

	// Apply row permutations from pivoting (P^T)
	for i = N - 1; i >= 0; i-- {
		if ipiv[i] != i {
			swapRows(aInv, ldInv, i, ipiv[i], N)
		}
	}

	// Solve L * Y = I for Y (forward substitution)
	// L is stored below diagonal in a
	for j = 0; j < N; j++ {
		for i = 0; i < N; i++ {
			sum = getElem(aInv, ldInv, i, j)
			for k = 0; k < i; k++ {
				sum -= getElem(a, ldA, i, k) * getElem(aInv, ldInv, k, j)
			}
			// L has unit diagonal
			setElem(aInv, ldInv, i, j, sum)
		}
	}

	// Solve U * X = Y for X (back substitution)
	// U is stored on/above diagonal in a
	for j = 0; j < N; j++ {
		for i = N - 1; i >= 0; i-- {
			sum = getElem(aInv, ldInv, i, j)
			for k = i + 1; k < N; k++ {
				sum -= getElem(a, ldA, i, k) * getElem(aInv, ldInv, k, j)
			}
			uii := getElem(a, ldA, i, i)
			if math32.Abs(uii) < 1e-6 {
				return ErrSingularMatrix
			}
			setElem(aInv, ldInv, i, j, sum/uii)
		}
	}

	return nil
}

// GEQRF: Compute QR decomposition A = Q * R
// On input: a contains M × N matrix
// On output: a contains R in upper triangular part and Householder vectors below diagonal
//
//	tau contains scalar factors for Householder vectors (length min(M,N))
func Geqrf(a []float32, tau []float32, ldA, M, N int) error {
	if M <= 0 || N <= 0 {
		return ErrBadDimensions
	}
	if len(a) < M*ldA {
		return ErrBadDimensions
	}
	minMN := imin(M, N)
	if len(tau) < minMN {
		return ErrBadDimensions
	}

	var k, i, j int
	var scale, sigma, sum, tauVal float32

	// Householder transformations
	for k = 0; k < minMN-1; k++ {
		// Find scale factor
		scale = 0.0
		for i = k; i < M; i++ {
			val := math32.Abs(getElem(a, ldA, i, k))
			scale = fmax(scale, val)
		}

		if scale == 0.0 {
			// Singular case
			tau[k] = 0.0
		} else {
			// Normalize column k
			for i = k; i < M; i++ {
				val := getElem(a, ldA, i, k) / scale
				setElem(a, ldA, i, k, val)
			}

			// Compute sigma
			sum = 0.0
			for i = k; i < M; i++ {
				val := getElem(a, ldA, i, k)
				sum += val * val
			}
			sigma = sign(math32.Sqrt(sum), getElem(a, ldA, k, k))
			setElem(a, ldA, k, k, getElem(a, ldA, k, k)+sigma)
			tau[k] = sigma * getElem(a, ldA, k, k)

			// Apply transformation to remaining columns
			for j = k + 1; j < N; j++ {
				sum = 0.0
				for i = k; i < M; i++ {
					sum += getElem(a, ldA, i, k) * getElem(a, ldA, i, j)
				}
				tauVal = sum / tau[k]
				for i = k; i < M; i++ {
					val := getElem(a, ldA, i, j) - tauVal*getElem(a, ldA, i, k)
					setElem(a, ldA, i, j, val)
				}
			}

			// Restore scale
			for i = k; i < M; i++ {
				val := getElem(a, ldA, i, k) * scale
				setElem(a, ldA, i, k, val)
			}
			setElem(a, ldA, k, k, -scale*sigma)
		}
	}

	// Last column for rectangular matrices
	if M >= N && k == N-1 {
		tau[k] = 0.0
	}

	return nil
}

// ORGQR: Generate orthogonal matrix Q from QR decomposition
// Input: a and tau from GEQRF
// Output: q contains M × M orthogonal matrix Q
func Orgqr(q, a []float32, tau []float32, ldA, ldQ, M, N, K int) error {
	if M <= 0 || N <= 0 || K <= 0 {
		return ErrBadDimensions
	}
	if len(a) < M*ldA {
		return ErrBadDimensions
	}
	if len(q) < M*ldQ {
		return ErrBadDimensions
	}
	minMN := imin(M, N)
	if len(tau) < minMN {
		return ErrBadDimensions
	}

	var i, j, k int
	var sum, tauVal float32

	// Initialize Q as identity matrix
	for i = 0; i < M; i++ {
		for j = 0; j < M; j++ {
			if i == j {
				setElem(q, ldQ, i, j, 1.0)
			} else {
				setElem(q, ldQ, i, j, 0.0)
			}
		}
	}

	// Apply Householder transformations from right to left
	K = imin(K, minMN)
	for k = K - 1; k >= 0; k-- {
		if tau[k] != 0.0 {
			// Apply transformation to columns k to M-1 of Q
			for j = k; j < M; j++ {
				sum = 0.0
				for i = k; i < M; i++ {
					sum += getElem(a, ldA, i, k) * getElem(q, ldQ, i, j)
				}
				tauVal = sum / tau[k]
				for i = k; i < M; i++ {
					val := getElem(q, ldQ, i, j) - tauVal*getElem(a, ldA, i, k)
					setElem(q, ldQ, i, j, val)
				}
			}
		}
	}

	return nil
}

// GEPSEU: Compute Moore-Penrose pseudo-inverse A^+
// A^+ = V * Σ^+ * U^T where Σ^+ is pseudo-inverse of diagonal matrix
// Input: a contains M × N matrix
// Output: aPinv contains N × M pseudo-inverse (row-major, ldApinv ≥ M)
// This implementation uses QR decomposition for overdetermined systems
// For full SVD-based implementation, use GESVD
func Gepseu(aPinv, a []float32, ldA, ldApinv, M, N int) error {
	if M <= 0 || N <= 0 {
		return ErrBadDimensions
	}
	if len(a) < M*ldA {
		return ErrBadDimensions
	}
	if len(aPinv) < N*ldApinv {
		return ErrBadDimensions
	}

	// Full implementation requires GESVD
	// Placeholder indicates this needs full SVD-based implementation
	return errors.New("GEPSEU: requires GESVD - full implementation pending")
}

// GESVD: Compute SVD decomposition A = U * Σ * V^T
// On input: a contains M × N matrix
// On output: u contains M × M left singular vectors (row-major, ldU ≥ M)
//
//	s contains min(M,N) singular values (sorted descending)
//	vt contains N × N right singular vectors transposed (row-major, ldVt ≥ N)
//
// NOTE: This is a placeholder implementation. Full SVD requires bidiagonalization
// and QR iteration as implemented in mat/svd.go. For production use, consider
// using the mat package implementation.
func Gesvd(u, s, vt, a []float32, ldA, ldU, ldVt, M, N int) error {
	// Full SVD implementation is complex and would require ~300+ lines
	// This is a placeholder that indicates the function needs full implementation
	// based on the algorithm in mat/svd.go
	return errors.New("GESVD: full implementation pending - use mat.SVD for now")
}

// GNNLS: Solve non-negative least squares min ||AX - B|| subject to X >= 0
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//
//	b contains M × 1 right-hand side vector
//
// Output: x contains N × 1 solution vector (non-negative)
// Returns: residual norm ||AX - B||
// NOTE: This is a placeholder implementation. Full NNLS requires Lawson-Hanson
// active set method as implemented in mat/nnls.go (~500 lines). For production use,
// consider using the mat package implementation.
func Gnnls(x, a, b []float32, ldA, M, N int) (rNorm float32, err error) {
	// Full NNLS implementation is complex and would require ~500+ lines
	// This is a placeholder that indicates the function needs full implementation
	// based on the algorithm in mat/nnls.go
	return 0, errors.New("GNNLS: full implementation pending - use mat.NNLS for now")
}
