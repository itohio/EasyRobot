package fp32

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

// pytag computes sqrt(a²+b²) without overflow.
// Used in SVD and other numerical algorithms to avoid numerical overflow.
func pytag(a, b float32) float32 {
	absa := math32.Abs(a)
	absb := math32.Abs(b)
	if absa > absb {
		return absa * math32.Sqrt(1.0+(absb/absa)*(absb/absa))
	}
	if absb == 0.0 {
		return 0.0
	}
	return absb * math32.Sqrt(1.0+(absa/absb)*(absa/absb))
}

// G1 computes Givens rotation matrix.
// Computes [cs  sn] such that [cs  sn] [a] -> [sig]
//
//	[-sn cs]           [-sn cs] [b]    [0 ]
//
// Returns cosine (cs), sine (sn), and sigma (sig = sqrt(a²+b²))
func G1(a, b float32) (cs, sn, sig float32) {
	var xr, yr float32

	if math32.Abs(a) > math32.Abs(b) {
		xr = b / a
		yr = math32.Sqrt(1 + xr*xr)
		cs = sign(1/yr, a)
		sn = cs * xr
		sig = math32.Abs(a) * yr
	} else {
		if b == 0 {
			sig = 0
			cs = 0
			sn = 1
		} else {
			xr = a / b
			yr = math32.Sqrt(1 + xr*xr)
			sn = sign(1/yr, b)
			cs = sn * xr
			sig = math32.Abs(b) * yr
		}
	}
	return cs, sn, sig
}

// G2 applies Givens rotation to (x, y).
// Applies the rotation computed by G1 to the pair (x, y).
func G2(cs, sn float32, x, y *float32) {
	xr := cs*(*x) + sn*(*y)
	*y = -sn*(*x) + cs*(*y)
	*x = xr
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
			// Apply Householder transformation: H = I - tau * v * v^T
			// Standard LAPACK formula: x_new = x - v * (v^T * x) * (2 / (v^T * v))
			// Compute v^T * v (norm squared of Householder vector)
			vNormSq := float32(0.0)
			for i = k; i < M; i++ {
				val := getElem(a, ldA, i, k)
				vNormSq += val * val
			}

			if vNormSq > 0.0 {
				// Apply transformation to columns k to M-1 of Q
				tauNorm := 2.0 / vNormSq
				for j = k; j < M; j++ {
					sum = 0.0
					for i = k; i < M; i++ {
						sum += getElem(a, ldA, i, k) * getElem(q, ldQ, i, j)
					}
					tauVal = sum * tauNorm
					for i = k; i < M; i++ {
						val := getElem(q, ldQ, i, j) - tauVal*getElem(a, ldA, i, k)
						setElem(q, ldQ, i, j, val)
					}
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
// Algorithm: Uses GESVD for full SVD-based implementation
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

	minMN := imin(M, N)

	// Allocate working matrices for SVD
	u := make([]float32, M*M)
	s := make([]float32, N) // Singular values vector - allocate N even though only minMN are used
	vt := make([]float32, N*N)
	ldU := M
	ldVt := N

	// Compute SVD: A = U * Σ * V^T
	// Note: Gesvd returns V (not V^T) in vt
	if err := Gesvd(u, s, vt, a, ldA, ldU, ldVt, M, N); err != nil {
		return err
	}

	// Transpose V to get V^T (pseudo-inverse expects V^T, consistent with svd.go)
	vTemp := make([]float32, N*N)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			vTemp[i*N+j] = getElem(vt, ldVt, j, i) // V^T[i][j] = V[j][i]
		}
	}
	vt = vTemp
	// ldVt stays N since it's still N×N

	// Tolerance for singular values (treat values below this as zero)
	const tol = float32(1e-10)

	// Compute pseudo-inverse: A^+ = V * Σ^+ * U^T
	// where Σ^+[i] = 1/s[i] if |s[i]| > tol, else 0
	// A^+[i,j] = sum_k (V[i,k] * Σ^+[k] * U[j,k])
	// Since vt now contains V^T, we use vt[k][i] = V^T[k][i] = V[i][k]
	for i := 0; i < N; i++ {
		for j := 0; j < M; j++ {
			var sum float32 = 0.0
			for k := 0; k < minMN; k++ {
				if math32.Abs(s[k]) > tol {
					// A^+[i,j] = V[i,k] * (1/s[k]) * U[j,k]
					// vt[k][i] = V^T[k][i] = V[i][k]
					vik := getElem(vt, ldVt, k, i) // V^T[k][i] = V[i][k]
					ujk := getElem(u, ldU, j, k)   // U[j][k]
					sum += vik * (1.0 / s[k]) * ujk
				}
			}
			setElem(aPinv, ldApinv, i, j, sum)
		}
	}

	return nil
}

// GESVD: Compute SVD decomposition A = U * Σ * V^T
// On input: a contains M × N matrix
// On output: u contains M × M left singular vectors (row-major, ldU ≥ M)
//
//	s contains min(M,N) singular values (sorted descending)
//	vt contains N × N right singular vectors transposed (row-major, ldVt ≥ N)
//
// Algorithm: Golub-Reinsch (Householder bidiagonalization + QR iteration)
// Reference: Numerical Recipes in C, W. H. Press et al.
// Gesvd computes Singular Value Decomposition using Golub-Reinsch algorithm.
// A = U * S * V^T where:
//   - A is M×N (input matrix, row-major, ldA >= N)
//   - U is M×M (left singular vectors, row-major, ldU >= M)
//   - S is N (singular values, only first min(M,N) are meaningful)
//   - V^T is N×N (right singular vectors transposed, row-major, ldVt >= N)
//
// REQUIREMENT: M >= N (rows >= cols). The algorithm does not support M < N.
// For M < N cases, transpose the input matrix first.
func Gesvd(u, s, vt, a []float32, ldA, ldU, ldVt, M, N int) error {
	if M <= 0 || N <= 0 {
		return ErrBadDimensions
	}
	// Golub-Reinsch algorithm requires M >= N
	if M < N {
		return ErrBadDimensions // M < N not supported
	}
	if len(a) < M*ldA {
		return ErrBadDimensions
	}
	if len(u) < M*ldU {
		return ErrBadDimensions
	}
	if len(s) < N {
		return ErrBadDimensions
	}
	if len(vt) < N*ldVt {
		return ErrBadDimensions
	}

	// Create working copy of matrix A (since we modify it)
	// Work matrix will be used for bidiagonalization - needs M rows and N columns
	work := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			setElem(work, N, i, j, getElem(a, ldA, i, j))
		}
	}

	// Working vector for bidiagonalization
	rv1 := make([]float32, N)

	// Initialize output matrices
	// Vt will accumulate transformations
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			setElem(vt, ldVt, i, j, 0.0)
		}
	}

	var flag bool
	var i, its, j, jj, k, l, nm int
	var anorm, c, f, g, h, sVal, scale, x, y, z float32

	// Householder reduction to bidiagonal form
	g = 0.0
	scale = 0.0
	anorm = 0.0
	for i = 0; i < N; i++ {
		l = i + 1
		rv1[i] = scale * g
		g = 0.0
		sVal = 0.0
		scale = 0.0

		if i < M {
			for k = i; k < M; k++ {
				scale += math32.Abs(getElem(work, N, k, i))
			}
			if scale != 0.0 {
				for k = i; k < M; k++ {
					val := getElem(work, N, k, i) / scale
					setElem(work, N, k, i, val)
					sVal += val * val
				}
				f = getElem(work, N, i, i)
				g = -sign(math32.Sqrt(sVal), f)
				h = f*g - sVal
				setElem(work, N, i, i, f-g)
				for j = l; j < N; j++ {
					sVal = 0.0
					for k = i; k < M; k++ {
						sVal += getElem(work, N, k, i) * getElem(work, N, k, j)
					}
					f = sVal / h
					for k = i; k < M; k++ {
						val := getElem(work, N, k, j) + f*getElem(work, N, k, i)
						setElem(work, N, k, j, val)
					}
				}
				for k = i; k < M; k++ {
					val := getElem(work, N, k, i) * scale
					setElem(work, N, k, i, val)
				}
			}
		}
		s[i] = scale * g
		g = 0.0
		sVal = 0.0
		scale = 0.0

		if i < M && i != (N-1) {
			for k = l; k < N; k++ {
				scale += math32.Abs(getElem(work, N, i, k))
			}
			if scale != 0.0 {
				for k = l; k < N; k++ {
					val := getElem(work, N, i, k) / scale
					setElem(work, N, i, k, val)
					sVal += val * val
				}
				f = getElem(work, N, i, l)
				g = -sign(math32.Sqrt(sVal), f)
				h = f*g - sVal
				setElem(work, N, i, l, f-g)
				for k = l; k < N; k++ {
					rv1[k] = getElem(work, N, i, k) / h
				}
				for j = l; j < M; j++ {
					sVal = 0.0
					for k = l; k < N; k++ {
						sVal += getElem(work, N, j, k) * getElem(work, N, i, k)
					}
					for k = l; k < N; k++ {
						val := getElem(work, N, j, k) + sVal*rv1[k]
						setElem(work, N, j, k, val)
					}
				}
				for k = l; k < N; k++ {
					val := getElem(work, N, i, k) * scale
					setElem(work, N, i, k, val)
				}
			}
		}
		anorm = fmax(anorm, math32.Abs(s[i])+math32.Abs(rv1[i]))
	}

	// Accumulation of right-hand transformation
	for i = N - 1; i >= 0; i-- {
		if i < (N - 1) {
			if g != 0.0 {
				for j = l; j < N; j++ {
					val := (getElem(work, N, i, j) / getElem(work, N, i, l)) / g
					setElem(vt, ldVt, j, i, val)
				}
				for j = l; j < N; j++ {
					sVal = 0.0
					for k = l; k < N; k++ {
						sVal += getElem(work, N, i, k) * getElem(vt, ldVt, k, j)
					}
					for k = l; k < N; k++ {
						val := getElem(vt, ldVt, k, j) + sVal*getElem(vt, ldVt, k, i)
						setElem(vt, ldVt, k, j, val)
					}
				}
			}
			for j = l; j < N; j++ {
				setElem(vt, ldVt, i, j, 0.0)
				setElem(vt, ldVt, j, i, 0.0)
			}
		}
		setElem(vt, ldVt, i, i, 1.0)
		g = rv1[i]
		l = i
	}

	// Accumulation of left-hand transformations
	// Apply transformations to U matrix - work matrix accumulates U
	minMN := imin(M, N)
	for i = minMN - 1; i >= 0; i-- {
		l = i + 1
		g = s[i]
		for j = l; j < N; j++ {
			setElem(work, N, i, j, 0.0)
		}
		if g != 0.0 {
			g = 1.0 / g
			for j = l; j < N; j++ {
				sVal = 0.0
				for k = l; k < M; k++ {
					sVal += getElem(work, N, k, i) * getElem(work, N, k, j)
				}
				f = (sVal / getElem(work, N, i, i)) * g
				for k = i; k < M; k++ {
					val := getElem(work, N, k, j) + f*getElem(work, N, k, i)
					setElem(work, N, k, j, val)
				}
			}
			for j = i; j < M; j++ {
				val := getElem(work, N, j, i) * g
				setElem(work, N, j, i, val)
			}
		} else {
			for j = i; j < M; j++ {
				setElem(work, N, j, i, 0.0)
			}
		}
		val := getElem(work, N, i, i) + 1.0
		setElem(work, N, i, i, val)
	}

	// Diagonalization of the bidiagonal form: Loop over singular values
	maxIterations := 30
	for k = N - 1; k >= 0; k-- {
		for its = 1; its <= maxIterations; its++ {
			flag = true
			for l = k; l >= 0; l-- {
				nm = l - 1
				if float32(math32.Abs(rv1[l])+anorm) == anorm {
					flag = false
					break
				}
				if nm >= 0 && float32(math32.Abs(s[nm])+anorm) == anorm {
					break
				}
			}
			if flag {
				c = 0.0
				sVal = 1.0
				for i = l; i <= k; i++ {
					f = sVal * rv1[i]
					rv1[i] = c * rv1[i]
					if float32(math32.Abs(f)+anorm) == anorm {
						break
					}
					g = s[i]
					h = pytag(f, g)
					s[i] = h
					h = 1.0 / h
					c = g * h
					sVal = -f * h
					for j = 0; j < M; j++ {
						y = getElem(work, N, j, nm)
						z = getElem(work, N, j, i)
						setElem(work, N, j, nm, y*c+z*sVal)
						setElem(work, N, j, i, z*c-y*sVal)
					}
				}
			}
			z = s[k]
			if l == k {
				// Convergence
				if z < 0.0 {
					// Singular value is made nonnegative
					s[k] = -z
					for j = 0; j < N; j++ {
						val := getElem(vt, ldVt, j, k)
						setElem(vt, ldVt, j, k, -val)
					}
				}
				break
			}
			if its == maxIterations {
				return ErrMaxIterations
			}
			x = s[l]
			nm = k - 1
			y = s[nm]
			g = rv1[nm]
			h = rv1[k]
			f = ((y-z)*(y+z) + (g-h)*(g+h)) / (2.0 * h * y)
			g = pytag(f, 1.0)
			f = ((x-z)*(x+z) + h*((y/(f+sign(g, f)))-h)) / x
			c = 1.0
			sVal = 1.0
			// Next QR transformation
			for j = l; j < nm+1; j++ {
				i = j + 1
				g = rv1[i]
				y = s[i]
				h = sVal * g
				g = c * g
				z = pytag(f, h)
				rv1[j] = z
				c = f / z
				sVal = h / z
				f = x*c + g*sVal
				g = g*c - x*sVal
				h = y * sVal
				y *= c
				for jj = 0; jj < N; jj++ {
					x = getElem(vt, ldVt, jj, j)
					z = getElem(vt, ldVt, jj, i)
					setElem(vt, ldVt, jj, j, x*c+z*sVal)
					setElem(vt, ldVt, jj, i, z*c-x*sVal)
				}
				z = pytag(f, h)
				s[j] = z
				// Rotation can be arbitrary if z = 0
				if z != 0.0 {
					z = 1.0 / z
					c = f * z
					sVal = h * z
				}
				f = c*g + sVal*y
				x = c*y - sVal*g
				for jj = 0; jj < M; jj++ {
					y = getElem(work, N, jj, j)
					z = getElem(work, N, jj, i)
					setElem(work, N, jj, j, y*c+z*sVal)
					setElem(work, N, jj, i, z*c-y*sVal)
				}
			}
			rv1[l] = 0.0
			rv1[k] = f
			s[k] = x
		}
	}

	// Copy accumulated U from work to output u (work is M x N, but U is M x M)
	for i = 0; i < M; i++ {
		for j = 0; j < N && j < M; j++ {
			setElem(u, ldU, i, j, getElem(work, N, i, j))
		}
		// For M > N, compute remaining columns to complete orthonormal basis
		// Use Gram-Schmidt on standard basis vectors
		for j = N; j < M; j++ {
			// Start with standard basis vector e_j = [0,0,...,1,...,0] at position j
			for k := 0; k < M; k++ {
				if k == j {
					setElem(u, ldU, k, j, 1.0)
				} else {
					setElem(u, ldU, k, j, 0.0)
				}
			}
			// Subtract projections onto previous columns
			for k := 0; k < j; k++ {
				var dot float32
				for l := 0; l < M; l++ {
					dot += getElem(u, ldU, l, j) * getElem(u, ldU, l, k)
				}
				for l = 0; l < M; l++ {
					val := getElem(u, ldU, l, j) - dot*getElem(u, ldU, l, k)
					setElem(u, ldU, l, j, val)
				}
			}
			// Normalize
			var norm float32
			for k := 0; k < M; k++ {
				val := getElem(u, ldU, k, j)
				norm += val * val
			}
			norm = math32.Sqrt(norm)
			if norm > 0.0 {
				for k := 0; k < M; k++ {
					val := getElem(u, ldU, k, j) / norm
					setElem(u, ldU, k, j, val)
				}
			}
		}
	}

	return nil
}

// GNNLS: Solve non-negative least squares min ||AX - B|| subject to X >= 0
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//
//	b contains M × 1 right-hand side vector (modified in-place)
//
// Output: x contains N × 1 solution vector (non-negative)
// Returns: residual norm ||AX - B||
// Algorithm: Lawson-Hanson active set method with Householder QR decomposition
// Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'
func Gnnls(x, a, b []float32, ldA, M, N int) (rNorm float32, err error) {
	if M <= 0 || N <= 0 {
		return 0, ErrBadDimensions
	}
	if len(a) < M*ldA {
		return 0, ErrBadDimensions
	}
	if len(b) < M {
		return 0, ErrBadDimensions
	}
	if len(x) < N {
		return 0, ErrBadDimensions
	}

	const (
		zero   = float32(0.0)
		two    = float32(2.0)
		factor = float32(0.0001)
	)

	// Working vectors
	zz := make([]float32, M)
	w := make([]float32, N)
	index := make([]int, N)

	// Initialize solution vector
	for i := 0; i < N; i++ {
		x[i] = zero
		index[i] = i
	}

	// Set Z indices
	iz2 := N - 1
	iz1 := 0
	nsetp := -1
	npp1 := 0

	itmax := 3 * N
	iter := 0

	// Main Loop
	for {
		if iz1 > iz2 || nsetp >= M-1 {
			goto terminate // Quit if all coefficients are already in solution
		}

		// Compute Components of the Dual (Negative Gradient) Vector W
		for iz := iz1; iz <= iz2; iz++ {
			j := index[iz]
			var sm float32 = zero
			for l := npp1; l < M; l++ {
				sm += getElem(a, ldA, l, j) * b[l]
			}
			w[j] = sm
		}

		// Find Largest Positive W[j]
		var wmax float32
		var izmax int
		var j int
		var up float32
		var found bool

		for {
			wmax = zero
			izmax = -1
			for iz := iz1; iz <= iz2; iz++ {
				jTest := index[iz]
				if w[jTest] > wmax {
					wmax = w[jTest]
					izmax = iz
				}
			}

			if wmax <= zero {
				goto terminate // Quit - Kuhn-Tucker Conditions Are Satisfied
			}

			iz := izmax
			j = index[iz]

			// The Sign of W[j] is OK for j to Be Moved to Set P.
			// Begin the Transformation and Check New Diagonal Element to Avoid
			// Near Linear Dependence

			asave := getElem(a, ldA, npp1, j)

			up, err = H1(a, j, npp1, npp1+1, ldA, DefaultRange)
			if err != nil {
				return 0, err
			}
			if up == zero {
				w[j] = zero
				found = false
				continue
			}

			var unorm float32 = zero
			for l := 0; l <= nsetp; l++ {
				val := getElem(a, ldA, l, j)
				unorm += val * val
			}
			unorm = math32.Sqrt(unorm)

			if (unorm + math32.Abs(getElem(a, ldA, npp1, j))*factor) > unorm {
				// Col j is Sufficiently Independent
				// Copy B into ZZ, update ZZ
				// Solve for Ztest ( = Proposed New Value for X[j] )

				for l := 0; l < M; l++ {
					zz[l] = b[l]
				}
				if err := H2(a, zz, j, npp1, npp1+1, up, ldA, DefaultRange); err != nil {
					return 0, err
				}

				// See if ztest is Positive
				// Reject j as a Candidate to be Moved from Set Z to Set P
				// Restore A(npp1,j), Set W(j)=0., and Loop Back to Test Dual

				if getElem(a, ldA, npp1, j) == zero {
					setElem(a, ldA, npp1, j, asave)
					w[j] = zero
					continue
				}

				ztest := zz[npp1] / getElem(a, ldA, npp1, j)
				if ztest > zero {
					found = true
					break
				}
			}

			if found {
				break
			}

			// Coeffs Again
			setElem(a, ldA, npp1, j, asave)
			w[j] = zero
		}

		// The Index j=index(iz) has been Selected to be Moved from
		// Set Z to Set P. Update B, Update Indices, Apply Householder
		// Transformation to Cols in New Set Z, Zero Subdiagonal ELTS
		// in Col j, Set W(j)=0.

		for l := 0; l < M; l++ {
			b[l] = zz[l]
		}

		index[izmax] = index[iz1]
		index[iz1] = j
		jSelected := j
		upSelected := up
		iz1++
		nsetp = npp1
		npp1++

		if iz1 <= iz2 {
			for jz := iz1; jz <= iz2; jz++ {
				jj := index[jz]
				if err := H3(a, jSelected, nsetp, npp1, upSelected, jj, ldA, DefaultRange); err != nil {
					return 0, err
				}
			}
		}

		if nsetp != M-1 {
			for l := npp1; l < M; l++ {
				setElem(a, ldA, l, jSelected, zero)
			}
		}
		w[jSelected] = zero

		// Solve Triangular System
		// Store Temporal Solution in ZZ

		for l := 0; l <= nsetp; l++ {
			ip := nsetp - l
			if l != 0 {
				jj := index[ip+1]
				for ii := 0; ii <= ip; ii++ {
					zz[ii] -= getElem(a, ldA, ii, jj) * zz[ip+1]
				}
			}
			jj := index[ip]
			ajj := getElem(a, ldA, ip, jj)
			if ajj == zero {
				return 0, ErrSingularMatrix
			}
			zz[ip] /= ajj
		}

		// Secondary Loop
		for {
			iter++ // Iteration Counter
			if iter > itmax {
				return 0, ErrMaxIterations
			}

			// See if all new Constrained Coeffs are Feasible
			var alpha float32 = two
			var jj int
			for ip := 0; ip <= nsetp; ip++ {
				l := index[ip]
				if zz[ip] <= zero {
					var t float32 = -x[l] / (zz[ip] - x[l])
					if alpha > t {
						alpha = t
						jj = ip
					}
				}
			}

			// If all new constrained Coeffs are Feasible then alpha will
			// still = 2. If so exit from Secondary Loop

			if alpha == two {
				break // Exit from Secondary Loop
			}

			// Otherwise use alpha which will be between 0. and 1. To
			// Interpolate between the old X and new ZZ.

			for ip := 0; ip <= nsetp; ip++ {
				l := index[ip]
				x[l] += alpha * (zz[ip] - x[l])
			}

			// Modify A and B and the Index Arrays to Move Coefficient i
			// from Set P to Set Z

			i := index[jj]

			for {
				x[i] = zero

				if jj != nsetp {
					jj++
					for jCol := jj; jCol <= nsetp; jCol++ {
						ii := index[jCol]
						index[jCol-1] = ii
						cc, ss, sig := G1(getElem(a, ldA, jCol-1, ii), getElem(a, ldA, jCol, ii))
						setElem(a, ldA, jCol-1, ii, sig)
						setElem(a, ldA, jCol, ii, zero)
						for l := 0; l < N; l++ {
							if l != ii {
								var xVal, yVal float32
								xVal = getElem(a, ldA, jCol-1, l)
								yVal = getElem(a, ldA, jCol, l)
								G2(cc, ss, &xVal, &yVal)
								setElem(a, ldA, jCol-1, l, xVal)
								setElem(a, ldA, jCol, l, yVal)
							}
						}
						var xVal, yVal float32
						xVal = b[jCol-1]
						yVal = b[jCol]
						G2(cc, ss, &xVal, &yVal)
						b[jCol-1] = xVal
						b[jCol] = yVal
					}
				}

				npp1 = nsetp
				nsetp--
				iz1--
				index[iz1] = i

				// See if the Remaining Coeffs in Set P are Feasible. They should
				// be because of the Way alpha was Determined.
				// If any are infeasible it is Due to Round-off Error. Any
				// that are Nonpositive will be set to zero
				// and Moved from Set P to Set Z.

				found = false
				for jj := 0; jj <= nsetp; jj++ {
					i = index[jj]
					if x[i] <= zero {
						found = true
						break
					}
				}
				if !found {
					break
				}
			}

			// Copy B into ZZ. Then Solve again and Loop Back.

			for i := 0; i < M; i++ {
				zz[i] = b[i]
			}

			for l := 0; l <= nsetp; l++ {
				ip := nsetp - l
				if l != 0 {
					jj := index[ip+1]
					for ii := 0; ii <= ip; ii++ {
						zz[ii] -= getElem(a, ldA, ii, jj) * zz[ip+1]
					}
				}
				jj := index[ip]
				ajj := getElem(a, ldA, ip, jj)
				if ajj == zero {
					return 0, ErrSingularMatrix
				}
				zz[ip] /= ajj
			}
		}

		// End of Secondary Loop

		for ip := 0; ip <= nsetp; ip++ {
			i := index[ip]
			x[i] = zz[ip]
		}
	} // All New Coeffs are Positive. Loop Back to Beginning

	// End of Main Loop

terminate:
	// Compute the norm of the Final Residual Vector
	var sm float32 = zero

	if npp1 >= M {
		for j := 0; j < N; j++ {
			w[j] = zero
		}
	} else {
		for i := npp1; i < M; i++ {
			sm += b[i] * b[i]
		}
	}

	rNorm = math32.Sqrt(sm)
	return rNorm, nil
}
