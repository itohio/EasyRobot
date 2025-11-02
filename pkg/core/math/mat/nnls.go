// Package mat provides Non-Negative Least Squares (NNLS) and Least Distance Programming (LDP) implementations.
// Algorithm: Lawson-Hanson active set method
// Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'
//
// NNLS solves: min ||AX - B|| subject to X >= 0
// LDP solves: min ||X|| subject to G*X >= H

package mat

import (
	"errors"
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// NNLSResult holds the result of Non-Negative Least Squares.
type NNLSResult struct {
	X     vec.Vector // Solution vector (n length)
	W     vec.Vector // Dual solution vector (n length)
	RNorm float32    // Euclidean norm of residual vector
}

// NNLS solves non-negative least squares problem.
// Minimize ||AX - B|| subject to X >= 0
// A must be (m x n), B must be (m length), X will be (n length)
// Note: A and B are modified during computation (they contain Q*A and Q*B on output)
// Returns error code:
//   nil = success
//   ErrNNLSBadDimensions = bad dimensions (m<=0 or n<=0)
//   ErrNNLSMaxIterations = maximum number (3n) of iterations exceeded
func NNLS(A Matrix, B vec.Vector, dst *NNLSResult, rangeVal float32) error {
	if len(A) == 0 || len(A[0]) == 0 {
		return ErrNNLSBadDimensions
	}

	m := len(A)
	n := len(A[0])

	if len(B) != m {
		return ErrNNLSBadDimensions
	}

	if m <= 0 || n <= 0 {
		return ErrNNLSBadDimensions
	}

	// Allocate result vectors
	dst.X = make(vec.Vector, n)
	dst.W = make(vec.Vector, n)
	ZZ := make(vec.Vector, m)
	index := make([]int, n)

	const (
		zero   = 0.0
		two    = 2.0
		factor = 0.0001
	)

	// Initialize
	for i := 0; i < n; i++ {
		dst.X[i] = 0
		index[i] = i
	}

	iz2 := n - 1 // Set Z
	iz1 := 0
	nsetp := -1 // No Set P
	npp1 := 0

	itmax := 3 * n
	iter := 0

	// Main Loop
	for {
		if iz1 > iz2 || nsetp >= m-1 {
			goto terminate // Quit if all coefficients are already in solution
		}

		// Compute Components of the Dual (Negative Gradient) Vector W
		for iz := iz1; iz <= iz2; iz++ {
			j := index[iz]
			var sm float32 = zero
			for l := npp1; l < m; l++ {
				sm += A[l][j] * B[l]
			}
			dst.W[j] = sm
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
				if dst.W[jTest] > wmax {
					wmax = dst.W[jTest]
					izmax = iz
				}
			}

			if wmax <= 0 {
				goto terminate // Quit - Kuhn-Tucker Conditions Are Satisfied
			}

			iz := izmax
			j = index[iz]

			// The Sign of W[j] is OK for j to Be Moved to Set P.
			// Begin the Transformation and Check New Diagonal Element to Avoid
			// Near Linear Dependence

			asave := A[npp1][j]

			up, err := A.H1(j, npp1, npp1+1, rangeVal)
			if err != nil {
				return err
			}
			if up == 0 {
				dst.W[j] = 0
				found = false
				continue
			}

			var unorm float32 = zero
			for l := 0; l <= nsetp; l++ {
				unorm += A[l][j] * A[l][j]
			}
			unorm = math32.Sqrt(unorm)

			if (unorm + math32.Abs(A[npp1][j])*factor) > unorm {
				// Col j is Sufficiently Independent
				// Copy B into ZZ, update ZZ
				// Solve for Ztest ( = Proposed New Value for X[j] )

				for l := 0; l < m; l++ {
					ZZ[l] = B[l]
				}
				if err := A.H2(j, npp1, npp1+1, up, ZZ, rangeVal); err != nil {
					return err
				}

				// See if ztest is Positive
				// Reject j as a Candidate to be Moved from Set Z to Set P
				// Restore A(npp1,j), Set W(j)=0., and Loop Back to Test Dual

				if A[npp1][j] == 0 {
					A[npp1][j] = asave
					dst.W[j] = 0
					continue
				}

				ztest := ZZ[npp1] / A[npp1][j]
				if ztest > 0 {
					found = true
					break
				}
			}

			if found {
				break
			}

			// Coeffs Again
			A[npp1][j] = asave
			dst.W[j] = 0
		}

		// The Index j=index(iz) has been Selected to be Moved from
		// Set Z to Set P. Update B, Update Indices, Apply Householder
		// Transformation to Cols in New Set Z, Zero Subdiagonal ELTS
		// in Col j, Set W(j)=0.

		for l := 0; l < m; l++ {
			B[l] = ZZ[l]
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
				if err := A.H3(jSelected, nsetp, npp1, upSelected, jj, rangeVal); err != nil {
					return err
				}
			}
		}

		if nsetp != m-1 {
			for l := npp1; l < m; l++ {
				A[l][jSelected] = zero
			}
		}
		dst.W[jSelected] = 0

		// Solve Triangular System
		// Store Temporal Solution in ZZ

		for l := 0; l <= nsetp; l++ {
			ip := nsetp - l
			if l != 0 {
				jj := index[ip+1]
				for ii := 0; ii <= ip; ii++ {
					ZZ[ii] -= A[ii][jj] * ZZ[ip+1]
				}
			}
			jj := index[ip]
			if A[ip][jj] == 0 {
				return ErrNNLSSingular
			}
			ZZ[ip] /= A[ip][jj]
		}

		// Secondary Loop
		for {
			iter++ // Iteration Counter
			if iter > itmax {
				return ErrNNLSMaxIterations
			}

			// See if all new Constrained Coeffs are Feasible
			var alpha float32 = two
			var jj int
			for ip := 0; ip <= nsetp; ip++ {
				l := index[ip]
				if ZZ[ip] <= 0 {
					var t float32 = -dst.X[l] / (ZZ[ip] - dst.X[l])
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
				dst.X[l] += alpha * (ZZ[ip] - dst.X[l])
			}

			// Modify A and B and the Index Arrays to Move Coefficient i
			// from Set P to Set Z

			i := index[jj]

			for {
				dst.X[i] = zero

				if jj != nsetp {
					jj++
					for jCol := jj; jCol <= nsetp; jCol++ {
						ii := index[jCol]
						index[jCol-1] = ii
						cc, ss, sig := G1(A[jCol-1][ii], A[jCol][ii])
						A[jCol-1][ii] = sig
						A[jCol][ii] = zero
						for l := 0; l < n; l++ {
							if l != ii {
								var x, y float32
								x = A[jCol-1][l]
								y = A[jCol][l]
								G2(cc, ss, &x, &y)
								A[jCol-1][l] = x
								A[jCol][l] = y
							}
						}
						var x, y float32
						x = B[jCol-1]
						y = B[jCol]
						G2(cc, ss, &x, &y)
						B[jCol-1] = x
						B[jCol] = y
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

				var found bool
				for jj := 0; jj <= nsetp; jj++ {
					i = index[jj]
					if dst.X[i] <= zero {
						found = true
						break
					}
				}
				if !found {
					break
				}
			}

			// Copy B into ZZ. Then Solve again and Loop Back.

			for i := 0; i < m; i++ {
				ZZ[i] = B[i]
			}

			for l := 0; l <= nsetp; l++ {
				ip := nsetp - l
				if l != 0 {
					jj := index[ip+1]
					for ii := 0; ii <= ip; ii++ {
						ZZ[ii] -= A[ii][jj] * ZZ[ip+1]
					}
				}
				jj := index[ip]
				if A[ip][jj] == 0 {
					return ErrNNLSSingular
				}
				ZZ[ip] /= A[ip][jj]
			}
		}

		// End of Secondary Loop

		for ip := 0; ip <= nsetp; ip++ {
			i := index[ip]
			dst.X[i] = ZZ[ip]
		}
	} // All New Coeffs are Positive. Loop Back to Beginning

	// End of Main Loop

terminate:
	// Compute the norm of the Final Residual Vector
	var sm float32 = zero

	if npp1 >= m {
		for j := 0; j < n; j++ {
			dst.W[j] = zero
		}
	} else {
		for i := npp1; i < m; i++ {
			sm += B[i] * B[i]
		}
	}

	dst.RNorm = math32.Sqrt(sm)
	return nil
}

// Error definitions for NNLS
var (
	ErrNNLSBadDimensions = errors.New("nnls: bad dimensions (m<=0 or n<=0)")
	ErrNNLSMaxIterations = errors.New("nnls: maximum number (3n) of iterations exceeded")
	ErrNNLSSingular      = errors.New("nnls: singular matrix encountered")
)

// LDPResult holds the result of Least Distance Programming.
type LDPResult struct {
	X     vec.Vector // Solution vector (n length)
	XNorm float32    // Euclidean norm of solution vector
}

// LDP solves least distance programming problem.
// Minimize ||X|| subject to G*X >= H
// G must be (m x n), H must be (m length), X will be (n length)
// Returns error code:
//   nil = success
//   ErrLDPBadDimensions = bad dimensions
//   ErrLDPMaxIterations = maximum iterations (NNLS)
//   ErrLDPIncompatible = constraints incompatible
func LDP(G Matrix, H vec.Vector, dst *LDPResult, rangeVal float32) error {
	if len(G) == 0 || len(G[0]) == 0 {
		return ErrLDPBadDimensions
	}

	m := len(G)
	n := len(G[0])

	if len(H) != m {
		return ErrLDPBadDimensions
	}

	if m == 0 {
		// No Constraints - Successful Return
		dst.X = make(vec.Vector, n)
		dst.XNorm = 0
		return nil
	}

	// Construct E Matrix: E = [G^T; H^T]
	// E has size (n+1) x m
	GT := New(n+1, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			GT[j][i] = G[i][j]
		}
		GT[n][i] = H[i]
	}

	// Construct Vector F = {0,0,...,0,1}
	F := make(vec.Vector, n+1)
	for i := 0; i < n; i++ {
		F[i] = 0
	}
	F[n] = 1

	// Call Nonnegative Least Squares
	var nnlsResult NNLSResult
	if err := NNLS(GT, F, &nnlsResult, rangeVal); err != nil {
		if err == ErrNNLSMaxIterations {
			return ErrLDPMaxIterations
		}
		if err == ErrNNLSBadDimensions {
			return ErrLDPBadDimensions
		}
		return err
	}

	if nnlsResult.RNorm <= 0 {
		return ErrLDPIncompatible
	}

	// Extract solution from NNLS result
	Y := nnlsResult.X

	var fac float32 = 1.0
	for i := 0; i < m; i++ {
		fac -= H[i] * Y[i]
	}

	// Check if constraints are compatible
	// Use a small epsilon for floating point comparison
	const epsilon = 1e-10
	if fac <= epsilon {
		return ErrLDPIncompatible
	}

	fac = 1 / fac

	// Compute X = fac * G^T * Y
	dst.X = make(vec.Vector, n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			dst.X[j] += G[i][j] * Y[i]
		}
		dst.X[j] *= fac
	}

	// Compute norm of X (excluding first element if n>1)
	dst.XNorm = 0
	for j := 1; j < n; j++ {
		dst.XNorm += dst.X[j] * dst.X[j]
	}
	if n > 1 {
		dst.XNorm = math32.Sqrt(dst.XNorm)
	} else if n == 1 {
		dst.XNorm = math32.Abs(dst.X[0])
	}

	return nil
}

// Error definitions for LDP
var (
	ErrLDPBadDimensions = errors.New("ldp: bad dimensions")
	ErrLDPMaxIterations = errors.New("ldp: maximum iterations (NNLS) exceeded")
	ErrLDPIncompatible  = errors.New("ldp: constraints incompatible")
)

