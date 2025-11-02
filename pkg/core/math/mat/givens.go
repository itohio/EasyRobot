// Package mat provides Givens rotation functions.
// Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'

package mat

import (
	"github.com/chewxy/math32"
)

// G1 computes Givens rotation matrix.
// Computes [cs  sn] such that [cs  sn] [a] -> [sig]
//          [-sn cs]           [-sn cs] [b]    [0 ]
// Returns cosine (cs), sine (sn), and sigma (sig = sqrt(a²+b²))
func G1(a, b float32) (cs, sn, sig float32) {
	var xr, yr float32

	if math32.Abs(a) > math32.Abs(b) {
		xr = b / a
		yr = math32.Sqrt(1 + xr*xr)
		cs = SIGN(1/yr, a)
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
			sn = SIGN(1/yr, b)
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

