package gps

import (
	mat "github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Trilateration performs trilateration using three known points and their distances to an unknown point.
// P1, P2, P3 are ECEF coordinates of the three known points.
// DistA, DistB, DistC are distances from the unknown point to P1, P2, P3 respectively.
// Returns the ECEF coordinates of the unknown point.
func Trilateration(P1, P2, P3 ECEF, DistA, DistB, DistC float32) ECEF {
	// Use stack arrays for temporary calculations (no allocations)
	var P2P1, P3P1, ex, ey, iex, P3P1MinusIex vec.Vector3D

	// Transform to get circle 1 at origin
	// Transform to get circle 2 on x axis
	P2P1 = vec.Vector3D(P2)
	P2P1 = P2P1.Sub(vec.Vector3D(P1)).(vec.Vector3D)

	// Normalize P2P1 to get ex
	normP2P1 := P2P1.Magnitude()
	if normP2P1 == 0 {
		return ECEF{} // Invalid: P1 and P2 are the same point
	}

	ex = P2P1.Normal().(vec.Vector3D)

	// Calculate i (dot product of ex with P3-P1)
	P3P1 = vec.Vector3D(P3)
	P3P1 = P3P1.Sub(vec.Vector3D(P1)).(vec.Vector3D)

	i := ex.Dot(P3P1)

	// Calculate ey
	iex = ex.MulC(i).(vec.Vector3D)

	P3P1MinusIex = P3P1.Sub(iex).(vec.Vector3D)

	normP3P1MinusIex := P3P1MinusIex.Magnitude()
	if normP3P1MinusIex == 0 {
		return ECEF{} // Invalid: P3 is collinear with P1 and P2
	}

	ey = P3P1MinusIex.Normal().(vec.Vector3D)

	d := normP2P1
	j := ey.Dot(P3P1)

	// Calculate x and y in the transformed coordinate system
	x := (DistA*DistA - DistB*DistB + d*d) / (2.0 * d)
	y := (DistA*DistA-DistC*DistC+i*i+j*j)/(2.0*j) - (i * x / j)

	// Transform back to ECEF coordinates using vector operations
	var result vec.Vector3D
	result = vec.Vector3D(P1)
	result = result.Add(ex.MulC(x).(vec.Vector3D)).(vec.Vector3D)
	result = result.Add(ey.MulC(y).(vec.Vector3D)).(vec.Vector3D)

	return ECEF(result)
}

// Multilateration performs multilateration using multiple known points and their distances.
// points is a slice of (Coord, ECEF) pairs where Coord is the geographic coordinate
// and ECEF is the corresponding ECEF coordinate.
// distances is a slice of distances from the unknown point to each known point.
// Returns the ECEF coordinates of the unknown point.
// Requires at least 3 points.
func Multilateration(points []struct {
	Coord Coord
	ECEF  ECEF
}, distances []float32) (ECEF, error) {
	if len(points) < 3 {
		return ECEF{}, ErrInsufficientPoints
	}
	if len(points) != len(distances) {
		return ECEF{}, ErrMismatchedLengths
	}

	// For 3 points, use trilateration
	if len(points) == 3 {
		return Trilateration(
			points[0].ECEF,
			points[1].ECEF,
			points[2].ECEF,
			distances[0],
			distances[1],
			distances[2],
		), nil
	}

	// For more than 3 points, use least squares multilateration
	return multilaterationLS(points, distances)
}

// multilaterationLS performs multilateration using least squares method.
func multilaterationLS(points []struct {
	Coord Coord
	ECEF  ECEF
}, distances []float32) (ECEF, error) {
	n := len(points)
	if n < 3 {
		return ECEF{}, ErrInsufficientPoints
	}

	// Build matrix A for least squares: A * x = b
	// Each row represents: 2*(P[i] - P[0]) / d[i] - 2*(P[0]) / d[0]
	A := mat.New(n-1, 3)
	b := mat.New(n-1, 1)

	for i := range n - 1 {
		m := i + 1
		Am := 2.0*points[m].ECEF[0]/distances[m] - 2.0*points[0].ECEF[0]/distances[0]
		Bm := 2.0*points[m].ECEF[1]/distances[m] - 2.0*points[0].ECEF[1]/distances[0]
		Cm := 2.0*points[m].ECEF[2]/distances[m] - 2.0*points[0].ECEF[2]/distances[0]

		tm := vec.Vector3D(points[m].ECEF).SumSqr()
		t0 := vec.Vector3D(points[0].ECEF).SumSqr()

		Dm := distances[m] - distances[0] - tm/distances[m] + t0/distances[0]

		A[i][0] = Am
		A[i][1] = Bm
		A[i][2] = Cm
		b[i][0] = Dm
	}

	// Solve using pseudo-inverse: x = A^+ * b
	// Calculate pseudo-inverse of A
	APlus := mat.New(3, n-1)
	if err := A.PseudoInverse(APlus); err != nil {
		return ECEF{}, err
	}

	// Calculate x = A^+ * b
	x := mat.New(3, 1)
	x.Mul(APlus, b)

	return ECEF{
		x[0][0],
		x[1][0],
		x[2][0],
	}, nil
}
