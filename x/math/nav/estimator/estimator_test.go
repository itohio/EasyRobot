package estimator

import (
	"math"
	"testing"
)

// levels checks that q turns f to (0, 0, |f|).
func levels(t *testing.T, f [3]float32, q [4]float32) {
	t.Helper()
	w, x, y, z := float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])
	r := [3][3]float64{
		{1 - 2*(y*y+z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)},
		{2 * (x*y + w*z), 1 - 2*(x*x+z*z), 2 * (y*z - w*x)},
		{2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x+y*y)},
	}
	n := math.Sqrt(float64(f[0])*float64(f[0]) + float64(f[1])*float64(f[1]) + float64(f[2])*float64(f[2]))
	for i := range 3 {
		v := r[i][0]*float64(f[0]) + r[i][1]*float64(f[1]) + r[i][2]*float64(f[2])
		want := 0.0
		if i == 2 {
			want = n
		}
		if math.Abs(v-want) > 1e-4*n {
			t.Errorf("%v: R·f = %.5f on axis %d (want %.5f)", f, v, i, want)
		}
	}
}

// Level turns the force's direction to ẑ: the quaternion applied to the
// force gives (0, 0, |f|), in any pose, upside down too; a zero force has
// no level.
func TestLevel(t *testing.T) {
	for _, f := range [][3]float32{{0, 0, 9.8}, {0, 9.8, 0}, {3, -4, 5}, {0, 0, -9.8}, {1e-3, 0, -9.8}, {-7, 2, -0.5}} {
		q, ok := Level(f)
		if !ok {
			t.Fatalf("%v: no level", f)
		}
		levels(t, f, q)
	}
	if _, ok := Level([3]float32{}); ok {
		t.Errorf("a zero force has a level")
	}
}

// Level's quaternion is a unit one however close to upside down the force
// is: |q| within 1e-6 of 1 as 1 + u_z goes from 0.1 to 1e-12 and to 0, the
// force an accelerometer's (9.81 m/s², so float32's |f| is not exact).
func TestLevelIsUnitNearUpsideDown(t *testing.T) {
	for _, diag := range []bool{false, true} {
		for k := 1; k <= 13; k++ {
			uz := -1.0
			if k <= 12 {
				uz = -1 + math.Pow(10, -float64(k))
			}
			h := math.Sqrt(1 - uz*uz)
			u := [3]float64{h, 0, uz}
			if diag {
				u = [3]float64{h / math.Sqrt2, -h / math.Sqrt2, uz}
			}
			f := [3]float32{float32(9.81 * u[0]), float32(9.81 * u[1]), float32(9.81 * u[2])}
			q, ok := Level(f)
			if !ok {
				t.Fatalf("%v: no level", f)
			}
			n := math.Sqrt(float64(q[0])*float64(q[0]) + float64(q[1])*float64(q[1]) + float64(q[2])*float64(q[2]) + float64(q[3])*float64(q[3]))
			if math.Abs(n-1) > 1e-6 {
				t.Errorf("1+u_z %.0e (diagonal %v): |q| − 1 = %.3g (want within 1e-6)", 1+uz, diag, n-1)
			}
			levels(t, f, q)
		}
	}
}
