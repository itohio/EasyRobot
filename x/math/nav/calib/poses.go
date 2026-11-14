package calib

import (
	"errors"
	"math"
)

// Multi-pose calibration (CAL-1): one pose cannot tell the accelerometer's
// offset from the surface's tilt (see the package doc), but several can. At
// rest the true specific force has length g whatever the pose, so the
// readings a of a sensor with per-axis offset o and scale s (a = s∘f + o)
// lie on an ellipsoid; the fit finds the o and s that put every pose's
// corrected force f = (a − o)/s on the sphere |f| = g. The tilt of the table
// doesn't matter, only that the poses look in different directions:
//
//   - 3 to 5 poses fit the offset alone (OffsetOnly, the scale taken as 1):
//     the owner lays the badge on at least three faces whose up axes differ
//     (flat, on its left side, upright). A scale error then leaks into the
//     offset, by about the error times g (0.2 m/s² for 2 %) on each axis.
//   - 6 or more fit offset and scale (OffsetScale): the six faces, best with
//     some tilted poses besides.
//
// The fit is Gauss-Newton with Levenberg damping on the residuals |f| − g,
// in float64. Its quality: each pose's residual after the fit (|f| − g, in
// m/s²: a pose pressed on, or not quite still, stands out, given more poses
// than unknowns; with exactly 3 or 6 the fit passes through every pose and
// the residuals are 0), their RMS, and the fit's condition number
// √(λmax/λmin) of the normal matrix JᵀJ. The parameters are already scaled
// alike (the offsets in m/s², the inverse scales times g), so it isn't
// rescaled: a direction the poses don't see shows as a small λmin. It is
// about 1 when the poses span the axes evenly (six faces 1.2, three faces
// 1.1), 4–12 for poses within a 30° cone, 7–20 within 15° (offset only), and
// 10⁴ and more for poses in one plane. Above MaxCond (10) the fit is refused
// (ErrIllPosed): below it the offsets came within 0.006 m/s² and the scales
// 0.12 % at an MPU6050's noise over 3 s, where up to 20 let 0.021 m/s²
// through (four faces seeing z only through the table's tilt; verify's
// probes and this package's tests, CAL-1). The gyro bias is the mean of the
// poses' gyro means.
//
// A Solver keeps up to MaxPoses poses in itself: AddPose, Solve and Reset
// allocate nothing.

// MaxPoses is how many poses a Solver keeps.
const MaxPoses = 24

// MaxCond is the largest condition number a fit is accepted with (see the
// fit's quality above).
const MaxCond = 10

var (
	// ErrNotStill: AddPose of a pose whose spread is above the Config's
	// stillness thresholds, or with a value that isn't finite.
	ErrNotStill = errors.New("calib: the pose wasn't still")
	// ErrFull: AddPose of a pose beyond MaxPoses.
	ErrFull = errors.New("calib: no room for another pose")
	// ErrTooFew: Solve with fewer poses than the model has unknowns
	// (3 for the offset, 6 for offset and scale).
	ErrTooFew = errors.New("calib: too few poses")
	// ErrIllPosed: the poses don't fix the fit (they don't look in enough
	// different directions), or it didn't converge.
	ErrIllPosed = errors.New("calib: the poses don't fix the fit")
)

// Pose is one still pose: the means over its samples and their standard
// deviations, in the sensor frame (m/s², rad/s).
type Pose struct {
	Accel, Gyro           [3]float32
	AccelNoise, GyroNoise [3]float32
}

// PoseOf is the pose of a finished Calibrator run: its mean force (the
// offset reading plus the rest target), its gyro bias and the noise of
// both; false until the run is Done.
func PoseOf(c *Calibrator) (Pose, bool) {
	r, ok := c.Result()
	if !ok {
		return Pose{}, false
	}
	rest := c.AccelRest()
	p := Pose{Gyro: r.GyroBias, AccelNoise: r.AccelNoise, GyroNoise: r.GyroNoise}
	for i := range 3 {
		p.Accel[i] = r.AccelOffset[i] + rest[i]
	}
	return p, true
}

// Model is what a fit solves for.
type Model uint8

const (
	OffsetOnly  Model = iota // per-axis offset, scale 1
	OffsetScale              // per-axis offset and scale
)

func (m Model) String() string {
	if m == OffsetScale {
		return "offset+scale"
	}
	return "offset"
}

// Fit is a solved calibration. Apply corrects a sample with it.
type Fit struct {
	Model    Model
	Offset   [3]float32        // m/s²: subtract from the accelerometer's samples
	Scale    [3]float32        // raw per true (1 for OffsetOnly): divide by it after the offset
	GyroBias [3]float32        // rad/s: subtract from the gyro's samples
	Poses    int               // how many poses it fitted
	Residual [MaxPoses]float32 // per pose, |f| − g after the fit (m/s²)
	RMS      float32           // of the residuals
	Cond     float32           // the fit's condition number (1: the poses span the axes evenly)
}

// Apply corrects an accelerometer sample: (a − Offset)/Scale.
func (f *Fit) Apply(a [3]float32) [3]float32 {
	var out [3]float32
	for i := range 3 {
		out[i] = (a[i] - f.Offset[i]) / f.Scale[i]
	}
	return out
}

// Solver collects still poses and fits them.
type Solver struct {
	cfg   Config
	poses [MaxPoses]Pose
	n     int
	// the fit's scratch (6×6 float64 is 288 bytes: TinyGo would put each
	// on the heap as a local)
	jtj, a, eig [6][6]float64
}

// NewSolver checks c (its G and stillness thresholds) and returns an empty
// Solver.
func NewSolver(c Config) (Solver, error) {
	if !(c.G > 0) || !(c.AccelStill > 0) || !(c.GyroStill > 0) || !finite(c.G+c.AccelStill+c.GyroStill) {
		return Solver{}, ErrConfig
	}
	return Solver{cfg: c}, nil
}

// Len is how many poses the Solver holds.
func (s *Solver) Len() int { return s.n }

// Reset drops every pose.
func (s *Solver) Reset() { s.n = 0 }

// AddPose keeps p; ErrNotStill if any axis's spread is at or above the Config's
// threshold (the Calibrator's own stillness test) or a value isn't finite,
// ErrFull past MaxPoses.
func (s *Solver) AddPose(p Pose) error {
	for i := range 3 {
		if !finite(p.Accel[i]+p.Gyro[i]+p.AccelNoise[i]+p.GyroNoise[i]) || !(p.AccelNoise[i] < s.cfg.AccelStill) || !(p.GyroNoise[i] < s.cfg.GyroStill) {
			return ErrNotStill
		}
	}
	if s.n == MaxPoses {
		return ErrFull
	}
	s.poses[s.n] = p
	s.n++
	return nil
}

// Solve fits the poses held: offset and scale from 6 or more, the offset
// alone from 3 to 5.
func (s *Solver) Solve() (Fit, error) {
	if s.n >= 6 {
		return s.SolveModel(OffsetScale)
	}
	return s.SolveModel(OffsetOnly)
}

// SolveModel fits the poses held for model m (OffsetOnly may take any number
// from 3).
func (s *Solver) SolveModel(m Model) (Fit, error) {
	np := 3
	if m == OffsetScale {
		np = 6
	}
	if s.n < np {
		return Fit{}, ErrTooFew
	}
	g := float64(s.cfg.G)
	// θ: offset o (3), then the inverse scale k (3) times g, so that every
	// column of the Jacobian is about 1 in size.
	var th [6]float64
	th[3], th[4], th[5] = g, g, g
	cost := s.cost(&th, g)
	lambda := 1e-3
	converged := false
	for iter := 0; iter < 200 && !converged; iter++ {
		s.jtj = [6][6]float64{}
		var jtr [6]float64
		s.normal(&th, g, np, &s.jtj, &jtr)
		for {
			s.a = [6][6]float64{}
			var b [6]float64
			for i := range np {
				for j := range np {
					s.a[i][j] = s.jtj[i][j]
				}
				s.a[i][i] += lambda * max(s.jtj[i][i], 1e-12)
				b[i] = -jtr[i]
			}
			step, ok := solve(&s.a, &b, np)
			if !ok {
				return Fit{}, ErrIllPosed
			}
			next := th
			size := 0.0
			for i := range np {
				next[i] += step[i]
				size = max(size, math.Abs(step[i]))
			}
			if c := s.cost(&next, g); c <= cost {
				th, cost = next, c
				lambda = max(lambda/10, 1e-12)
				converged = size < 1e-12*g
				break
			}
			lambda *= 10
			if lambda > 1e12 {
				converged = true // no step lowers the cost: at the minimum
				break
			}
		}
	}
	if !converged {
		return Fit{}, ErrIllPosed
	}
	s.jtj = [6][6]float64{}
	var jtr [6]float64
	s.normal(&th, g, np, &s.jtj, &jtr)
	cond := condition(&s.jtj, np, &s.eig)
	if !(cond <= MaxCond) {
		return Fit{}, ErrIllPosed
	}

	f := Fit{Model: m, Poses: s.n, Cond: float32(cond)}
	for i := range 3 {
		f.Offset[i] = float32(th[i])
		f.Scale[i] = float32(g / th[3+i])
	}
	var sq float64
	for p := range s.n {
		r := s.residual(&th, g, p)
		f.Residual[p] = float32(r)
		sq += r * r
		for i := range 3 {
			f.GyroBias[i] += s.poses[p].Gyro[i]
		}
	}
	for i := range 3 {
		f.GyroBias[i] /= float32(s.n)
	}
	f.RMS = float32(math.Sqrt(sq / float64(s.n)))
	return f, nil
}

// residual is pose p's |f| − g at θ.
func (s *Solver) residual(th *[6]float64, g float64, p int) float64 {
	var n2 float64
	for i := range 3 {
		v := th[3+i] / g * (float64(s.poses[p].Accel[i]) - th[i])
		n2 += v * v
	}
	return math.Sqrt(n2) - g
}

// cost is the sum of the squared residuals at θ.
func (s *Solver) cost(th *[6]float64, g float64) float64 {
	var c float64
	for p := range s.n {
		r := s.residual(th, g, p)
		c += r * r
	}
	return c
}

// normal builds JᵀJ and Jᵀr for the first np parameters at θ.
func (s *Solver) normal(th *[6]float64, g float64, np int, jtj *[6][6]float64, jtr *[6]float64) {
	for p := range s.n {
		var v, d [3]float64
		var n2 float64
		for i := range 3 {
			d[i] = float64(s.poses[p].Accel[i]) - th[i]
			v[i] = th[3+i] / g * d[i]
			n2 += v[i] * v[i]
		}
		n := math.Sqrt(n2)
		if n == 0 {
			continue
		}
		r := n - g
		var j [6]float64
		for i := range 3 {
			j[i] = -th[3+i] / g * v[i] / n // ∂r/∂o
			j[3+i] = d[i] / g * v[i] / n   // ∂r/∂(k·g)
		}
		for a := range np {
			jtr[a] += j[a] * r
			for b := range np {
				jtj[a][b] += j[a] * j[b]
			}
		}
	}
}

// solve solves a·x = b for the first n unknowns by Gaussian elimination with
// partial pivoting; false if a is singular.
func solve(a *[6][6]float64, b *[6]float64, n int) ([6]float64, bool) {
	var x [6]float64
	for c := range n {
		p := c
		for r := c + 1; r < n; r++ {
			if math.Abs(a[r][c]) > math.Abs(a[p][c]) {
				p = r
			}
		}
		if !(math.Abs(a[p][c]) > 1e-300) {
			return x, false
		}
		a[c], a[p] = a[p], a[c]
		b[c], b[p] = b[p], b[c]
		for r := c + 1; r < n; r++ {
			f := a[r][c] / a[c][c]
			for k := c; k < n; k++ {
				a[r][k] -= f * a[c][k]
			}
			b[r] -= f * b[c]
		}
	}
	for c := n - 1; c >= 0; c-- {
		s := b[c]
		for k := c + 1; k < n; k++ {
			s -= a[c][k] * x[k]
		}
		x[c] = s / a[c][c]
	}
	return x, true
}

// condition is √(λmax/λmin) of the first n×n of the symmetric m (Jacobi
// eigenvalues, in the scratch a); +Inf if singular. No diagonal scaling: that would bring a
// column the poses barely see back to size 1 and hide the degeneracy.
func condition(m *[6][6]float64, n int, a *[6][6]float64) float64 {
	*a = [6][6]float64{}
	for i := range n {
		for j := range n {
			a[i][j] = m[i][j]
		}
	}
	for sweep := 0; sweep < 50; sweep++ {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += a[i][j] * a[i][j]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				if a[p][q] == 0 {
					continue
				}
				theta := (a[q][q] - a[p][p]) / (2 * a[p][q])
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range n {
					akp, akq := a[k][p], a[k][q]
					a[k][p], a[k][q] = c*akp-s*akq, s*akp+c*akq
				}
				for k := range n {
					apk, aqk := a[p][k], a[q][k]
					a[p][k], a[q][k] = c*apk-s*aqk, s*apk+c*aqk
				}
			}
		}
	}
	lo, hi := math.Inf(1), 0.0
	for i := range n {
		lo, hi = min(lo, a[i][i]), max(hi, a[i][i])
	}
	if !(lo > 0) {
		return math.Inf(1)
	}
	return math.Sqrt(hi / lo)
}
