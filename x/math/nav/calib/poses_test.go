package calib

import (
	"errors"
	"math"
	"testing"
)

// face is the true specific force (+g up) for the poseSensor lying on a face,
// tilted by small angles (rad) about two axes: what a table that isn't level
// gives.
func face(axis int, sign float64, tiltA, tiltB float64) [3]float64 {
	var f [3]float64
	a, b := (axis+1)%3, (axis+2)%3
	f[axis] = sign * StandardGravity * math.Cos(tiltA) * math.Cos(tiltB)
	f[a] = StandardGravity * math.Sin(tiltA)
	f[b] = StandardGravity * math.Cos(tiltA) * math.Sin(tiltB)
	return f
}

// poseSensor is a synthetic accelerometer: raw = scale∘f + offset, the pose's
// mean off by its noise averaged over n samples.
type poseSensor struct {
	offset, scale [3]float64
	noise         float64 // m/s² per sample
	n             int     // samples per pose
	r             *lcg
}

func (s poseSensor) pose(f [3]float64, gyroBias [3]float32) Pose {
	var p Pose
	for i := range 3 {
		mean := s.scale[i]*f[i] + s.offset[i]
		if s.n > 0 {
			mean += s.noise / math.Sqrt(float64(s.n)) * s.r.gauss()
		}
		p.Accel[i] = float32(mean)
		p.AccelNoise[i] = float32(s.noise)
		p.Gyro[i] = gyroBias[i]
		p.GyroNoise[i] = 0.002
	}
	return p
}

// seeded is a noise source started at seed.
func seeded(seed lcg) *lcg { return &seed }

// uniform draws from [0, 1).
func (r *lcg) uniform() float64 {
	*r = *r*1664525 + 1013904223
	return float64(*r) / (1 << 32)
}

// sixFaces are the six faces, each on a table tilted by 1–3°.
func sixFaces(r *lcg) [][3]float64 {
	var out [][3]float64
	for axis := range 3 {
		for _, sign := range []float64{1, -1} {
			out = append(out, face(axis, sign, (1+2*r.uniform())*math.Pi/180, (1+2*r.uniform())*math.Pi/180))
		}
	}
	return out
}

func solverWith(t *testing.T, poses ...Pose) *Solver {
	t.Helper()
	s, err := NewSolver(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range poses {
		if err := s.AddPose(p); err != nil {
			t.Fatalf("pose %d: %v", i, err)
		}
	}
	return &s
}

func near(got float32, want, tol float64) bool { return math.Abs(float64(got)-want) <= tol }

// Six faces, each on a slightly tilted table, recover the per-axis offset
// and scale: within 0.01 m/s² and 0.2 % at an MPU6050's noise (0.015 m/s²)
// averaged over 3 s at 100 Hz, and exactly (1e-4) without noise. The tilt
// doesn't matter: the fit uses only the force's length.
func TestSixFacesRecoverOffsetAndScale(t *testing.T) {
	for _, c := range []struct {
		name          string
		noise         float64
		offTol, scTol float64
	}{{"no noise", 0, 1e-4, 1e-5}, {"MPU6050 noise, 3 s", 0.015, 0.01, 0.002}} {
		r := seeded(1002)
		sen := poseSensor{offset: [3]float64{0.31, -0.22, 0.47}, scale: [3]float64{1.02, 0.97, 1.035}, noise: c.noise, n: 300, r: r}
		var poses []Pose
		for _, f := range sixFaces(r) {
			poses = append(poses, sen.pose(f, [3]float32{}))
		}
		fit, err := solverWith(t, poses...).Solve()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if fit.Model != OffsetScale || fit.Poses != 6 {
			t.Errorf("%s: model %v from %d poses, want offset+scale from 6", c.name, fit.Model, fit.Poses)
		}
		for i := range 3 {
			if !near(fit.Offset[i], sen.offset[i], c.offTol) || !near(fit.Scale[i], sen.scale[i], c.scTol) {
				t.Errorf("%s: axis %d: offset %.5f scale %.5f, want %.5f %.5f", c.name, i, fit.Offset[i], fit.Scale[i], sen.offset[i], sen.scale[i])
			}
		}
		if !(fit.Cond > 0 && fit.Cond < 10) {
			t.Errorf("%s: condition %v of a six-face fit, want small", c.name, fit.Cond)
		}
	}
}

// Three faces fit the offset (the scale taken as exact): with a true scale of
// 1 the offset comes back within 1e-4 without noise and 0.01 m/s² with an
// MPU6050's. Faces down on +z, +x and −y, tilted 1–3°.
func TestThreeFacesRecoverTheOffset(t *testing.T) {
	r := seeded(3004)
	for _, c := range []struct {
		noise, tol float64
	}{{0, 1e-4}, {0.015, 0.01}} {
		sen := poseSensor{offset: [3]float64{0.25, -0.4, 0.15}, scale: [3]float64{1, 1, 1}, noise: c.noise, n: 300, r: r}
		tilt := func() float64 { return (1 + 2*r.uniform()) * math.Pi / 180 }
		fit, err := solverWith(t,
			sen.pose(face(2, 1, tilt(), tilt()), [3]float32{}),
			sen.pose(face(0, 1, tilt(), tilt()), [3]float32{}),
			sen.pose(face(1, -1, tilt(), tilt()), [3]float32{}),
		).Solve()
		if err != nil {
			t.Fatal(err)
		}
		if fit.Model != OffsetOnly {
			t.Errorf("model %v from 3 poses, want offset only", fit.Model)
		}
		for i := range 3 {
			if !near(fit.Offset[i], sen.offset[i], c.tol) || fit.Scale[i] != 1 {
				t.Errorf("noise %v axis %d: offset %.5f scale %v, want %.5f and 1", c.noise, i, fit.Offset[i], fit.Scale[i], sen.offset[i])
			}
		}
	}
}

// Six faces and four tilted poses (about 45°) fit too, and the fitted model
// gives every pose |f| = g within the noise.
func TestTiltedPosesJoinTheFit(t *testing.T) {
	r := seeded(5006)
	sen := poseSensor{offset: [3]float64{-0.12, 0.3, 0.2}, scale: [3]float64{0.98, 1.01, 1.03}, noise: 0.015, n: 300, r: r}
	var poses []Pose
	for _, f := range sixFaces(r) {
		poses = append(poses, sen.pose(f, [3]float32{}))
	}
	for _, f := range [][3]float64{face(0, 1, 0.8, 0.5), face(1, -1, 0.6, 0.7), face(2, 1, 0.7, -0.6), face(2, -1, -0.5, 0.8)} {
		poses = append(poses, sen.pose(f, [3]float32{}))
	}
	fit, err := solverWith(t, poses...).Solve()
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		if !near(fit.Offset[i], sen.offset[i], 0.01) || !near(fit.Scale[i], sen.scale[i], 0.002) {
			t.Errorf("axis %d: offset %.4f scale %.4f, want %.4f %.4f", i, fit.Offset[i], fit.Scale[i], sen.offset[i], sen.scale[i])
		}
	}
	for i := range fit.Poses {
		f := fit.Apply(poses[i].Accel)
		if g := math.Sqrt(float64(f[0]*f[0] + f[1]*f[1] + f[2]*f[2])); math.Abs(g-StandardGravity) > 0.01 || !near(fit.Residual[i], g-StandardGravity, 1e-4) {
			t.Errorf("pose %d: |f| %.4f after Apply, residual %.5f", i, g, fit.Residual[i])
		}
	}
	if fit.RMS > 0.005 {
		t.Errorf("RMS residual %v at noise 0.0009 per pose", fit.RMS)
	}
}

// A pose not on the sphere (pressed on, or measured while moving) shows in
// its residual: well above the others', and in the RMS.
func TestTheResidualShowsABadPose(t *testing.T) {
	r := seeded(7008)
	sen := poseSensor{offset: [3]float64{0.1, 0.1, 0.1}, scale: [3]float64{1, 1, 1}, noise: 0.015, n: 300, r: r}
	var poses []Pose
	for _, f := range sixFaces(r) {
		poses = append(poses, sen.pose(f, [3]float32{}))
	}
	poses = append(poses, sen.pose(face(2, 1, 0.3, 0.4), [3]float32{}))
	poses[6].Accel[2] += 0.5 // pressed on
	fit, err := solverWith(t, poses...).Solve()
	if err != nil {
		t.Fatal(err)
	}
	worst := 0
	for i := range fit.Poses {
		if math.Abs(float64(fit.Residual[i])) > math.Abs(float64(fit.Residual[worst])) {
			worst = i
		}
	}
	if worst != 6 || !(math.Abs(float64(fit.Residual[6])) > 0.1) || !(fit.RMS > 0.05) {
		t.Errorf("worst pose %d, residual %.3f, RMS %.3f: want pose 6 standing out", worst, fit.Residual[6], fit.RMS)
	}
}

// The gyro bias is the mean of the poses' gyro means.
func TestTheGyroBiasIsTheMeanOverThePoses(t *testing.T) {
	r := seeded(9010)
	sen := poseSensor{scale: [3]float64{1, 1, 1}, r: r}
	var poses []Pose
	for k, f := range sixFaces(r) {
		poses = append(poses, sen.pose(f, [3]float32{0.01 + float32(k)*0.001, -0.02, 0.003}))
	}
	fit, err := solverWith(t, poses...).Solve()
	if err != nil {
		t.Fatal(err)
	}
	if want := [3]float64{0.0125, -0.02, 0.003}; !near(fit.GyroBias[0], want[0], 1e-6) || !near(fit.GyroBias[1], want[1], 1e-6) || !near(fit.GyroBias[2], want[2], 1e-6) {
		t.Errorf("gyro bias %v, want %v", fit.GyroBias, want)
	}
}

// A pose that wasn't still (spread above the Config's thresholds) is
// refused and not kept; so is one with a non-finite value.
func TestANonStillPoseIsRefused(t *testing.T) {
	s := solverWith(t)
	still := poseSensor{scale: [3]float64{1, 1, 1}}.pose(face(2, 1, 0, 0), [3]float32{})
	shaky := still
	shaky.AccelNoise[1] = 0.2
	turning := still
	turning.GyroNoise[0] = 0.05
	bad := still
	bad.Accel[0] = float32(math.NaN())
	for name, p := range map[string]Pose{"shaky": shaky, "turning": turning, "NaN": bad} {
		if err := s.AddPose(p); !errors.Is(err, ErrNotStill) {
			t.Errorf("%s: %v, want ErrNotStill", name, err)
		}
	}
	if s.Len() != 0 {
		t.Errorf("%d poses kept, want 0", s.Len())
	}
	if err := s.AddPose(still); err != nil || s.Len() != 1 {
		t.Errorf("the still pose: %v, %d kept", err, s.Len())
	}
}

// Fewer than three poses can't fit; poses that don't span the axes (all on
// one face, or two faces of one axis) can't either: ErrIllPosed, not a
// made-up fit.
func TestTooFewOrDegeneratePosesAreRefused(t *testing.T) {
	sen := poseSensor{offset: [3]float64{0.1, 0.2, 0.3}, scale: [3]float64{1, 1, 1}}
	up := sen.pose(face(2, 1, 0, 0), [3]float32{})
	if _, err := solverWith(t, up, sen.pose(face(0, 1, 0, 0), [3]float32{})).Solve(); !errors.Is(err, ErrTooFew) {
		t.Errorf("two poses: %v, want ErrTooFew", err)
	}
	if _, err := solverWith(t, up, up, up, up, up, up).Solve(); !errors.Is(err, ErrIllPosed) {
		t.Errorf("six poses on one face: %v, want ErrIllPosed", err)
	}
	if _, err := solverWith(t, up, sen.pose(face(2, -1, 0, 0), [3]float32{}), up).Solve(); !errors.Is(err, ErrIllPosed) {
		t.Errorf("z up and down only: %v, want ErrIllPosed", err)
	}
}

// The Solver keeps up to MaxPoses; Reset empties it.
func TestTheSolverIsBoundedAndResets(t *testing.T) {
	s := solverWith(t)
	p := poseSensor{scale: [3]float64{1, 1, 1}}.pose(face(2, 1, 0, 0), [3]float32{})
	for range MaxPoses {
		if err := s.AddPose(p); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.AddPose(p); !errors.Is(err, ErrFull) {
		t.Errorf("pose %d: %v, want ErrFull", MaxPoses+1, err)
	}
	s.Reset()
	if s.Len() != 0 {
		t.Errorf("%d poses after Reset", s.Len())
	}
}

// A finished Calibrator run is a pose: its mean force (the offset reading
// plus the rest target) and noise, and its gyro bias.
func TestACalibratorRunIsAPose(t *testing.T) {
	c, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := PoseOf(&c); ok {
		t.Error("a pose before the run is done")
	}
	force := [3]float32{0.2, -0.3, 9.7}
	gyro := [3]float32{0.004, -0.003, 0.002}
	for range 1000 {
		c.Update(0.01, force, gyro)
	}
	p, ok := PoseOf(&c)
	if !ok {
		t.Fatal("no pose after the run")
	}
	for i := range 3 {
		if !near(p.Accel[i], float64(force[i]), 1e-5) || !near(p.Gyro[i], float64(gyro[i]), 1e-6) || p.AccelNoise[i] != 0 {
			t.Errorf("axis %d: pose %+v", i, p)
		}
	}
}

// Add and Solve allocate nothing.
func TestTheSolverDoesNotAllocate(t *testing.T) {
	r := seeded(11012)
	sen := poseSensor{offset: [3]float64{0.1, 0.2, 0.3}, scale: [3]float64{1.01, 0.99, 1}, noise: 0.015, n: 300, r: r}
	var poses []Pose
	for _, f := range sixFaces(r) {
		poses = append(poses, sen.pose(f, [3]float32{}))
	}
	s := solverWith(t)
	if n := testing.AllocsPerRun(20, func() {
		s.Reset()
		for _, p := range poses {
			_ = s.AddPose(p)
		}
		if _, err := s.Solve(); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("%v allocations per run", n)
	}
}

// onSphere draws a direction uniform on the sphere, times g.
func onSphere(r *lcg) [3]float64 {
	z := 2*r.uniform() - 1
	ph := 2 * math.Pi * r.uniform()
	s := math.Sqrt(1 - z*z)
	return [3]float64{StandardGravity * s * math.Cos(ph), StandardGravity * s * math.Sin(ph), StandardGravity * z}
}

// inCone draws a direction within half (rad) of +z, uniform on the cap,
// times g.
func inCone(r *lcg, half float64) [3]float64 {
	cz := 1 - r.uniform()*(1-math.Cos(half))
	sz := math.Sqrt(1 - cz*cz)
	ph := 2 * math.Pi * r.uniform()
	return [3]float64{StandardGravity * sz * math.Cos(ph), StandardGravity * sz * math.Sin(ph), StandardGravity * cz}
}

// Poses that don't look in enough directions are refused: in one plane (the
// badge turned about one axis), or all within 5° of one direction. Every fit
// that is accepted is accurate: of 500 sets of 8 random poses at an
// MPU6050's noise, those accepted (about two thirds; the rest leave some
// direction under-seen) have their offsets within 0.006 m/s² and scales
// within 0.12 % (a condition scaled to a unit
// diagonal let coplanar sets through, 0.16 m/s² off).
func TestDegeneratePoseSetsAreRefusedAndAcceptedFitsAreAccurate(t *testing.T) {
	r := seeded(13014)
	one := poseSensor{offset: [3]float64{0.3, -0.25, 0.2}, scale: [3]float64{1, 1, 1}, noise: 0.015, n: 300, r: r}
	plane := solverWith(t)
	for k := range 8 {
		th := float64(k)*math.Pi/4 + 0.3
		_ = plane.AddPose(one.pose([3]float64{StandardGravity * math.Cos(th), StandardGravity * math.Sin(th), 0}, [3]float32{}))
	}
	if f, err := plane.Solve(); !errors.Is(err, ErrIllPosed) {
		t.Errorf("8 poses in a plane: %v (offset %v, cond %v), want ErrIllPosed", err, f.Offset, f.Cond)
	}
	cone := solverWith(t)
	for range 8 {
		_ = cone.AddPose(one.pose(inCone(r, 5*math.Pi/180), [3]float32{}))
	}
	if f, err := cone.SolveModel(OffsetOnly); !errors.Is(err, ErrIllPosed) {
		t.Errorf("8 poses within 5°: %v (offset %v, cond %v), want ErrIllPosed", err, f.Offset, f.Cond)
	}

	sen := poseSensor{offset: [3]float64{0.3, -0.25, 0.2}, scale: [3]float64{1.02, 0.98, 1.01}, noise: 0.015, n: 300, r: r}
	accepted := 0
	for k := range 500 {
		s := solverWith(t)
		for range 8 {
			_ = s.AddPose(sen.pose(onSphere(r), [3]float32{}))
		}
		f, err := s.Solve()
		if errors.Is(err, ErrIllPosed) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		accepted++
		for i := range 3 {
			if !near(f.Offset[i], sen.offset[i], 0.006) || !near(f.Scale[i], sen.scale[i], 0.0012*sen.scale[i]) {
				t.Errorf("set %d (cond %.1f): axis %d offset %.4f scale %.4f, want %.4f %.4f", k, f.Cond, i, f.Offset[i], f.Scale[i], sen.offset[i], sen.scale[i])
			}
		}
	}
	if accepted < 250 {
		t.Errorf("%d of 500 random sets accepted, want most", accepted)
	}
}

// The condition number reads how evenly the poses span the axes: about 1
// for the six faces and for three faces of different axes, larger for
// poses within a 30° cone.
func TestTheConditionNumberReadsThePoseSpread(t *testing.T) {
	exact := poseSensor{offset: [3]float64{0.3, -0.25, 0.2}, scale: [3]float64{1.02, 0.98, 1.01}}
	var six []Pose
	for axis := range 3 {
		for _, sg := range []float64{1, -1} {
			six = append(six, exact.pose(face(axis, sg, 0, 0), [3]float32{}))
		}
	}
	f6, err := solverWith(t, six...).Solve()
	if err != nil || !(f6.Cond > 1 && f6.Cond < 1.2) {
		t.Errorf("six faces: cond %v %v, want about 1", f6.Cond, err)
	}
	f3, err := solverWith(t, six[0], six[2], six[5]).Solve()
	if err != nil || !(f3.Cond > 1 && f3.Cond < 1.2) {
		t.Errorf("three faces: cond %v %v, want about 1", f3.Cond, err)
	}
	r := seeded(15016)
	unit := poseSensor{offset: exact.offset, scale: [3]float64{1, 1, 1}}
	cone := solverWith(t)
	for range 8 {
		_ = cone.AddPose(unit.pose(inCone(r, 30*math.Pi/180), [3]float32{}))
	}
	fc, err := cone.SolveModel(OffsetOnly)
	if err != nil || !(fc.Cond > 2 && fc.Cond < MaxCond) {
		t.Errorf("8 poses within 30°: cond %v %v, want between 2 and %d", fc.Cond, err, MaxCond)
	}
}

// The RMS is the root mean square of the poses' residuals.
func TestTheRMSIsTheResidualsRootMeanSquare(t *testing.T) {
	r := seeded(17018)
	sen := poseSensor{offset: [3]float64{0.1, 0.1, 0.1}, scale: [3]float64{1, 1, 1}, noise: 0.015, n: 300, r: r}
	var poses []Pose
	for _, f := range sixFaces(r) {
		poses = append(poses, sen.pose(f, [3]float32{}))
	}
	poses = append(poses, sen.pose(face(2, 1, 0.3, 0.4), [3]float32{}), sen.pose(face(0, -1, 0.5, 0.2), [3]float32{}))
	poses[7].Accel[0] += 0.2
	fit, err := solverWith(t, poses...).Solve()
	if err != nil {
		t.Fatal(err)
	}
	var sq float64
	for i := range fit.Poses {
		sq += float64(fit.Residual[i]) * float64(fit.Residual[i])
	}
	if want := math.Sqrt(sq / float64(fit.Poses)); !near(fit.RMS, want, 1e-6*want) {
		t.Errorf("RMS %v, want %v", fit.RMS, want)
	}
}

// Four and five poses fit the offset alone; six switch to offset and scale.
// The faces come in the order a guided run asks for them: three different
// axes first (+x, +y, +z), then the opposite faces.
func TestFourAndFivePosesFitTheOffset(t *testing.T) {
	r := seeded(19020)
	sen := poseSensor{offset: [3]float64{0.2, -0.1, 0.35}, scale: [3]float64{1, 1, 1}, noise: 0.015, n: 300, r: r}
	six := sixFaces(r) // +x, −x, +y, −y, +z, −z
	faces := [][3]float64{six[0], six[2], six[4], six[1], six[3], six[5]}
	for n := 4; n <= 6; n++ {
		var poses []Pose
		for _, f := range faces[:n] {
			poses = append(poses, sen.pose(f, [3]float32{}))
		}
		fit, err := solverWith(t, poses...).Solve()
		if err != nil {
			t.Fatalf("%d poses: %v", n, err)
		}
		want := OffsetOnly
		if n == 6 {
			want = OffsetScale
		}
		if fit.Model != want {
			t.Errorf("%d poses: model %v, want %v", n, fit.Model, want)
		}
		for i := range 3 {
			if !near(fit.Offset[i], sen.offset[i], 0.01) {
				t.Errorf("%d poses: axis %d offset %.4f, want %.4f", n, i, fit.Offset[i], sen.offset[i])
			}
		}
	}
}

// NewSolver refuses a Config without a positive, finite G and stillness
// thresholds.
func TestNewSolverRefusesABadConfig(t *testing.T) {
	for name, mod := range map[string]func(*Config){
		"G 0":            func(c *Config) { c.G = 0 },
		"G NaN":          func(c *Config) { c.G = float32(math.NaN()) },
		"AccelStill 0":   func(c *Config) { c.AccelStill = 0 },
		"GyroStill -1":   func(c *Config) { c.GyroStill = -1 },
		"AccelStill Inf": func(c *Config) { c.AccelStill = float32(math.Inf(1)) },
	} {
		c := DefaultConfig()
		mod(&c)
		if _, err := NewSolver(c); !errors.Is(err, ErrConfig) {
			t.Errorf("%s: %v, want ErrConfig", name, err)
		}
	}
}

// Six poses (offset-only fit) on which Gauss-Newton does not settle in 200
// iterations: without the convergence check it returned a fit of condition 8.9 whose offsets are
// off by 6 m/s²; the convergence check refuses it.
func TestANonConvergedFitIsRefused(t *testing.T) {
	s, _ := NewSolver(DefaultConfig())
	for _, a := range [][3]float32{
		{-3.0600805, 2.665948, -9.1866255}, {1.3461702, 2.8475528, 3.0076733}, {1.3480643, -0.57198256, 8.445862},
		{1.3400834, -1.792725, 5.37704}, {1.2271448, -2.58245, 8.397431}, {1.3572617, -3.8390985, 8.155185},
	} {
		if err := s.AddPose(Pose{Accel: a, AccelNoise: [3]float32{0.015, 0.015, 0.015}, GyroNoise: [3]float32{0.002, 0.002, 0.002}}); err != nil {
			t.Fatal(err)
		}
	}
	if fit, err := s.SolveModel(OffsetOnly); !errors.Is(err, ErrIllPosed) {
		t.Errorf("SolveModel on poses that don't converge: %v offsets %v cond %v, want ErrIllPosed", err, fit.Offset, fit.Cond)
	}
}
