package dh

import (
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const pi = math32.Pi

func rev(theta, d, a, alpha, lo, hi float32) Config {
	return Config{Min: lo, Max: hi, Theta: theta, Alpha: alpha, R: a, D: d, Index: 0}
}

// hexleg is the cmd/manipulator hexapod leg (39/45/100 mm) in metres with the
// review's joint limits (coxa ±90°, femur ±90°, tibia ±162°).
func hexleg() []Config {
	return []Config{
		rev(0, 0, 0.039, pi/2, -pi/2, pi/2),
		rev(0, 0, 0.045, 0, -pi/2, pi/2),
		rev(0, 0, 0.100, 0, -pi*0.9, pi*0.9),
	}
}

var hexlegNominal = []float32{0, 0.3, -1.6}

func newChain(t testing.TB, eps float32, maxIter int, cfg ...Config) *DenavitHartenberg {
	t.Helper()
	k, err := New(eps, maxIter, cfg...)
	require.NoError(t, err)
	return k
}

func fkPos(t testing.TB, k *DenavitHartenberg, q []float32) [3]float32 {
	t.Helper()
	st := mat.New(len(q), 1)
	for i, v := range q {
		st[i][0] = v
	}
	dst := mat.New(effectorSize, 1)
	require.NoError(t, k.Forward(st, dst, nil))
	return [3]float32{dst[0][0], dst[1][0], dst[2][0]}
}

// solve runs Backward from `seed` towards `target`. Controls start as NaN so a
// solver that leaves them untouched is detectable.
func solve(k *DenavitHartenberg, seed []float32, target [3]float32) ([]float32, error) {
	st := mat.New(len(seed), 1)
	for i, v := range seed {
		st[i][0] = v
	}
	dst := mat.New(effectorSize, 1)
	dst[0][0], dst[1][0], dst[2][0] = target[0], target[1], target[2]
	dst[6][0] = 1
	ctl := mat.New(len(seed), 1)
	for i := range ctl {
		ctl[i][0] = math32.NaN()
	}
	err := k.Backward(st, dst, ctl)
	q := make([]float32, len(seed))
	for i := range q {
		q[i] = ctl[i][0]
	}
	return q, err
}

func dist(a, b [3]float32) float32 {
	dx, dy, dz := a[0]-b[0], a[1]-b[1], a[2]-b[2]
	return math32.Sqrt(dx*dx + dy*dy + dz*dz)
}

func sampleQ(r *rand.Rand, lo, hi []float32) []float32 {
	q := make([]float32, len(lo))
	for i := range q {
		q[i] = lo[i] + r.Float32()*(hi[i]-lo[i])
	}
	return q
}

func limits(cfg []Config) (lo, hi []float32) {
	lo = make([]float32, len(cfg))
	hi = make([]float32, len(cfg))
	for i, c := range cfg {
		lo[i], hi[i] = c.Min, c.Max
	}
	return lo, hi
}

// convergence counts targets FK(q_true) that Backward solves (err == nil and an
// independently checked residual ≤ eps).
func convergence(t *testing.T, cfg []Config, seedFn func(qTrue []float32) []float32, lo, hi []float32, n int, rngSeed uint64) int {
	t.Helper()
	const eps = 1e-4
	r := rand.New(rand.NewPCG(rngSeed, 0))
	k := newChain(t, eps, 100, cfg...)
	ref := newChain(t, eps, 100, cfg...)
	ok := 0
	for range n {
		qt := sampleQ(r, lo, hi)
		target := fkPos(t, ref, qt)
		q, err := solve(k, seedFn(qt), target)
		if err != nil {
			continue
		}
		if d := dist(fkPos(t, ref, q), target); d <= eps*1.01 {
			ok++
		} else {
			t.Errorf("err == nil but residual %.3g > eps for q_true %v", d, qt)
		}
	}
	return ok
}

// Chains with more than three joints (J is 3×n, n > 3) must solve
// reachable targets (review probes G1/G2).
func TestBackward_RedundantChains(t *testing.T) {
	planar4 := []Config{rev(0, 0, 1, 0, -pi, pi), rev(0, 0, 1, 0, -pi, pi), rev(0, 0, 1, 0, -pi, pi), rev(0, 0, 1, 0, -pi, pi)}
	puma := []Config{
		rev(0, 0.67, 0, pi/2, -pi, pi),
		rev(0, 0, 0.432, 0, -pi, pi),
		rev(0, 0.15, 0.02, -pi/2, -pi, pi),
		rev(0, 0.432, 0, pi/2, -pi, pi),
		rev(0, 0, 0, -pi/2, -pi, pi),
		rev(0, 0.056, 0, 0, -pi, pi),
	}
	tests := []struct {
		name    string
		cfg     []Config
		qTrue   []float32
		seed    []float32
		minRate float32 // over 300 random reachable targets from `seed`
	}{
		{"G2 4-link planar", planar4, []float32{0.2, 0.3, 0.4, 0.5}, []float32{0.1, 0.1, 0.1, 0.1}, 0.98},
		{"G1 6-DOF PUMA-like", puma, []float32{0.1, 0.4, 0.2, 0.1, 0.6, 0.1}, []float32{0, 0.3, 0.3, 0, 0.5, 0}, 0.98},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := newChain(t, 1e-4, 100, tt.cfg...)
			target := fkPos(t, k, tt.qTrue)
			q, err := solve(k, tt.seed, target)
			require.NoError(t, err)
			assert.LessOrEqual(t, dist(fkPos(t, k, q), target), float32(1e-4))

			lo, hi := limits(tt.cfg)
			const n = 300
			ok := convergence(t, tt.cfg, func([]float32) []float32 { return tt.seed }, lo, hi, n, 1)
			t.Logf("%s: %d/%d random reachable targets converged", tt.name, ok, n)
			assert.GreaterOrEqual(t, float32(ok)/n, tt.minRate)
		})
	}
}

// Index 1 (α) and Index 2 (r) joints are valid Config kinds; the Jacobian must
// support them so Backward solves reachable targets.
func TestBackward_AlphaAndLengthJoints(t *testing.T) {
	cfg := []Config{
		{Min: -pi, Max: pi, R: 0.5, Alpha: pi / 2, Index: 0},
		{Min: -pi / 2, Max: pi / 2, Theta: pi / 4, R: 0.5, Index: 1},
		{Min: 0.1, Max: 0.6, Theta: pi / 3, Index: 2},
	}
	lo, hi := limits(cfg)
	seed := []float32{0, 0, 0.35}
	const n = 200
	ok := convergence(t, cfg, func([]float32) []float32 { return seed }, lo, hi, n, 3)
	t.Logf("α/r joints: %d/%d converged", ok, n)
	assert.GreaterOrEqual(t, float32(ok)/n, float32(0.9))
}

// Realistic hexapod foot workspace (review probe R): coxa ±60°,
// femur −60…72°, tibia −149…−20°; nominal seed must converge ≥ 95 % within
// 100 iterations.
func TestBackward_HexlegRealisticWorkspace(t *testing.T) {
	lo := []float32{-pi / 3, -pi / 3, -2.6}
	hi := []float32{pi / 3, pi / 2.5, -0.35}
	const n = 500
	nominal := convergence(t, hexleg(), func([]float32) []float32 { return hexlegNominal }, lo, hi, n, 5)
	zero := convergence(t, hexleg(), func([]float32) []float32 { return []float32{0, 0, 0} }, lo, hi, n, 5)
	wr := rand.New(rand.NewPCG(7, 0))
	warm := convergence(t, hexleg(), func(qt []float32) []float32 {
		s := make([]float32, len(qt))
		for i := range s {
			s[i] = qt[i] + (wr.Float32()*2-1)*0.2
		}
		return s
	}, lo, hi, n, 5)
	t.Logf("probe R: nominal %d/%d, zero %d/%d, warm±0.2 %d/%d", nominal, n, zero, n, warm, n)
	assert.GreaterOrEqual(t, nominal, 475, "nominal seed")
	assert.GreaterOrEqual(t, warm, 495, "warm seed")
}

// Failures are typed and still write a best-effort q to controls.
func TestBackward_TypedFailures(t *testing.T) {
	ref := newChain(t, 1e-4, 100, hexleg()...)
	reach := float32(0.039 + 0.045 + 0.100)

	tibiaLimited := hexleg()
	tibiaLimited[2] = rev(0, 0, 0.100, 0, -pi/2, 0)

	tests := []struct {
		name   string
		cfg    []Config
		seed   []float32
		target [3]float32
		want   error
		check  func(t *testing.T, q []float32, target [3]float32)
	}{
		{
			name:   "F1 target needs tibia -2.0, tibia limit [-pi/2,0]",
			cfg:    tibiaLimited,
			seed:   []float32{0, 0.3, -1.0},
			target: fkPos(t, ref, []float32{0, 0.6, -2.0}),
			want:   kintypes.ErrJointLimit,
			check: func(t *testing.T, q []float32, _ [3]float32) {
				assert.InDelta(t, -pi/2, q[2], 1e-5, "tibia pinned at its limit")
			},
		},
		{
			name:   "E1 unreachable 1.5*reach on +x",
			cfg:    hexleg(),
			seed:   hexlegNominal,
			target: [3]float32{1.5 * reach, 0, 0},
			want:   kintypes.ErrUnreachable,
			check: func(t *testing.T, q []float32, _ [3]float32) {
				got := fkPos(t, ref, q)
				assert.Less(t, dist(got, [3]float32{reach, 0, 0}), float32(1e-3), "best effort is the stretched leg towards the target, got %v", got)
			},
		},
		{
			name:   "E2 target behind body needs coxa 180°, limit ±90°",
			cfg:    hexleg(),
			seed:   hexlegNominal,
			target: [3]float32{-0.1, 0, -0.05},
			want:   kintypes.ErrJointLimit,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := newChain(t, 1e-4, 100, tt.cfg...)
			q, err := solve(k, tt.seed, tt.target)
			require.Error(t, err)
			assert.True(t, errors.Is(err, tt.want), "got %v, want %v", err, tt.want)
			for i, v := range q {
				require.False(t, math32.IsNaN(v), "controls[%d] left untouched", i)
				assert.GreaterOrEqual(t, v, tt.cfg[i].Min)
				assert.LessOrEqual(t, v, tt.cfg[i].Max)
			}
			lim := newChain(t, 1e-4, 100, tt.cfg...)
			assert.Less(t, dist(fkPos(t, lim, q), tt.target), dist(fkPos(t, lim, tt.seed), tt.target), "best effort improves on the seed")
			if tt.check != nil {
				tt.check(t, q, tt.target)
			}
		})
	}
}

// Review K10: failure classification of blocked solves.
func TestBackward_BlockedClassification(t *testing.T) {
	cfg := hexleg()
	ref := newChain(t, 1e-4, 100, cfg...)
	t.Run("blocked local minimum is ErrNoConvergence", func(t *testing.T) {
		k := newChain(t, 1e-4, 100, cfg...)
		// Classification is stable for this target with and without -race
		// (FMA fusion differs); some nearby targets flip between classes.
		target := [3]float32{-0.07514365, -0.011285744, 0.1573218}
		q, err := solve(k, hexlegNominal, target)
		assert.True(t, errors.Is(err, kintypes.ErrNoConvergence), "got %v", err)
		assert.Less(t, dist(fkPos(t, ref, q), target), dist(fkPos(t, ref, hexlegNominal), target), "best effort improves on the seed")
	})
	t.Run("lifted solve finds an in-limit branch: solved", func(t *testing.T) {
		// The restarts from this seed end pinned; only the limit-free
		// classification descent reaches the in-limit knee-up branch.
		k := newChain(t, 1e-4, 100, cfg...)
		target := fkPos(t, ref, []float32{-0.3441907, -0.7327828, 2.6878688})
		q, err := solve(k, []float32{0.5, 1.2, -0.3}, target)
		require.NoError(t, err)
		assert.LessOrEqual(t, dist(fkPos(t, ref, q), target), float32(1e-4))
		for i, v := range q {
			assert.GreaterOrEqual(t, v, cfg[i].Min)
			assert.LessOrEqual(t, v, cfg[i].Max)
		}
	})
}

// Review K1: a residual that overflows float32 must not leave the previous
// call's answer in the best-effort buffer; best effort starts from this call's
// seed.
func TestBackward_OverflowResidualNoStaleAnswer(t *testing.T) {
	k := newChain(t, 1e-4, 100, hexleg()...)
	first := []float32{0.3, 0.4, -1.2}
	q, err := solve(k, hexlegNominal, fkPos(t, k, first))
	require.NoError(t, err)
	require.InDeltaSlice(t, first, q, 1e-3)

	q, err = solve(k, hexlegNominal, [3]float32{2e19, 0, 0})
	assert.True(t, errors.Is(err, kintypes.ErrUnreachable), "got %v", err)
	assert.Equal(t, hexlegNominal, q, "best effort is the seed (every residual is +Inf), not the previous answer")
}

// Review K2: targets inside the inner workspace hole (closer to the femur
// pivot than the folded leg reaches) are unreachable, as the planar solver
// classifies them; best effort is the folded leg (tibia on its limit).
func TestBackward_InnerHoleUnreachable(t *testing.T) {
	cfg := hexleg()
	for _, target := range [][3]float32{{0, 0, 0}, {0.039, 0, 0}} {
		k := newChain(t, 1e-4, 100, cfg...)
		q, err := solve(k, hexlegNominal, target)
		assert.True(t, errors.Is(err, kintypes.ErrUnreachable), "target %v: got %v", target, err)
		assert.InDelta(t, pi*0.9, math32.Abs(q[2]), 1e-5, "target %v: tibia folded onto its limit, q %v", target, q)
	}
}

// Review K10: the best-effort answer of a failed solve is the minimum-residual
// in-limit pose (checked against a dense grid over the joint box).
func TestBackward_BestEffortIsMinimumResidual(t *testing.T) {
	cfg := hexleg()
	k := newChain(t, 1e-4, 100, cfg...)
	ref := newChain(t, 1e-4, 100, cfg...)
	lo, hi := limits(cfg)
	// The solver is local: these targets have their global minimum in a basin
	// the restarts reach (inner-hole targets do not, see TestBackward_InnerHoleUnreachable).
	for _, target := range [][3]float32{{-0.1, 0, -0.05}, {-0.05, -0.08, 0.02}} {
		q, err := solve(k, hexlegNominal, target)
		require.Error(t, err, "target %v", target)
		got := dist(fkPos(t, ref, q), target)
		const n = 48
		best := math32.Inf(1)
		g := make([]float32, 3)
		for a := 0; a <= n; a++ {
			for b := 0; b <= n; b++ {
				for c := 0; c <= n; c++ {
					for i, s := range [3]int{a, b, c} {
						g[i] = lo[i] + (hi[i]-lo[i])*float32(s)/n
					}
					best = math32.Min(best, dist(fkPos(t, ref, g), target))
				}
			}
		}
		assert.LessOrEqual(t, got, best+1e-4, "target %v (%v): residual %.5g, grid minimum %.5g", target, err, got, best)
	}
}

// The solve path must not allocate (embedded targets).
func TestBackward_NoAllocations(t *testing.T) {
	k := newChain(t, 1e-4, 100, hexleg()...)
	target := fkPos(t, k, []float32{0.2, 0.5, -1.2})
	st := mat.New(3, 1)
	dst := mat.New(effectorSize, 1)
	dst[0][0], dst[1][0], dst[2][0], dst[6][0] = target[0], target[1], target[2], 1
	ctl := mat.New(3, 1)
	// Convert to the interface once: boxing a slice header at the call site allocates.
	var stI, dstI, ctlI mattype.Matrix = st, dst, ctl
	var err error
	allocs := testing.AllocsPerRun(100, func() {
		st[0][0], st[1][0], st[2][0] = hexlegNominal[0], hexlegNominal[1], hexlegNominal[2]
		err = k.Backward(stI, dstI, ctlI)
	})
	require.NoError(t, err)
	assert.Zero(t, allocs, "allocations per Backward")
}

// The analytic Jacobian (all four joint kinds) matches central differences of
// Forward.
func TestJacobian_MatchesFiniteDifferences(t *testing.T) {
	chains := map[string][]Config{
		"hexleg": hexleg(),
		"α and r joints": {
			{Min: -pi, Max: pi, R: 0.5, Alpha: pi / 2, Index: 0},
			{Min: -pi / 2, Max: pi / 2, Theta: pi / 4, R: 0.5, Index: 1},
			{Min: 0.1, Max: 0.6, Theta: pi / 3, Index: 2},
		},
		"revolute + prismatic": {
			{Min: -pi, Max: pi, R: 0.2, Alpha: pi / 2, Index: 0},
			{Min: 0, Max: 2, Theta: 0.3, Alpha: 0.4, Index: 3},
		},
	}
	const h = 1e-2
	for name, cfg := range chains {
		t.Run(name, func(t *testing.T) {
			k := newChain(t, 1e-4, 100, cfg...)
			ref := newChain(t, 1e-4, 100, cfg...)
			n := len(cfg)
			r := rand.New(rand.NewPCG(9, 0))
			lo, hi := limits(cfg)
			for i := range lo { // keep ±h inside the limits
				lo[i], hi[i] = lo[i]+2*h, hi[i]-2*h
			}
			for range 200 {
				q := sampleQ(r, lo, hi)
				copy(k.params, q)
				require.NoError(t, k.computeFrames())
				k.jacobian()
				for j := range n {
					qp := append([]float32(nil), q...)
					qm := append([]float32(nil), q...)
					qp[j] += h
					qm[j] -= h
					pp, pm := fkPos(t, ref, qp), fkPos(t, ref, qm)
					for row := range 3 {
						fd := (pp[row] - pm[row]) / (2 * h)
						assert.InDelta(t, fd, k.jac[row*n+j]/k.colScale(j), 2e-4, "q %v joint %d row %d", q, j, row)
					}
				}
			}
		})
	}
}

// Review probe D1: from the stretched (singular) zero pose the Jacobian is
// orthogonal to a radial error; the solver must escape instead of stalling.
func TestBackward_SingularSeed(t *testing.T) {
	k := newChain(t, 1e-4, 100, hexleg()...)
	target := [3]float32{0.999 * (0.039 + 0.045 + 0.100), 0, 0}
	q, err := solve(k, []float32{0, 0, 0}, target)
	require.NoError(t, err)
	assert.LessOrEqual(t, dist(fkPos(t, k, q), target), float32(1e-4))
}

// Review ER-017 #1: with the leg's full limits, FK-generated (hence in-limit
// reachable) targets — half of them knee-up — must never be reported as
// ErrJointLimit; from the knee-down nominal seed the solver has to find the
// other elbow branch when the femur pins.
func TestBackward_FullLimitsNoFalseJointLimit(t *testing.T) {
	cfg := hexleg()
	k := newChain(t, 1e-4, 100, cfg...)
	ref := newChain(t, 1e-4, 100, cfg...)
	lo, hi := limits(cfg)
	r := rand.New(rand.NewPCG(42, 0))
	const n = 2000
	ok, jointLimit := 0, 0
	for range n {
		qt := sampleQ(r, lo, hi)
		target := fkPos(t, ref, qt)
		q, err := solve(k, hexlegNominal, target)
		switch {
		case err == nil:
			require.LessOrEqual(t, dist(fkPos(t, ref, q), target), float32(1.01e-4))
			ok++
		case errors.Is(err, kintypes.ErrJointLimit):
			jointLimit++
		}
	}
	t.Logf("full-limit targets, nominal seed: %d/%d converged, %d ErrJointLimit", ok, n, jointLimit)
	assert.Zero(t, jointLimit, "in-limit reachable targets reported as ErrJointLimit")
	assert.GreaterOrEqual(t, ok, 1990)
}

// Review ER-017 #2: prismatic columns are unitless, revolute columns are
// lengths; the solver must behave the same whether the chain is declared in
// metres or millimetres.
func TestBackward_PrismaticUnitInvariance(t *testing.T) {
	gantry := func(s float32) []Config {
		return []Config{
			{Min: 0, Max: 0.5 * s, Alpha: -pi / 2, Index: 3},
			{Min: 0, Max: 0.5 * s, Theta: -pi / 2, Alpha: -pi / 2, Index: 3},
			{Min: 0, Max: 0.5 * s, Index: 3},
		}
	}
	scara := func(s float32) []Config {
		return []Config{
			{Min: -pi, Max: pi, R: 0.2 * s, Index: 0},
			{Min: -pi * 0.9, Max: pi * 0.9, R: 0.15 * s, Alpha: pi, Index: 0},
			{Min: 0, Max: 0.2 * s, Index: 3},
		}
	}
	rate := func(cfg []Config, s float32, iters int) int {
		k := newChain(t, 1e-4*s, iters, cfg...)
		r := rand.New(rand.NewPCG(5, 0))
		seed := make([]float32, len(cfg))
		for i, c := range cfg {
			seed[i] = (c.Min + c.Max) / 2
		}
		lo, hi := limits(cfg)
		ok := 0
		for range 300 {
			if _, err := solve(k, seed, fkPos(t, k, sampleQ(r, lo, hi))); err == nil {
				ok++
			}
		}
		return ok
	}
	for _, tt := range []struct {
		name string
		mk   func(float32) []Config
	}{{"gantry", gantry}, {"SCARA", scara}} {
		t.Run(tt.name, func(t *testing.T) {
			for _, iters := range []int{20, 100} {
				m, mm := rate(tt.mk(1), 1, iters), rate(tt.mk(1000), 1000, iters)
				t.Logf("%s, %d iterations: metres %d/300, millimetres %d/300", tt.name, iters, m, mm)
				assert.InDelta(t, m, mm, 3, "unit dependence at %d iterations", iters)
				if iters == 100 {
					assert.GreaterOrEqual(t, m, 297)
				}
			}
		})
	}
}

// Review ER-017 #3: a non-finite target or seed is rejected with
// kintypes.ErrNonFinite before solving; controls are left untouched (they used
// to receive the previous call's answer).
func TestBackward_NonFiniteInput(t *testing.T) {
	nan, inf := math32.NaN(), math32.Inf(1)
	k := newChain(t, 1e-4, 100, hexleg()...)
	good := fkPos(t, k, []float32{0.3, 0.4, -1.2})
	tests := []struct {
		name   string
		seed   []float32
		target [3]float32
	}{
		{"NaN target x", hexlegNominal, [3]float32{nan, 0, 0}},
		{"Inf target z", hexlegNominal, [3]float32{0.1, 0, -inf}},
		{"NaN seed", []float32{0, nan, -1.6}, good},
		{"Inf seed", []float32{inf, 0.3, -1.6}, good},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := solve(k, []float32{0.1, 0.2, -1.0}, good) // a previous, successful solve
			require.NoError(t, err)
			st := mat.New(3, 1)
			for i, v := range tt.seed {
				st[i][0] = v
			}
			dst := mat.New(effectorSize, 1)
			dst[0][0], dst[1][0], dst[2][0], dst[6][0] = tt.target[0], tt.target[1], tt.target[2], 1
			ctl := mat.New(3, 1)
			ctl[0][0], ctl[1][0], ctl[2][0] = 7, 8, 9
			err = k.Backward(st, dst, ctl)
			assert.True(t, errors.Is(err, kintypes.ErrNonFinite), "got %v", err)
			assert.Equal(t, []float32{7, 8, 9}, []float32{ctl[0][0], ctl[1][0], ctl[2][0]}, "controls untouched")
		})
	}
}
