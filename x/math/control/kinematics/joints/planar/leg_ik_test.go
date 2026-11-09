package planar

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

// legCfg is the cmd/manipulator hexapod leg (39/45/100 mm) in metres with the
// given joint limits.
func legCfg(coxa, femur, tibia [2]float32) [3]Config {
	return [3]Config{
		{Min: coxa[0], Max: coxa[1], Length: 0.039},
		{Min: femur[0], Max: femur[1], Length: 0.045},
		{Min: tibia[0], Max: tibia[1], Length: 0.100},
	}
}

var (
	probeRLimits = legCfg([2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, pi / 2}, [2]float32{-pi * 0.9, pi * 0.9})
	legLimits    = legCfg([2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, pi / 2}, [2]float32{-pi * 0.9, 0})
	fullLimits   = legCfg([2]float32{-pi, pi}, [2]float32{-pi, pi}, [2]float32{-pi, pi})
)

func newP3(t testing.TB, cfg [3]Config) *p3d {
	t.Helper()
	k, err := New3DOF(cfg)
	require.NoError(t, err)
	return k
}

func p3fk(t testing.TB, k *p3d, q [3]float32) [3]float32 {
	t.Helper()
	st := mat.New(3, 1)
	for i, v := range q {
		st[i][0] = v
	}
	dst := mat.New(planarEffectorSize, 1)
	require.NoError(t, k.Forward(st, dst, nil))
	return [3]float32{dst[0][0], dst[1][0], dst[2][0]}
}

// p3ik solves for `target`; controls start as NaN so untouched controls show.
func p3ik(k *p3d, target [3]float32) ([3]float32, error) {
	dst := mat.New(planarEffectorSize, 1)
	dst[0][0], dst[1][0], dst[2][0] = target[0], target[1], target[2]
	ctl := mat.New(3, 1)
	for i := range ctl {
		ctl[i][0] = math32.NaN()
	}
	err := k.Backward(nil, dst, ctl)
	return [3]float32{ctl[0][0], ctl[1][0], ctl[2][0]}, err
}

func dist3(a, b [3]float32) float32 {
	dx, dy, dz := a[0]-b[0], a[1]-b[1], a[2]-b[2]
	return math32.Sqrt(dx*dx + dy*dy + dz*dz)
}

// roundTrip checks FK(IK(FK(q_true))) == FK(q_true) within 1e-4 m for every
// sample and returns how many feet were behind the coxa yaw axis.
func roundTrip(t *testing.T, cfg [3]Config, lo, hi [3]float32, n int, seed uint64) (exact, behind int) {
	t.Helper()
	r := rand.New(rand.NewPCG(seed, 0))
	k := newP3(t, cfg)
	ref := newP3(t, cfg)
	for range n {
		var qt [3]float32
		for i := range qt {
			qt[i] = lo[i] + r.Float32()*(hi[i]-lo[i])
		}
		target := p3fk(t, ref, qt)
		if cfg[0].Length+cfg[1].Length*math32.Cos(qt[1])+cfg[2].Length*math32.Cos(qt[1]+qt[2]) < 0 {
			behind++
		}
		q, err := p3ik(k, target)
		if err != nil {
			t.Errorf("q_true %v target %v: unexpected %v", qt, target, err)
			continue
		}
		if d := dist3(p3fk(t, ref, q), target); d > 1e-4 {
			t.Errorf("q_true %v: FK(IK(t)) off by %.3g m (q %v)", qt, d, q)
			continue
		}
		exact++
	}
	return exact, behind
}

// T-IK-05: review probe R (coxa ±60°, femur −60…72°, tibia −149…−20°) must be
// solved exactly, including feet behind the coxa yaw axis.
func TestPlanar3DOF_ProbeRWorkspace(t *testing.T) {
	const n = 500
	exact, behind := roundTrip(t, probeRLimits, [3]float32{-pi / 3, -pi / 3, -2.6}, [3]float32{pi / 3, pi / 2.5, -0.35}, n, 5)
	t.Logf("probe R: %d/%d exact (%d feet behind the yaw axis)", exact, n, behind)
	assert.Equal(t, n, exact)
}

// T-IK-05: with ±π limits every knee-down pose is reachable; feet behind the yaw
// axis and femur angles beyond π (before wrapping) must round-trip exactly.
func TestPlanar3DOF_FullRangeExact(t *testing.T) {
	k := newP3(t, fullLimits)
	// Review minimal reproduction: foot 0.3 mm behind the axis.
	target := p3fk(t, k, [3]float32{0.839, -0.910, -3.069})
	q, err := p3ik(k, target)
	require.NoError(t, err)
	assert.LessOrEqual(t, dist3(p3fk(t, k, q), target), float32(1e-4), "q %v", q)

	const n = 2000
	exact, behind := roundTrip(t, fullLimits, [3]float32{-pi, -pi, -pi}, [3]float32{pi, pi, 0}, n, 11)
	t.Logf("±π limits: %d/%d exact (%d behind the yaw axis)", exact, n, behind)
	assert.Equal(t, n, exact)
}

// T-IK-05: unreachable targets and limit violations are typed errors with a
// best-effort (in-limit) solution written to controls.
func TestPlanar3DOF_TypedErrors(t *testing.T) {
	tibiaLimited := legCfg([2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, 0})
	ref := newP3(t, fullLimits)
	tests := []struct {
		name   string
		cfg    [3]Config
		target [3]float32
		want   error
		check  func(t *testing.T, k *p3d, q [3]float32)
	}{
		{
			name: "beyond reach (0.5,0,0)", cfg: legLimits, target: [3]float32{0.5, 0, 0}, want: kintypes.ErrUnreachable,
			check: func(t *testing.T, k *p3d, q [3]float32) {
				assert.Less(t, dist3(p3fk(t, k, q), [3]float32{0.184, 0, 0}), float32(1e-4), "best effort: stretched towards the target")
			},
		},
		{
			name: "inside the femur-tibia hole for both coxa directions (on the yaw axis)", cfg: legLimits, target: [3]float32{0, 0, 0.02}, want: kintypes.ErrUnreachable,
		},
		{
			name: "at the femur joint: only the reversed coxa (180°) reaches it", cfg: legLimits, target: [3]float32{0.039, 0, 0}, want: kintypes.ErrJointLimit,
		},
		{
			name: "at the femur joint, coxa ±180° allowed", cfg: fullLimits, target: [3]float32{0.039, 0, 0}, want: nil,
		},
		{
			name: "behind the body, coxa would need 180° (limit ±90°)", cfg: legLimits, target: [3]float32{-0.1, 0, -0.05}, want: kintypes.ErrJointLimit,
		},
		{
			name: "behind the body, coxa ±180° allowed", cfg: fullLimits, target: [3]float32{-0.1, 0, -0.05}, want: nil,
		},
		{
			name: "needs tibia -2.0, tibia limit [-pi/2,0]", cfg: tibiaLimited, target: p3fk(t, ref, [3]float32{0, 0.6, -2.0}), want: kintypes.ErrJointLimit,
			check: func(t *testing.T, _ *p3d, q [3]float32) {
				assert.InDelta(t, -pi/2, q[2], 1e-6, "tibia clamped to its limit")
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := newP3(t, tt.cfg)
			q, err := p3ik(k, tt.target)
			for i, v := range q {
				require.False(t, math32.IsNaN(v), "controls[%d] left untouched", i)
				assert.GreaterOrEqual(t, v, tt.cfg[i].Min)
				assert.LessOrEqual(t, v, tt.cfg[i].Max)
			}
			if tt.want == nil {
				require.NoError(t, err)
				assert.LessOrEqual(t, dist3(p3fk(t, k, q), tt.target), float32(1e-4))
				return
			}
			require.Error(t, err)
			assert.True(t, errors.Is(err, tt.want), "got %v, want %v", err, tt.want)
			if tt.check != nil {
				tt.check(t, k, q)
			}
		})
	}
}

// A NaN/Inf target is rejected with kintypes.ErrNonFinite; controls untouched.
func TestPlanar3DOF_NonFiniteTarget(t *testing.T) {
	k := newP3(t, legLimits)
	for _, target := range [][3]float32{{math32.NaN(), 0, 0}, {0.1, math32.Inf(-1), 0}, {0.1, 0, math32.NaN()}} {
		dst := mat.New(planarEffectorSize, 1)
		dst[0][0], dst[1][0], dst[2][0] = target[0], target[1], target[2]
		ctl := mat.New(3, 1)
		ctl[0][0], ctl[1][0], ctl[2][0] = 7, 8, 9
		err := k.Backward(nil, dst, ctl)
		assert.True(t, errors.Is(err, kintypes.ErrNonFinite), "target %v: got %v", target, err)
		assert.Equal(t, []float32{7, 8, 9}, []float32{ctl[0][0], ctl[1][0], ctl[2][0]}, "controls untouched")
	}
}

// Review K3: p2d rejects a NaN/Inf target with kintypes.ErrNonFinite and
// leaves controls untouched (it used to write NaN controls and return nil).
func TestPlanar2DOF_NonFiniteTarget(t *testing.T) {
	k := newP2(t, [2]Config{{Min: -pi, Max: pi, Length: 0.05}, {Min: -pi, Max: pi, Length: 0.1}})
	for _, target := range [][3]float32{{math32.NaN(), 0, 0}, {0.1, math32.Inf(1), 0}, {0.1, 0, math32.Inf(-1)}} {
		dst := mat.New(planarEffectorSize, 1)
		dst[0][0], dst[1][0], dst[2][0] = target[0], target[1], target[2]
		ctl := mat.New(2, 1)
		ctl[0][0], ctl[1][0] = 7, 8
		err := k.Backward(nil, dst, ctl)
		assert.True(t, errors.Is(err, kintypes.ErrNonFinite), "target %v: got %v", target, err)
		assert.Equal(t, []float32{7, 8}, []float32{ctl[0][0], ctl[1][0]}, "target %v: controls untouched", target)
	}
}

// Review K7: knee-up solutions are found when only knee-up is in the limits,
// and when both branches are in the limits knee-down comes first (the walker
// relies on it).
func TestPlanar3DOF_KneeBranches(t *testing.T) {
	kneeUp := legCfg([2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, pi / 2}, [2]float32{0, pi * 0.9})
	const n = 500
	exact, _ := roundTrip(t, kneeUp, [3]float32{-pi / 2, -pi / 2, 0.2}, [3]float32{pi / 2, pi / 2, pi * 0.85}, n, 21)
	assert.Equal(t, n, exact, "knee-up-only limits round-trip")

	k := newP3(t, probeRLimits)
	for _, qt := range [][3]float32{{0.2, -0.3, 1.2}, {0.3, -0.8, 1.0}, {0, 0.1, 0.6}} {
		target := p3fk(t, k, qt)
		q, err := p3ik(k, target)
		require.NoError(t, err, "q_true %v", qt)
		assert.LessOrEqual(t, dist3(p3fk(t, k, q), target), float32(1e-4), "q_true %v", qt)
		assert.InDelta(t, -qt[2], q[2], 1e-3, "q_true %v (knee up): knee-down mirror returned first, got %v", qt, q)
	}
}

// gridMin is the smallest |FK(q) − target| over a dense grid of in-limit q.
func gridMin(t *testing.T, k *p3d, cfg [3]Config, target [3]float32) float32 {
	t.Helper()
	const n = 60
	best := math32.Inf(1)
	for a := 0; a <= n; a++ {
		for b := 0; b <= n; b++ {
			for c := 0; c <= n; c++ {
				var q [3]float32
				for i, s := range [3]int{a, b, c} {
					q[i] = cfg[i].Min + (cfg[i].Max-cfg[i].Min)*float32(s)/n
				}
				best = math32.Min(best, dist3(p3fk(t, k, q), target))
			}
		}
	}
	return best
}

// Review K8: best effort on the error paths. Unreachable off-axis targets get
// the leg stretched from the femur pivot straight at the target; ErrJointLimit
// best effort is no farther than any in-limit pose (dense grid).
func TestPlanar3DOF_BestEffortQuality(t *testing.T) {
	t.Run("unreachable off-axis: stretched towards the target", func(t *testing.T) {
		k := newP3(t, legLimits)
		for _, target := range [][3]float32{{0.2, 0.25, 0.1}, {0.1, -0.3, -0.2}} {
			q, err := p3ik(k, target)
			require.True(t, errors.Is(err, kintypes.ErrUnreachable), "target %v: got %v", target, err)
			yaw := math32.Atan2(target[1], target[0])
			pivot := [3]float32{0.039 * math32.Cos(yaw), 0.039 * math32.Sin(yaw), 0}
			d := dist3(target, pivot)
			var want [3]float32
			for i := range want {
				want[i] = pivot[i] + (target[i]-pivot[i])*0.145/d
			}
			assert.Less(t, dist3(p3fk(t, k, q), want), float32(1e-4), "target %v: foot %v, want %v", target, p3fk(t, k, q), want)
		}
	})
	t.Run("joint limit: residual is the in-limit minimum", func(t *testing.T) {
		tibiaLimited := legCfg([2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, pi / 2}, [2]float32{-pi / 2, 0})
		ref := newP3(t, fullLimits)
		for _, tc := range []struct {
			cfg    [3]Config
			target [3]float32
		}{
			{tibiaLimited, p3fk(t, ref, [3]float32{0.3, 0.6, -2.0})},
			{tibiaLimited, p3fk(t, ref, [3]float32{-0.5, -0.2, -2.4})},
			{legLimits, [3]float32{-0.1, 0.02, -0.05}},
		} {
			k := newP3(t, tc.cfg)
			q, err := p3ik(k, tc.target)
			require.True(t, errors.Is(err, kintypes.ErrJointLimit), "target %v: got %v", tc.target, err)
			got := dist3(p3fk(t, k, q), tc.target)
			assert.LessOrEqual(t, got, gridMin(t, k, tc.cfg, tc.target)+1e-4, "target %v: q %v", tc.target, q)
		}
	})
}

// Review K9: boundary poses round-trip exactly: full stretch, full fold and
// joints exactly at a limit.
func TestPlanar3DOF_BoundaryRoundTrips(t *testing.T) {
	tests := []struct {
		name string
		cfg  [3]Config
		q    [3]float32
	}{
		{"full stretch, tibia at its 0 limit", legLimits, [3]float32{0.3, 0.2, 0}},
		{"full stretch along +x", legLimits, [3]float32{0, 0, 0}},
		{"full fold, tibia -pi", fullLimits, [3]float32{0.2, 0.3, -pi}},
		{"tibia at its -0.9pi limit", legLimits, [3]float32{-0.4, 0.5, -pi * 0.9}},
		{"coxa and femur at limits", legLimits, [3]float32{-pi / 2, pi / 2, -1}},
		{"femur at its lower limit", legLimits, [3]float32{pi / 2, -pi / 2, -0.7}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := newP3(t, tt.cfg)
			target := p3fk(t, k, tt.q)
			q, err := p3ik(k, target)
			require.NoError(t, err, "q %v", q)
			assert.LessOrEqual(t, dist3(p3fk(t, k, q), target), float32(1e-4), "q %v", q)
		})
	}
}

// Backward allocates nothing on the exact, ErrJointLimit and ErrUnreachable
// paths.
func TestPlanar3DOF_BackwardNoAllocations(t *testing.T) {
	k := newP3(t, legLimits)
	dst := mat.New(planarEffectorSize, 1)
	ctl := mat.New(3, 1)
	var dstI, ctlI mattype.Matrix = dst, ctl // box once: boxing a slice header allocates
	for _, target := range [][3]float32{{0.12, 0.03, -0.05}, {-0.1, 0.02, -0.05}, {0.5, 0, 0}} {
		dst[0][0], dst[1][0], dst[2][0] = target[0], target[1], target[2]
		allocs := testing.AllocsPerRun(100, func() { _ = k.Backward(nil, dstI, ctlI) })
		assert.Zero(t, allocs, "target %v", target)
	}
}
