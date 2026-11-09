package chain

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHexapodLayout(t *testing.T) {
	b := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	require.NoError(t, b.Validate())
	require.Len(t, b.Mounts, 6)
	require.Len(t, b.Neutral, 6)

	// Leg order: left front→back, then right front→back (scheduler.HexL1…HexR3).
	hips := []vec.Vector3D{{0.10, 0.06, 0}, {0, 0.06, 0}, {-0.10, 0.06, 0}, {0.10, -0.06, 0}, {0, -0.06, 0}, {-0.10, -0.06, 0}}
	lim := DefaultLegLimits()
	for i, m := range b.Mounts {
		assert.Equal(t, hips[i], m.Hip, "leg %d hip", i)
		assert.InDelta(t, math.Atan2(float64(hips[i][1]), float64(hips[i][0])), float64(m.Yaw), 1e-6, "leg %d radial yaw", i)
		require.Len(t, m.Leg.Joints, 3)
		reach := probeRGeometry.Coxa + probeRGeometry.Femur
		want := vec.Vector3D{
			m.Hip[0] + reach*math32.Cos(m.Yaw),
			m.Hip[1] + reach*math32.Sin(m.Yaw),
			-probeRGeometry.Tibia,
		}
		for k := range 3 {
			assert.InDelta(t, want[k], b.Neutral[i][k], 1e-6, "leg %d neutral[%d]", i, k)
		}
		left := i < 3
		for j, jt := range m.Leg.Joints {
			assert.Equal(t, 3*i+j, jt.Servo.Channel)
			assert.Equal(t, lim[j][0], jt.Min)
			assert.Equal(t, lim[j][1], jt.Max)
			assert.Equal(t, DefaultJointMaxVel, jt.MaxVel)
			wantSign := int8(1)
			if left && j > 0 {
				wantSign = -1
			}
			assert.Equal(t, wantSign, jt.Servo.Sign, "leg %d joint %d sign", i, j)
			// The limit range is centred on the servo travel.
			lo := jt.Servo.ZeroDeg + float32(jt.Servo.Sign)*jt.Min*180/math32.Pi
			hi := jt.Servo.ZeroDeg + float32(jt.Servo.Sign)*jt.Max*180/math32.Pi
			assert.InDelta(t, 180, lo+hi, 1e-3, "leg %d joint %d centred", i, j)
		}
	}
	// The neutral foot is the pose (0, 0, −90°) in every hip frame.
	for i := range b.Mounts {
		p := b.FootInLeg(i, Pose{Z: probeRGeometry.Tibia}, vec.Vector3D{b.Neutral[i][0], b.Neutral[i][1], 0})
		assert.InDelta(t, 0.084, p[0], 1e-6)
		assert.InDelta(t, 0, p[1], 1e-6)
		assert.InDelta(t, -0.1, p[2], 1e-6)
	}
}

func TestQuadrupedLayout(t *testing.T) {
	b := Quadruped(0.12, 0.05, probeRGeometry)
	require.NoError(t, b.Validate())
	require.Len(t, b.Mounts, 4)
	// QuadLF, QuadLH, QuadRF, QuadRH.
	hips := []vec.Vector3D{{0.12, 0.05, 0}, {-0.12, 0.05, 0}, {0.12, -0.05, 0}, {-0.12, -0.05, 0}}
	for i, m := range b.Mounts {
		assert.Equal(t, hips[i], m.Hip)
		assert.InDelta(t, math.Atan2(float64(hips[i][1]), float64(hips[i][0])), float64(m.Yaw), 1e-6)
		assert.Equal(t, 3*i, m.Leg.Joints[0].Servo.Channel)
	}
	assert.Equal(t, int8(-1), b.Mounts[1].Leg.Joints[1].Servo.Sign)
	assert.Equal(t, int8(1), b.Mounts[3].Leg.Joints[1].Servo.Sign)
}

func TestBodyValidate(t *testing.T) {
	b := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	short := Body{Mounts: b.Mounts, Neutral: b.Neutral[:5]}
	assert.ErrorIs(t, short.Validate(), kintypes.ErrInvalidConfig)
	mounts := append([]Mount(nil), b.Mounts...)
	mounts[2].Leg.Joints = append([]Joint(nil), mounts[2].Leg.Joints...)
	mounts[2].Leg.Joints[1].Servo.Sign = 0
	assert.ErrorIs(t, Body{Mounts: mounts, Neutral: b.Neutral}.Validate(), ErrServoMap)
	assert.ErrorIs(t, Body{}.Validate(), kintypes.ErrInvalidConfig)
}

// rot64 is R = Rz(yaw)·Ry(pitch)·Rx(roll), computed independently in float64.
func rot64(roll, pitch, yaw float64) [3][3]float64 {
	cr, sr := math.Cos(roll), math.Sin(roll)
	cp, sp := math.Cos(pitch), math.Sin(pitch)
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	rz := [3][3]float64{{cy, -sy, 0}, {sy, cy, 0}, {0, 0, 1}}
	ry := [3][3]float64{{cp, 0, sp}, {0, 1, 0}, {-sp, 0, cp}}
	rx := [3][3]float64{{1, 0, 0}, {0, cr, -sr}, {0, sr, cr}}
	return mul3(mul3(rz, ry), rx)
}

func mul3(a, b [3][3]float64) (c [3][3]float64) {
	for i := range 3 {
		for j := range 3 {
			for k := range 3 {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return c
}

func apply(r [3][3]float64, p [3]float64) (q [3]float64) {
	for i := range 3 {
		for k := range 3 {
			q[i] += r[i][k] * p[k]
		}
	}
	return q
}

// hipToWorld64 is the forward transform world ← body ← hip in float64.
func hipToWorld64(m Mount, pose Pose, p vec.Vector3D) [3]float64 {
	hip := apply(rot64(0, 0, float64(m.Yaw)), [3]float64{float64(p[0]), float64(p[1]), float64(p[2])})
	body := [3]float64{hip[0] + float64(m.Hip[0]), hip[1] + float64(m.Hip[1]), hip[2] + float64(m.Hip[2])}
	w := apply(rot64(float64(pose.Roll), float64(pose.Pitch), float64(pose.Yaw)), body)
	return [3]float64{w[0] + float64(pose.X), w[1] + float64(pose.Y), w[2] + float64(pose.Z)}
}

func randomPose(r *rand.Rand) Pose {
	u := func(a float32) float32 { return (2*r.Float32() - 1) * a }
	return Pose{X: u(2), Y: u(2), Z: 0.05 + r.Float32()*0.1, Roll: u(0.3), Pitch: u(0.3), Yaw: u(math32.Pi)}
}

// T-IK-10: FootInLeg is the inverse of the forward body→hip→world transform.
func TestFootInLegInvertsForwardTransform(t *testing.T) {
	b := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	r := rand.New(rand.NewPCG(7, 8))
	for range 2000 {
		pose := randomPose(r)
		i := r.IntN(len(b.Mounts))
		p := vec.Vector3D{0.05 + 0.1*r.Float32(), 0.1 * (r.Float32() - 0.5), -0.15 * r.Float32()}
		w64 := hipToWorld64(b.Mounts[i], pose, p)
		world := vec.Vector3D{float32(w64[0]), float32(w64[1]), float32(w64[2])}

		got := b.FootInLeg(i, pose, world)
		for k := range 3 {
			require.InDelta(t, p[k], got[k], 2e-6, "FootInLeg leg %d axis %d", i, k)
		}
		fw := b.FootInWorld(i, pose, p)
		for k := range 3 {
			require.InDelta(t, world[k], fw[k], 2e-6, "FootInWorld leg %d axis %d", i, k)
		}
	}

	// Known value: level body at the origin, right middle leg (yaw −90°).
	got := b.FootInLeg(4, Pose{Z: 0.1}, vec.Vector3D{0, -0.2, 0})
	assert.InDelta(t, 0.14, got[0], 1e-6)
	assert.InDelta(t, 0, got[1], 1e-6)
	assert.InDelta(t, -0.1, got[2], 1e-6)
	// Body yawed +90° and moved: the left front leg's hip sits at (−0.06, 0.1)+(1, 2).
	got = b.FootInLeg(0, Pose{X: 1, Y: 2, Yaw: math32.Pi / 2}, vec.Vector3D{1 - 0.06, 2 + 0.1, 0})
	for k := range 3 {
		assert.InDelta(t, 0, got[k], 1e-6)
	}
}

func legFK(c Chain, q []float32) vec.Vector3D {
	l0, l1, l2 := c.Joints[0].A, c.Joints[1].A, c.Joints[2].A
	x := l0 + l1*math32.Cos(q[1]) + l2*math32.Cos(q[1]+q[2])
	z := l1*math32.Sin(q[1]) + l2*math32.Sin(q[1]+q[2])
	return vec.Vector3D{x * math32.Cos(q[0]), x * math32.Sin(q[0]), z}
}

// T-IK-10: SolveLegs reproduces known joint angles from world foot targets.
func TestSolveLegsReproducesJointAngles(t *testing.T) {
	b := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	s, err := b.NewLegSolver()
	require.NoError(t, err)
	n := len(b.Mounts)
	feet := make([]vec.Vector3D, n)
	q := make([][]float32, n)
	want := make([][]float32, n)
	errs := make([]error, n)
	r := rand.New(rand.NewPCG(9, 10))
	for range 300 {
		pose := randomPose(r)
		for i, m := range b.Mounts {
			want[i] = randomQ(r, m.Leg)
			feet[i] = b.FootInWorld(i, pose, legFK(m.Leg, want[i]))
			q[i] = make([]float32, 3)
		}
		require.NoError(t, s.SolveLegs(pose, feet, q, errs))
		for i := range n {
			require.NoError(t, errs[i])
			// The solution reproduces the foot position…
			got := b.FootInWorld(i, pose, legFK(b.Mounts[i].Leg, q[i]))
			for k := range 3 {
				require.InDelta(t, feet[i][k], got[k], 2e-5, "leg %d axis %d", i, k)
			}
			// …and the joint angles wherever the coxa angle is well conditioned
			// (foot ≥ 2 cm from the coxa axis; float32 world coordinates).
			if p := legFK(b.Mounts[i].Leg, want[i]); math32.Hypot(p[0], p[1]) < 0.02 {
				continue
			}
			for j := range 3 {
				require.InDelta(t, want[i][j], q[i][j], 5e-4, "leg %d joint %d", i, j)
			}
		}
	}

	// One unreachable foot: only that leg reports, with a typed error.
	pose := Pose{Z: 0.1}
	for i := range n {
		feet[i] = b.FootInWorld(i, pose, legFK(b.Mounts[i].Leg, []float32{0, 0, -math32.Pi / 2}))
	}
	feet[4] = b.FootInWorld(4, pose, vec.Vector3D{0.5, 0, -0.1})
	err = s.SolveLegs(pose, feet, q, errs)
	assert.ErrorIs(t, err, kintypes.ErrUnreachable)
	for i := range n {
		if i == 4 {
			assert.ErrorIs(t, errs[i], kintypes.ErrUnreachable)
			continue
		}
		assert.NoError(t, errs[i], "leg %d", i)
		assert.InDelta(t, -math32.Pi/2, q[i][2], 1e-4)
	}

	assert.ErrorIs(t, s.SolveLegs(pose, feet[:5], q, errs), kintypes.ErrInvalidDimensions)
	assert.ErrorIs(t, s.SolveLegs(pose, feet, q, errs[:2]), kintypes.ErrInvalidDimensions)

	for i := range n {
		feet[i] = b.FootInWorld(i, pose, legFK(b.Mounts[i].Leg, []float32{0.1, 0.2, -1.2}))
	}
	allocs := testing.AllocsPerRun(100, func() { _ = s.SolveLegs(pose, feet, q, errs) })
	assert.Zero(t, allocs, "SolveLegs allocations")
}

// A leg whose shape is not coxa–femur–tibia still solves through DH-DLS.
func TestSolveLegsDHFallback(t *testing.T) {
	b := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	mounts := append([]Mount(nil), b.Mounts...)
	leg := mounts[1].Leg
	leg.Joints = append([]Joint(nil), leg.Joints...)
	leg.Joints[2].D = 0.001 // knee offset: no longer planar
	mounts[1].Leg = leg
	b = Body{Mounts: mounts, Neutral: b.Neutral}
	_, err := leg.Planar3()
	require.ErrorIs(t, err, kintypes.ErrUnsupportedOperation)

	s, err := b.NewLegSolver()
	require.NoError(t, err)
	pose := Pose{Z: 0.1, Yaw: 0.3}
	feet := make([]vec.Vector3D, 6)
	q := make([][]float32, 6)
	errs := make([]error, 6)
	for i := range feet {
		feet[i] = b.FootInWorld(i, pose, legFK(b.Mounts[i].Leg, []float32{0, 0, -math32.Pi / 2}))
		q[i] = []float32{0, 0.3, -1.6} // seed
	}
	// Leg 1's foot is FK of its offset chain (the y offset moves the foot by 1 mm).
	feet[1] = b.FootInWorld(1, pose, vec.Vector3D{0.084, -0.001, -0.1})
	require.NoError(t, s.SolveLegs(pose, feet, q, errs))
	assert.InDelta(t, 0, q[1][0], 2e-3)
	assert.InDelta(t, 0, q[1][1], 2e-3)
	assert.InDelta(t, -math32.Pi/2, q[1][2], 2e-3)
}
