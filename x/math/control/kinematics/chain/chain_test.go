package chain

import (
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/joints/dh"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/joints/planar"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// probeR is the IK probe R hexapod leg: 39/45/100 mm, coxa ±60°, femur
// −60…72°, tibia −149…−20° (knee down only).
var (
	probeRGeometry = LegGeometry{Coxa: 0.039, Femur: 0.045, Tibia: 0.100}
	probeRLimits   = [3][2]float32{{Deg(-60), Deg(60)}, {Deg(-60), Deg(72)}, {Deg(-149), Deg(-20)}}
)

// servos maps the probe-R limits to the middle of a 0…180° servo travel.
func servos(sign int8) [3]ServoMap {
	var s [3]ServoMap
	for j := range s {
		mid := (probeRLimits[j][0] + probeRLimits[j][1]) / 2
		s[j] = ServoMap{Channel: j, ZeroDeg: 90 - float32(sign)*mid*180/math32.Pi, Sign: sign, Ratio: 1}
	}
	return s
}

func hexleg(sign int8) Chain {
	return CoxaFemurTibia("leg", probeRGeometry, probeRLimits, servos(sign))
}

func randomQ(r *rand.Rand, c Chain) []float32 {
	q := make([]float32, len(c.Joints))
	for j, jt := range c.Joints {
		q[j] = jt.Min + r.Float32()*(jt.Max-jt.Min)
	}
	return q
}

func forward(t testing.TB, m kintypes.Bidirectional, rows int, q []float32) [3]float32 {
	t.Helper()
	st := mat.New(len(q), 1)
	for i, v := range q {
		st[i][0] = v
	}
	dst := mat.New(rows, 1)
	require.NoError(t, m.Forward(st, dst, nil))
	return [3]float32{dst[0][0], dst[1][0], dst[2][0]}
}

func TestCoxaFemurTibiaShape(t *testing.T) {
	c := hexleg(1)
	require.Len(t, c.Joints, 3)
	assert.Equal(t, "leg", c.Name)
	assert.InDelta(t, math32.Pi/2, c.Joints[0].Alpha, 1e-7)
	for j, want := range [3]float32{0.039, 0.045, 0.100} {
		assert.Equal(t, Revolute, c.Joints[j].Kind)
		assert.Equal(t, want, c.Joints[j].A)
		assert.Equal(t, probeRLimits[j][0], c.Joints[j].Min)
		assert.Equal(t, probeRLimits[j][1], c.Joints[j].Max)
		assert.Equal(t, j, c.Joints[j].Servo.Channel)
	}
	assert.Zero(t, c.Joints[1].Alpha)
	assert.Zero(t, c.Joints[2].Alpha)
	require.NoError(t, c.Validate())
}

func TestDeg(t *testing.T) {
	assert.InDelta(t, math32.Pi, Deg(180), 1e-6)
	assert.InDelta(t, -math32.Pi/3, Deg(-60), 1e-6)
}

// The DH table and the planar configuration of the same hexleg chain
// describe the same mechanism.
func TestHexlegDHMatchesPlanar3(t *testing.T) {
	c := hexleg(1)
	dhCfg, err := c.DH()
	require.NoError(t, err)
	require.Len(t, dhCfg, 3)
	p3Cfg, err := c.Planar3()
	require.NoError(t, err)
	assert.Equal(t, [3]planar.Config{
		{Min: probeRLimits[0][0], Max: probeRLimits[0][1], Length: 0.039},
		{Min: probeRLimits[1][0], Max: probeRLimits[1][1], Length: 0.045},
		{Min: probeRLimits[2][0], Max: probeRLimits[2][1], Length: 0.100},
	}, p3Cfg)

	dhm, err := dh.New(1e-4, 100, dhCfg...)
	require.NoError(t, err)
	pm, err := planar.New3DOF(p3Cfg)
	require.NoError(t, err)

	r := rand.New(rand.NewPCG(1, 2))
	worst := float32(0)
	for range 2000 {
		q := randomQ(r, c)
		a := forward(t, dhm, 7, q)
		b := forward(t, pm, 6, q)
		for k := range 3 {
			worst = math32.Max(worst, math32.Abs(a[k]-b[k]))
		}
	}
	assert.Less(t, worst, float32(1e-6), "DH FK vs planar FK, worst |Δ| (m)")

	// Known pose: femur horizontal, tibia straight down.
	known := forward(t, dhm, 7, []float32{0, 0, -math32.Pi / 2})
	assert.InDelta(t, 0.084, known[0], 1e-6)
	assert.InDelta(t, 0, known[1], 1e-6)
	assert.InDelta(t, -0.1, known[2], 1e-6)
}

func TestPlanar3RejectsOtherShapes(t *testing.T) {
	base := hexleg(1)
	mods := map[string]func(c *Chain){
		"two joints":      func(c *Chain) { c.Joints = c.Joints[:2] },
		"prismatic femur": func(c *Chain) { c.Joints[1].Kind = Prismatic },
		"coxa alpha 0":    func(c *Chain) { c.Joints[0].Alpha = 0 },
		"femur alpha":     func(c *Chain) { c.Joints[1].Alpha = 0.1 },
		"tibia offset d":  func(c *Chain) { c.Joints[2].D = 0.01 },
		"theta offset":    func(c *Chain) { c.Joints[1].Offset = 0.2 },
	}
	for name, mod := range mods {
		t.Run(name, func(t *testing.T) {
			c := Chain{Name: base.Name, Joints: append([]Joint(nil), base.Joints...)}
			mod(&c)
			_, err := c.Planar3()
			assert.ErrorIs(t, err, kintypes.ErrUnsupportedOperation)
		})
	}
}

func TestModelPicksSolver(t *testing.T) {
	leg := hexleg(1)
	m, err := leg.Model()
	require.NoError(t, err)
	_, isDH := m.(*dh.DenavitHartenberg)
	assert.False(t, isDH, "a coxa–femur–tibia chain must get the analytic solver")

	// The analytic model solves FK(q) back to q exactly.
	r := rand.New(rand.NewPCG(3, 4))
	for range 200 {
		q := randomQ(r, leg)
		p := forward(t, m, 6, q)
		dst := mat.New(6, 1)
		dst[0][0], dst[1][0], dst[2][0] = p[0], p[1], p[2]
		ctl := mat.New(3, 1)
		require.NoError(t, m.Backward(nil, dst, ctl))
		for j := range 3 {
			assert.InDelta(t, q[j], ctl[j][0], 2e-4)
		}
	}

	arm := Chain{Name: "arm", Joints: []Joint{
		{Name: "a", A: 0.1, Min: -2, Max: 2, Servo: ServoMap{ZeroDeg: 90, Sign: 1, Ratio: 0.7}},
		{Name: "b", A: 0.1, Min: -2, Max: 2, Servo: ServoMap{ZeroDeg: 90, Sign: -1, Ratio: 0.7}},
	}}
	m, err = arm.Model(WithTolerance(1e-5), WithMaxIterations(50))
	require.NoError(t, err)
	_, isDH = m.(*dh.DenavitHartenberg)
	assert.True(t, isDH, "a 2-link arm gets the DH damped least-squares solver")

	bad := hexleg(1)
	bad.Joints = append([]Joint(nil), bad.Joints...)
	bad.Joints[0].Max = bad.Joints[0].Min
	_, err = bad.Model()
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig)
	_, err = bad.DH()
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig)
}

func TestServoMapping(t *testing.T) {
	c := Chain{Name: "m", Joints: []Joint{
		{A: 0.05, Min: Deg(-45), Max: Deg(45), Servo: ServoMap{ZeroDeg: 90, Sign: 1, Ratio: 1}},
		{A: 0.05, Min: Deg(-45), Max: Deg(45), Servo: ServoMap{ZeroDeg: 90, Sign: -1, Ratio: 1}},
		{A: 0.05, Min: Deg(-30), Max: Deg(30), Servo: ServoMap{ZeroDeg: 100, Sign: -1, Ratio: 2}},
	}}
	require.NoError(t, c.Validate())
	q := []float32{Deg(30), Deg(30), Deg(10)}
	deg := make([]float32, 3)
	require.NoError(t, c.ToServo(q, deg))
	assert.InDelta(t, 120, deg[0], 1e-4) // 90 + 30
	assert.InDelta(t, 60, deg[1], 1e-4)  // 90 − 30, mirrored
	assert.InDelta(t, 80, deg[2], 1e-4)  // 100 − 2·10

	back := make([]float32, 3)
	require.NoError(t, c.FromServo(deg, back))
	for j := range q {
		assert.InDelta(t, q[j], back[j], 1e-6)
	}

	// Out of the joint limit: clamped output, typed error.
	require.ErrorIs(t, c.ToServo([]float32{Deg(50), 0, 0}, deg), kintypes.ErrJointLimit)
	assert.InDelta(t, 135, deg[0], 1e-4)
	require.ErrorIs(t, c.FromServo([]float32{140, 90, 100}, back), kintypes.ErrJointLimit)
	assert.InDelta(t, Deg(50), back[0], 1e-6, "FromServo reports the reading, unclamped")

	assert.ErrorIs(t, c.ToServo(q[:2], deg), kintypes.ErrInvalidDimensions)
	assert.ErrorIs(t, c.FromServo(deg, back[:2]), kintypes.ErrInvalidDimensions)
}

// Radians → servo degrees → radians round trip, including mirrored
// (Sign −1) legs, without allocation.
func TestServoRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, sign := range []int8{1, -1} {
		c := hexleg(sign)
		require.NoError(t, c.Validate())
		deg := make([]float32, 3)
		back := make([]float32, 3)
		for range 1000 {
			q := randomQ(r, c)
			require.NoError(t, c.ToServo(q, deg))
			for j := range deg {
				s := c.Joints[j].Servo
				want := s.ZeroDeg + float32(s.Sign)*s.Ratio*q[j]*180/math32.Pi
				require.InDelta(t, want, deg[j], 1e-4)
				require.True(t, deg[j] >= 0 && deg[j] <= 180, "servo degrees %v within 0…180", deg[j])
			}
			require.NoError(t, c.FromServo(deg, back))
			for j := range q {
				require.InDelta(t, q[j], back[j], 2e-6)
			}
		}
		q := []float32{0.1, 0.2, -1}
		allocs := testing.AllocsPerRun(100, func() {
			_ = c.ToServo(q, deg)
			_ = c.FromServo(deg, back)
		})
		assert.Zero(t, allocs, "ToServo/FromServo allocations")
	}
}

// Validate catches bad limits, a zero Sign and a servo range that does
// not cover the joint limits.
func TestValidate(t *testing.T) {
	cases := []struct {
		name string
		mod  func(c *Chain)
		want error
	}{
		{"ok", func(*Chain) {}, nil},
		{"empty", func(c *Chain) { c.Joints = nil }, kintypes.ErrInvalidConfig},
		{"min == max", func(c *Chain) { c.Joints[1].Max = c.Joints[1].Min }, ErrLimits},
		{"min > max", func(c *Chain) { c.Joints[2].Min, c.Joints[2].Max = 0, -1 }, ErrLimits},
		{"zero limits", func(c *Chain) { c.Joints[0].Min, c.Joints[0].Max = 0, 0 }, ErrLimits},
		{"NaN limit", func(c *Chain) { c.Joints[0].Max = math32.NaN() }, ErrLimits},
		{"NaN length", func(c *Chain) { c.Joints[1].A = math32.NaN() }, kintypes.ErrInvalidConfig},
		{"Inf offset", func(c *Chain) { c.Joints[1].Offset = math32.Inf(1) }, kintypes.ErrInvalidConfig},
		{"negative MaxVel", func(c *Chain) { c.Joints[1].MaxVel = -1 }, kintypes.ErrInvalidConfig},
		{"bad kind", func(c *Chain) { c.Joints[1].Kind = 7 }, kintypes.ErrInvalidConfig},
		{"sign 0", func(c *Chain) { c.Joints[1].Servo.Sign = 0 }, ErrServoMap},
		{"sign 2", func(c *Chain) { c.Joints[1].Servo.Sign = 2 }, ErrServoMap},
		{"ratio 0", func(c *Chain) { c.Joints[2].Servo.Ratio = 0 }, ErrServoMap},
		{"NaN zero", func(c *Chain) { c.Joints[2].Servo.ZeroDeg = math32.NaN() }, ErrServoMap},
		{"range below 0°", func(c *Chain) { c.Joints[2].Servo.ZeroDeg = 90 }, ErrServoRange},
		{"ratio stretches range", func(c *Chain) { c.Joints[0].Servo.Ratio = 1.6 }, ErrServoRange},
		{"custom travel too short", func(c *Chain) { c.Joints[0].Servo.MinDeg, c.Joints[0].Servo.MaxDeg = 40, 140 }, ErrServoRange},
		{"custom travel ok", func(c *Chain) { c.Joints[0].Servo.MinDeg, c.Joints[0].Servo.MaxDeg = 30, 150 }, nil},
		{"custom travel inverted", func(c *Chain) { c.Joints[0].Servo.MinDeg, c.Joints[0].Servo.MaxDeg = 150, 30 }, ErrServoMap},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := hexleg(-1)
			c := Chain{Name: base.Name, Joints: append([]Joint(nil), base.Joints...)}
			tc.mod(&c)
			err := c.Validate()
			if tc.want == nil {
				assert.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.ErrorIs(t, err, tc.want)
			assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "every Validate error wraps ErrInvalidConfig")
		})
	}
}

func TestCheckZeroPose(t *testing.T) {
	c := hexleg(1)
	// q = 0 is outside the tibia limits here; CheckZeroPose uses unclamped FK.
	require.NoError(t, c.CheckZeroPose(vec.Vector3D{0.184, 0, 0}, 1e-4))
	err := c.CheckZeroPose(vec.Vector3D{0.184, 0, 0.01}, 1e-3)
	assert.ErrorIs(t, err, ErrZeroPose)
	err = c.CheckZeroPose(vec.Vector3D{-0.184, 0, 0}, 1e-3)
	assert.ErrorIs(t, err, ErrZeroPose)

	// A coxa mounted 90° off: FK(0) points along +y.
	turned := Chain{Name: "turned", Joints: append([]Joint(nil), c.Joints...)}
	turned.Joints[0].Offset = math32.Pi / 2
	require.NoError(t, turned.CheckZeroPose(vec.Vector3D{0, 0.184, 0}, 1e-4))
	assert.True(t, errors.Is(turned.CheckZeroPose(vec.Vector3D{0.184, 0, 0}, 1e-3), ErrZeroPose))
}
