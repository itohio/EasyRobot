package chain

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/joints/dh"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Review R-ER-017-018 G9 (C1): a NaN joint value must not reach the servo: it
// is clamped to Min, written finite, and reported as ErrJointLimit.
func TestToServoNaNJoint(t *testing.T) {
	c := hexleg(-1)
	deg := []float32{-1, -1, -1}
	q := []float32{0, math32.NaN(), -1}
	require.ErrorIs(t, c.ToServo(q, deg), kintypes.ErrJointLimit)
	for j, d := range deg {
		assert.False(t, math32.IsNaN(d), "servo %d", j)
	}
	assert.InDelta(t, c.Joints[1].Servo.toDeg(c.Joints[1].Min), deg[1], 1e-4, "NaN clamps to Min")
	assert.InDelta(t, c.Joints[2].Servo.toDeg(-1), deg[2], 1e-4)
}

// Review G6: a chain with the coxa–femur–tibia joint pattern but a femur or
// tibia length <= 0 or a negative coxa is rejected (ErrInvalidConfig), not
// solved by a different (DH) solver.
func TestLegGeometryValidated(t *testing.T) {
	for name, g := range map[string]LegGeometry{
		"femur 0":       {Coxa: 0.039, Femur: 0, Tibia: 0.1},
		"tibia < 0":     {Coxa: 0.039, Femur: 0.045, Tibia: -0.1},
		"coxa < 0":      {Coxa: -0.039, Femur: 0.045, Tibia: 0.1},
		"femur < 0":     {Coxa: 0.039, Femur: -0.045, Tibia: 0.1},
		"tibia 0":       {Coxa: 0.039, Femur: 0.045, Tibia: 0},
		"all zero legs": {},
	} {
		t.Run(name, func(t *testing.T) {
			c := CoxaFemurTibia("leg", g, probeRLimits, servos(1))
			_, err := c.Planar3()
			assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "Planar3")
			m, err := c.Model()
			assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "Model")
			_, isDH := m.(*dh.DenavitHartenberg)
			assert.False(t, isDH, "no silent DH fallback")
			b := Hexapod(0.1, 0, -0.1, 0.06, g)
			_, err = b.NewLegSolver()
			assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "NewLegSolver")
		})
	}
	// A zero coxa is a valid leg (femur pivot on the yaw axis).
	_, err := CoxaFemurTibia("leg", LegGeometry{Femur: 0.045, Tibia: 0.1}, probeRLimits, servos(1)).Planar3()
	assert.NoError(t, err)
}

// Review G4: the solver keeps a deep copy of the body; the caller's later
// edits (and edits to what Body returns) have no effect.
func TestLegSolverCopiesBody(t *testing.T) {
	b := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	s, err := b.NewLegSolver()
	require.NoError(t, err)
	ref, err := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry).NewLegSolver()
	require.NoError(t, err)
	b.Mounts[0].Hip[0] += 0.02
	b.Mounts[1].Yaw += 0.3
	b.Mounts[2].Leg.Joints[0].Min = 0
	b.Neutral[3][0] += 0.02
	got := s.Body()
	got.Mounts[4].Hip[1] += 0.02
	got.Mounts[5].Leg.Joints[1].Max = 0
	assert.Equal(t, Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry), s.Body())

	fresh := Hexapod(0.10, 0, -0.10, 0.06, probeRGeometry)
	pose := Pose{Z: 0.1, Yaw: 0.2}
	feet := make([]vec.Vector3D, 6)
	q, qr := make([][]float32, 6), make([][]float32, 6)
	for i := range feet {
		feet[i] = pose.ToWorld(fresh.Neutral[i])
		q[i], qr[i] = make([]float32, 3), make([]float32, 3)
	}
	errs := make([]error, 6)
	require.NoError(t, s.SolveLegs(pose, feet, q, errs))
	require.NoError(t, ref.SolveLegs(pose, feet, qr, errs))
	assert.Equal(t, qr, q)
}

// Clone shares no memory with the original.
func TestBodyClone(t *testing.T) {
	b := Quadruped(0.12, 0.06, probeRGeometry)
	c := b.Clone()
	require.Equal(t, b, c)
	c.Mounts[0].Leg.Joints[0].A = 1
	c.Mounts[0].Hip[0] = 1
	c.Neutral[0][0] = 1
	assert.Equal(t, Quadruped(0.12, 0.06, probeRGeometry), b)
	assert.Equal(t, Body{}, Body{}.Clone())
}
