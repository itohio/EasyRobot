package thrusters

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQuadcopterForwardHover(t *testing.T) {
	body := quadBody(1.0, 0.05, 0.05, 0.1)
	thrusters := quadThrusters(0.25, Range{Min: 0, Max: 10})
	model, err := NewModel(body, thrusters)
	require.NoError(t, err)

	state := mat.New(len(thrusters)*2, 1)
	destination := mat.New(wrenchSize, 1)
	const hoverThrust = 2.5
	for i := range thrusters {
		state[2*i][0] = hoverThrust
		state[2*i+1][0] = 0
	}

	require.NoError(t, model.Forward(state, destination, nil))

	eff := destination.View().(mat.Matrix)
	totalThrust := hoverThrust * float32(len(thrusters))
	assert.InDelta(t, 0, eff[0][0], 1e-5)
	assert.InDelta(t, 0, eff[1][0], 1e-5)
	assert.InDelta(t, totalThrust, eff[2][0], 1e-5)
	assert.InDelta(t, 0, eff[3][0], 1e-5)
	assert.InDelta(t, 0, eff[4][0], 1e-5)
	assert.InDelta(t, 0, eff[5][0], 1e-5)
}

func TestQuadcopterInverseHover(t *testing.T) {
	body := quadBody(1.0, 0.05, 0.05, 0.1)
	thrusters := quadThrusters(0.25, Range{Min: 0, Max: 10})
	model, err := NewModel(body, thrusters)
	require.NoError(t, err)

	destination := mat.New(wrenchSize, 1)
	destination[2][0] = 10
	controls := mat.New(len(thrusters)*2, 1)

	require.NoError(t, model.Backward(nil, destination, controls))

	ctrl := controls.View().(mat.Matrix)
	var thrustSum float32
	for i := 0; i < len(thrusters); i++ {
		thrust := ctrl[2*i][0]
		torque := ctrl[2*i+1][0]
		thrustSum += thrust
		assert.InDeltaf(t, 0, torque, 1e-3, "thruster %d torque", i)
	}
	avgThrust := thrustSum / float32(len(thrusters))
	for i := 0; i < len(thrusters); i++ {
		assert.InDeltaf(t, avgThrust, ctrl[2*i][0], 1e-3, "thruster %d thrust", i)
	}

	result := mat.New(wrenchSize, 1)
	require.NoError(t, model.Forward(controls, result, nil))
	eff := result.View().(mat.Matrix)
	assert.InDelta(t, destination[2][0], eff[2][0], 1e-3)
	assert.InDelta(t, 0, math32.Abs(eff[3][0]), 1e-3)
	assert.InDelta(t, 0, math32.Abs(eff[4][0]), 1e-3)
	assert.InDelta(t, 0, math32.Abs(eff[5][0]), 1e-3)
}

func TestInverseInfeasibleDueToLimits(t *testing.T) {
	body := quadBody(1.0, 0.05, 0.05, 0.1)
	thrusters := quadThrusters(0.25, Range{Min: 0, Max: 2})
	model, err := NewModel(body, thrusters)
	require.NoError(t, err)

	destination := mat.New(wrenchSize, 1)
	destination[2][0] = 10
	controls := mat.New(len(thrusters)*2, 1)

	err = model.Backward(nil, destination, controls)
	require.ErrorIs(t, err, ErrInfeasible)
}

func quadBody(mass, ix, iy, iz float32) Body {
	return Body{
		Mass:    mass,
		Inertia: mat.FromDiagonal3x3(ix, iy, iz),
	}
}

func quadThrusters(armLength float32, thrust Range) []Thruster {
	positions := [][2]float32{
		{-armLength, armLength},
		{armLength, armLength},
		{-armLength, -armLength},
		{armLength, -armLength},
	}
	spin := []float32{1, -1, -1, 1}
	thrusters := make([]Thruster, len(positions))
	for i, pos := range positions {
		thrusters[i] = Thruster{
			Position:    vec.Vector3D{pos[0], pos[1], 0},
			Direction:   vec.Vector3D{0, 0, 1},
			TorqueAxis:  vec.Vector3D{0, 0, spin[i]},
			ThrustLimit: thrust,
			TorqueLimit: Range{Min: -0.5, Max: 0.5},
		}
	}
	return thrusters
}
