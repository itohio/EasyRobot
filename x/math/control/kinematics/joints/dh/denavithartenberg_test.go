package dh

import (
	"errors"
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	cfg := []Config{
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
	}
	k := New(1e-5, 10, cfg...)

	require.NotNil(t, k)
	assert.Equal(t, len(cfg), k.Dimensions().StateRows)
	assert.Equal(t, len(cfg)+1, len(k.H0i))
	assert.Equal(t, []int{0, 0}, k.jointTypes)
	assert.Equal(t, len(cfg), len(k.params))
}

func TestDenavitHartenberg_Forward(t *testing.T) {
	tests := []struct {
		name    string
		cfg     []Config
		params  []float32
		want    [3]float32
		epsilon float32
	}{
		{
			name: "zero angles, 2DOF",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			params:  []float32{0, 0},
			want:    [3]float32{2.0, 0, 0},
			epsilon: 1e-5,
		},
		{
			name: "90 degree first joint, 2DOF",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			params:  []float32{math32.Pi / 2, 0},
			want:    [3]float32{0, 2.0, 0},
			epsilon: 1e-5,
		},
		{
			name: "3DOF chain",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			params:  []float32{0, 0, 0},
			want:    [3]float32{3.0, 0, 0},
			epsilon: 1e-5,
		},
		{
			name: "with D offset",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: math32.Pi / 2, R: 1.0, D: 0.5, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			params:  []float32{0, 0},
			want:    [3]float32{2.0, 0, 0.5},
			epsilon: 1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := New(1e-5, 10, tt.cfg...)
			state := mat.New(len(tt.params), 1)
			for i, v := range tt.params {
				state[i][0] = v
			}
			destination := mat.New(effectorSize, 1)

			require.NoError(t, k.Forward(state, destination, nil))

			effector := destination.View().(mat.Matrix)
			assert.InDelta(t, tt.want[0], effector[0][0], float64(tt.epsilon), "pos[0]")
			assert.InDelta(t, tt.want[1], effector[1][0], float64(tt.epsilon), "pos[1]")
			assert.InDelta(t, tt.want[2], effector[2][0], float64(tt.epsilon), "pos[2]")
		})
	}
}

func TestDenavitHartenberg_ForwardPopulatesTransforms(t *testing.T) {
	cfg := []Config{
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
	}

	k := New(1e-5, 10, cfg...)
	state := mat.New(2, 1)
	destination := mat.New(effectorSize, 1)

	params := []float32{math32.Pi / 4, math32.Pi / 6}
	for i, v := range params {
		state[i][0] = v
	}
	require.NoError(t, k.Forward(state, destination, nil))

	identity := mat.Matrix4x4{}.Eye().(mat.Matrix4x4)
	assertMatrix4x4AlmostEqual(t, identity, k.H0i[0], 1e-6)

	var expected1 mat.Matrix4x4
	require.True(t, cfg[0].CalculateTransform(params[0], &expected1))
	assertMatrix4x4AlmostEqual(t, expected1, k.H0i[1], 1e-5)

	var joint2 mat.Matrix4x4
	require.True(t, cfg[1].CalculateTransform(params[1], &joint2))
	expectedChain := (mat.Matrix4x4{}).Mul(expected1, joint2).(mat.Matrix4x4)
	assertMatrix4x4AlmostEqual(t, expectedChain, k.H0i[2], 1e-5)

	newParams := []float32{0, math32.Pi / 2}
	for i, v := range newParams {
		state[i][0] = v
	}
	require.NoError(t, k.Forward(state, destination, nil))
	assertMatrix4x4AlmostEqual(t, identity, k.H0i[0], 1e-6)

	var expected1b mat.Matrix4x4
	require.True(t, cfg[0].CalculateTransform(newParams[0], &expected1b))
	assertMatrix4x4AlmostEqual(t, expected1b, k.H0i[1], 1e-5)

	var joint2b mat.Matrix4x4
	require.True(t, cfg[1].CalculateTransform(newParams[1], &joint2b))
	expectedChainB := (mat.Matrix4x4{}).Mul(expected1b, joint2b).(mat.Matrix4x4)
	assertMatrix4x4AlmostEqual(t, expectedChainB, k.H0i[2], 1e-5)
}

func TestDenavitHartenberg_Backward(t *testing.T) {
	identityQuat := [4]float32{0, 0, 0, 1}
	tests := []struct {
		name       string
		cfg        []Config
		target     [3]float32
		eps        float32
		maxIter    int
		shouldFail bool
		tolerance  float32
	}{
		{
			name: "target at (2,0,0), 2DOF",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			target:    [3]float32{2, 0, 0},
			eps:       1e-4,
			maxIter:   50,
			tolerance: 1e-2,
		},
		{
			name: "target at (0,2,0), 2DOF",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			target:    [3]float32{0, 2, 0},
			eps:       1e-3,
			maxIter:   100,
			tolerance: 5e-2,
		},
		{
			name: "target with elevation",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: math32.Pi / 2, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			target:     [3]float32{1.5, 0, 0.5},
			eps:        1e-3,
			maxIter:    100,
			shouldFail: true,
		},
		{
			name: "unreachable target",
			cfg: []Config{
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
				{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
			},
			target:     [3]float32{10, 0, 0},
			eps:        1e-4,
			maxIter:    10,
			shouldFail: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := New(tt.eps, tt.maxIter, tt.cfg...)
			state := mat.New(len(tt.cfg), 1)
			dest := mat.New(effectorSize, 1)
			controls := mat.New(len(tt.cfg), 1)

			for i := 0; i < len(identityQuat); i++ {
				dest[i+3][0] = identityQuat[i]
			}
			for i := 0; i < 3; i++ {
				dest[i][0] = tt.target[i]
			}

			err := k.Backward(state, dest, controls)
			if tt.shouldFail {
				require.Error(t, err)
				assert.True(t, errors.Is(err, ErrNoConvergence) || errors.Is(err, kintypes.ErrUnsupportedOperation))
				return
			}

			require.NoError(t, err)

			result := mat.New(effectorSize, 1)
			require.NoError(t, k.Forward(controls, result, nil))
			eff := result.View().(mat.Matrix)
			dx := eff[0][0] - tt.target[0]
			dy := eff[1][0] - tt.target[1]
			dz := eff[2][0] - tt.target[2]
			dist := math32.Sqrt(dx*dx + dy*dy + dz*dz)
			assert.LessOrEqual(t, dist, tt.tolerance)
		})
	}
}

func TestDenavitHartenberg_RoundTrip(t *testing.T) {
	cfg := []Config{
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
	}

	testParams := [][]float32{{0, 0}, {math32.Pi / 4, math32.Pi / 4}, {math32.Pi / 2, -math32.Pi / 4}, {-math32.Pi / 4, math32.Pi / 3}}

	identityQuat := [4]float32{0, 0, 0, 1}

	for i, params := range testParams {
		t.Run(fmt.Sprintf("params_%d", i), func(t *testing.T) {
			k := New(1e-5, 50, cfg...)
			state := mat.New(len(cfg), 1)
			dest := mat.New(effectorSize, 1)
			controls := mat.New(len(cfg), 1)
			result := mat.New(effectorSize, 1)

			for idx, v := range params {
				state[idx][0] = v
			}

			require.NoError(t, k.Forward(state, dest, nil))
			for j := 0; j < len(identityQuat); j++ {
				dest[j+3][0] = identityQuat[j]
			}

			require.NoError(t, k.Backward(state, dest, controls))
			require.NoError(t, k.Forward(controls, result, nil))

			eff := result.View().(mat.Matrix)
			dx := eff[0][0] - dest[0][0]
			dy := eff[1][0] - dest[1][0]
			dz := eff[2][0] - dest[2][0]
			dist := math32.Sqrt(dx*dx + dy*dy + dz*dz)
			assert.LessOrEqual(t, dist, float32(1e-2), "round-trip position error")
		})
	}
}

func TestDenavitHartenberg_JointLimits(t *testing.T) {
	cfg := []Config{
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Theta: 0, Alpha: 0, R: 1.0, D: 0, Index: 0},
	}

	k := New(1e-5, 10, cfg...)

	tests := []struct {
		name     string
		paramIdx int
		value    float32
		expected float32
	}{
		{"below min", 0, -math32.Pi, -math32.Pi / 2},
		{"above max", 0, math32.Pi, math32.Pi / 2},
		{"within limits", 0, math32.Pi / 4, math32.Pi / 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := mat.New(len(cfg), 1)
			state[tt.paramIdx][0] = tt.value
			dest := mat.New(effectorSize, 1)

			require.NoError(t, k.Forward(state, dest, nil))
			assert.InDelta(t, tt.expected, k.params[tt.paramIdx], 1e-5)
		})
	}
}

func TestDenavitHartenberg_PrismaticJoint(t *testing.T) {
	cfg := []Config{
		{Min: -math32.Pi, Max: math32.Pi, Theta: 0, Alpha: 0, R: 0, D: 0, Index: 0},
		{Min: 0, Max: 2.0, Theta: 0, Alpha: 0, R: 0, D: 0, Index: 3},
	}

	k := New(1e-5, 10, cfg...)
	state := mat.New(len(cfg), 1)
	state[0][0] = 0
	state[1][0] = 1.0
	dest := mat.New(effectorSize, 1)

	require.NoError(t, k.Forward(state, dest, nil))
	eff := dest.View().(mat.Matrix)
	assert.InDelta(t, 0.0, eff[0][0], 1e-5)
	assert.InDelta(t, 0.0, eff[1][0], 1e-5)
	assert.InDelta(t, 1.0, eff[2][0], 1e-5)
}

func assertMatrix4x4AlmostEqual(t *testing.T, expected, actual mat.Matrix4x4, tol float32) {
	t.Helper()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			assert.InDeltaf(t, expected[i][j], actual[i][j], float64(tol), "m[%d][%d]", i, j)
		}
	}
}
