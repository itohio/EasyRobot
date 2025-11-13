package planar

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew2DOF(t *testing.T) {
	cfg := [2]Config{
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
	}
	k := New2DOF(cfg)

	require.NotNil(t, k)
	dims := k.Dimensions()
	assert.Equal(t, 2, dims.StateRows)
	assert.Equal(t, 1, dims.StateCols)
	assert.Equal(t, 2, len(k.params))
}

func TestPlanar2DOF_Forward(t *testing.T) {
	tests := []struct {
		name    string
		cfg     [2]Config
		params  [2]float32
		want    [3]float32
		epsilon float32
	}{
		{
			name:    "zero angles",
			cfg:     [2]Config{{Length: 1.0, Min: -math32.Pi, Max: math32.Pi}, {Length: 1.0, Min: -math32.Pi, Max: math32.Pi}},
			params:  [2]float32{0, 0},
			want:    [3]float32{2, 0, 0},
			epsilon: 1e-5,
		},
		{
			name:    "90 degree base rotation",
			cfg:     [2]Config{{Length: 1.0, Min: -math32.Pi, Max: math32.Pi}, {Length: 1.0, Min: -math32.Pi, Max: math32.Pi}},
			params:  [2]float32{math32.Pi / 2, 0},
			want:    [3]float32{0, 2, 0},
			epsilon: 1e-5,
		},
		{
			name:    "45 degree elbow",
			cfg:     [2]Config{{Length: 1.0, Min: -math32.Pi, Max: math32.Pi}, {Length: 1.0, Min: -math32.Pi, Max: math32.Pi}},
			params:  [2]float32{0, math32.Pi / 4},
			want:    [3]float32{1 + math32.Cos(math32.Pi/4), 0, math32.Sin(math32.Pi / 4)},
			epsilon: 1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := New2DOF(tt.cfg)
			state := mat.New(2, 1)
			dest := mat.New(planarEffectorSize, 1)
			state[0][0] = tt.params[0]
			state[1][0] = tt.params[1]

			require.NoError(t, k.Forward(state, dest, nil))

			eff := dest.View().(mat.Matrix)
			assert.InDelta(t, tt.want[0], eff[0][0], float64(tt.epsilon))
			assert.InDelta(t, tt.want[1], eff[1][0], float64(tt.epsilon))
			assert.InDelta(t, tt.want[2], eff[2][0], float64(tt.epsilon))
		})
	}
}

func TestPlanar2DOF_Backward(t *testing.T) {
	tests := []struct {
		name      string
		cfg       [2]Config
		target    [3]float32
		tolerance float32
	}{
		{
			name:      "target at (2,0,0)",
			cfg:       [2]Config{{Length: 1.0, Min: -math32.Pi, Max: math32.Pi}, {Length: 1.0, Min: -math32.Pi, Max: math32.Pi}},
			target:    [3]float32{2, 0, 0},
			tolerance: 1e-3,
		},
		{
			name:      "target at (0,2,0)",
			cfg:       [2]Config{{Length: 1.0, Min: -math32.Pi, Max: math32.Pi}, {Length: 1.0, Min: -math32.Pi, Max: math32.Pi}},
			target:    [3]float32{0, 2, 0},
			tolerance: 1e-3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := New2DOF(tt.cfg)
			state := mat.New(2, 1)
			dest := mat.New(planarEffectorSize, 1)
			controls := mat.New(2, 1)

			dest[0][0] = tt.target[0]
			dest[1][0] = tt.target[1]
			dest[2][0] = tt.target[2]

			require.NoError(t, k.Backward(state, dest, controls))

			result := mat.New(planarEffectorSize, 1)
			require.NoError(t, k.Forward(controls, result, nil))
			eff := result.View().(mat.Matrix)
			dx := eff[0][0] - tt.target[0]
			dy := eff[1][0] - tt.target[1]
			dz := eff[2][0] - tt.target[2]
			dist := math32.Sqrt(dx*dx + dy*dy + dz*dz)
			assert.InDelta(t, 0, dist, float64(tt.tolerance))
		})
	}
}

func TestPlanar2DOF_RoundTrip(t *testing.T) {
	cfg := [2]Config{{Length: 1.0, Min: -math32.Pi, Max: math32.Pi}, {Length: 1.0, Min: -math32.Pi, Max: math32.Pi}}

	scenarios := [][2]float32{{0, 0}, {math32.Pi / 4, math32.Pi / 4}, {math32.Pi / 2, -math32.Pi / 4}}

	for i, params := range scenarios {
		t.Run(fmt.Sprintf("round_trip_%d", i), func(t *testing.T) {
			k := New2DOF(cfg)
			state := mat.New(2, 1)
			dest := mat.New(planarEffectorSize, 1)
			controls := mat.New(2, 1)
			result := mat.New(planarEffectorSize, 1)

			state[0][0] = params[0]
			state[1][0] = params[1]

			require.NoError(t, k.Forward(state, dest, nil))
			require.NoError(t, k.Backward(state, dest, controls))
			require.NoError(t, k.Forward(controls, result, nil))

			eff := result.View().(mat.Matrix)
			dx := eff[0][0] - dest[0][0]
			dy := eff[1][0] - dest[1][0]
			dz := eff[2][0] - dest[2][0]
			dist := math32.Sqrt(dx*dx + dy*dy + dz*dz)
			assert.InDelta(t, 0, dist, 1e-3)
		})
	}
}

func TestPlanar2DOF_JointLimits(t *testing.T) {
	cfg := [2]Config{
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Length: 1.0},
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Length: 1.0},
	}

	k := New2DOF(cfg)
	state := mat.New(2, 1)
	dest := mat.New(planarEffectorSize, 1)

	tests := []struct {
		name     string
		idx      int
		value    float32
		expected float32
	}{
		{"below min", 0, -math32.Pi, -math32.Pi / 2},
		{"above max", 0, math32.Pi, math32.Pi / 2},
		{"within", 0, math32.Pi / 4, math32.Pi / 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state[0][0] = 0
			state[1][0] = 0
			state[tt.idx][0] = tt.value
			require.NoError(t, k.Forward(state, dest, nil))
			assert.InDelta(t, tt.expected, k.params[tt.idx], 1e-5)
		})
	}
}
