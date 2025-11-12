package planar

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew3DOF(t *testing.T) {
	cfg := [3]Config{
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
	}
	k := New3DOF(cfg)

	assert.Equal(t, 3, k.DOF())
	assert.Len(t, k.Params(), 3)
	assert.Len(t, k.Effector(), 6)
}

func TestPlanar3DOF_Forward(t *testing.T) {
	tests := []struct {
		name    string
		cfg     [3]Config
		params  [3]float32
		want    [3]float32 // x, y, z
		epsilon float32
	}{
		{
			name: "zero angles",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
			},
			params:  [3]float32{0, 0, 0},
			want:    [3]float32{3.0, 0, 0}, // l0 + l1 + l2 = 3.0
			epsilon: 1e-5,
		},
		{
			name: "90 degree base rotation",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
			},
			params:  [3]float32{math32.Pi / 2, 0, 0},
			want:    [3]float32{0, 3.0, 0},
			epsilon: 1e-5,
		},
		{
			name: "45 degree first joint",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
			},
			params:  [3]float32{0, math32.Pi / 4, 0},
			want:    [3]float32{1 + math32.Cos(math32.Pi/4) + math32.Cos(math32.Pi/4), 0, math32.Sin(math32.Pi/4) + math32.Sin(math32.Pi/4)}, // a2 = a1 + a2_param = Pi/4, l0 + l1*cos(a1) + l2*cos(a2), l1*sin(a1) + l2*sin(a2)
			epsilon: 1e-5,
		},
		{
			name: "different link lengths",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 0.5},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.5},
			},
			params:  [3]float32{0, 0, 0},
			want:    [3]float32{3.0, 0, 0}, // 0.5 + 1.0 + 1.5 = 3.0
			epsilon: 1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := New3DOF(tt.cfg).(*p3d)
			copy(k.params[:], tt.params[:])

			require.True(t, k.Forward(), "Forward() returned false")

			effector := k.Effector()
			assert.InDelta(t, tt.want[0], effector[0], float64(tt.epsilon), "pos[0]")
			assert.InDelta(t, tt.want[1], effector[1], float64(tt.epsilon), "pos[1]")
			assert.InDelta(t, tt.want[2], effector[2], float64(tt.epsilon), "pos[2]")
		})
	}
}

func TestPlanar3DOF_Inverse(t *testing.T) {
	tests := []struct {
		name    string
		cfg     [3]Config
		target  [3]float32 // x, y, z
		epsilon float32
		verify  func(t *testing.T, k *p3d, target [3]float32)
	}{
		{
			name: "target at (3, 0, 0)",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
			},
			target:  [3]float32{3.0, 0, 0},
			epsilon: 1e-4,
			verify: func(t *testing.T, k *p3d, target [3]float32) {
				effector := k.Effector()
				// Verify actual position is close to target after Forward() call
				assert.InDelta(t, target[0], effector[0], 1e-3, "actual pos[0]")
				assert.InDelta(t, target[1], effector[1], 1e-3, "actual pos[1]")
				assert.InDelta(t, target[2], effector[2], 1e-3, "actual pos[2]")
			},
		},
		{
			name: "target at (0, 3, 0)",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
			},
			target:  [3]float32{0, 3.0, 0},
			epsilon: 1e-4,
			verify: func(t *testing.T, k *p3d, target [3]float32) {
				effector := k.Effector()
				assert.InDelta(t, target[0], effector[0], 1e-3, "actual pos[0]")
				assert.InDelta(t, target[1], effector[1], 1e-3, "actual pos[1]")
			},
		},
		{
			name: "target with elevation",
			cfg: [3]Config{
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
				{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
			},
			target:  [3]float32{2.0, 0, 1.0},
			epsilon: 1e-4,
			verify: func(t *testing.T, k *p3d, target [3]float32) {
				effector := k.Effector()
				// Check distance from target (allowing some error from IK)
				dist := math32.Sqrt(
					(effector[0]-target[0])*(effector[0]-target[0]) +
						(effector[1]-target[1])*(effector[1]-target[1]) +
						(effector[2]-target[2])*(effector[2]-target[2]))
				assert.LessOrEqual(t, dist, float32(1e-2), "distance from target")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := New3DOF(tt.cfg).(*p3d)
			copy(k.pos[:3], tt.target[:])

			require.True(t, k.Inverse(), "Inverse() returned false")

			if tt.verify != nil {
				tt.verify(t, k, tt.target)
			}
		})
	}
}

func TestPlanar3DOF_RoundTrip(t *testing.T) {
	cfg := [3]Config{
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
		{Min: -math32.Pi, Max: math32.Pi, Length: 1.0},
	}

	testParams := [][3]float32{
		{0, 0, 0},
		{math32.Pi / 4, math32.Pi / 4, 0},
		{math32.Pi / 2, -math32.Pi / 4, math32.Pi / 4},
		{-math32.Pi / 4, math32.Pi / 3, -math32.Pi / 6},
	}

	for i, params := range testParams {
		t.Run(fmt.Sprintf("params_%d", i), func(t *testing.T) {
			k := New3DOF(cfg).(*p3d)
			originalParams := params

			// Forward: params -> position
			copy(k.params[:], originalParams[:])
			require.True(t, k.Forward(), "Forward() returned false")
			originalPos := k.pos

			// Inverse: position -> params
			require.True(t, k.Inverse(), "Inverse() returned false")

			// Forward again: new params -> position
			require.True(t, k.Forward(), "Forward() returned false after Inverse()")

			// Verify position matches (allowing some error from numerical precision)
			dist := math32.Sqrt(
				(k.pos[0]-originalPos[0])*(k.pos[0]-originalPos[0]) +
					(k.pos[1]-originalPos[1])*(k.pos[1]-originalPos[1]) +
					(k.pos[2]-originalPos[2])*(k.pos[2]-originalPos[2]))
			assert.LessOrEqual(t, dist, float32(1e-3), "round-trip position error")
		})
	}
}

func TestPlanar3DOF_JointLimits(t *testing.T) {
	cfg := [3]Config{
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Length: 1.0},
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Length: 1.0},
		{Min: -math32.Pi / 2, Max: math32.Pi / 2, Length: 1.0},
	}

	k := New3DOF(cfg).(*p3d)

	// Test limits
	tests := []struct {
		name     string
		paramIdx int
		value    float32
		expected float32
	}{
		{"below min", 0, -math32.Pi, -math32.Pi / 2},
		{"above max", 0, math32.Pi, math32.Pi / 2},
		{"within limits", 0, math32.Pi / 4, math32.Pi / 4},
		{"below min joint 1", 1, -math32.Pi, -math32.Pi / 2},
		{"above max joint 1", 1, math32.Pi, math32.Pi / 2},
		{"below min joint 2", 2, -math32.Pi, -math32.Pi / 2},
		{"above max joint 2", 2, math32.Pi, math32.Pi / 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k.params[tt.paramIdx] = tt.value
			require.True(t, k.Forward(), "Forward() returned false")
			// Check that parameter was limited
			result := k.c[tt.paramIdx].Limit(tt.value)
			assert.InDelta(t, tt.expected, result, 1e-5, "Limited value")
		})
	}
}
