package dh

import (
	"errors"
	"testing"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A zero-value Config (Min == Max == 0) silently pinned every joint
// at 0; it and other malformed configs must be rejected.
func TestConfig_Validate(t *testing.T) {
	nan, inf := math32.NaN(), math32.Inf(1)
	tests := []struct {
		name  string
		cfg   Config
		valid bool
	}{
		{"zero value (Min == Max == 0)", Config{}, false},
		{"lengths only, no limits", Config{R: 0.045, Alpha: pi / 2}, false},
		{"Min > Max", Config{Min: 1, Max: -1}, false},
		{"NaN Min", Config{Min: nan, Max: 1}, false},
		{"Inf Max", Config{Min: -1, Max: inf}, false},
		{"NaN R", Config{Min: -1, Max: 1, R: nan}, false},
		{"Inf D", Config{Min: -1, Max: 1, D: inf}, false},
		{"NaN Theta", Config{Min: -1, Max: 1, Theta: nan}, false},
		{"NaN Alpha", Config{Min: -1, Max: 1, Alpha: nan}, false},
		{"Index -1", Config{Min: -1, Max: 1, Index: -1}, false},
		{"Index 4", Config{Min: -1, Max: 1, Index: 4}, false},
		{"revolute θ", Config{Min: -pi, Max: pi, R: 1, Index: 0}, true},
		{"α joint", Config{Min: -1, Max: 1, Index: 1}, true},
		{"r joint", Config{Min: 0, Max: 1, Index: 2}, true},
		{"prismatic d", Config{Min: 0, Max: 2, Index: 3}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if tt.valid {
				assert.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.True(t, errors.Is(err, kintypes.ErrInvalidConfig), "got %v", err)
		})
	}
}

func TestNew_RejectsInvalidConfig(t *testing.T) {
	good := Config{Min: -pi, Max: pi, R: 1}
	tests := []struct {
		name    string
		eps     float32
		maxIter int
		cfg     []Config
	}{
		{"zero-value joint (review probe F2)", 1e-4, 100, []Config{{Alpha: pi / 2, R: 0.039}, {R: 0.045}, {R: 0.100}}},
		{"one bad joint among good ones", 1e-4, 100, []Config{good, {Min: 1, Max: 1, R: 1}}},
		{"no joints", 1e-4, 100, nil},
		{"eps <= 0", 0, 100, []Config{good}},
		{"eps NaN", math32.NaN(), 100, []Config{good}},
		{"negative iterations", 1e-4, -1, []Config{good}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k, err := New(tt.eps, tt.maxIter, tt.cfg...)
			require.Error(t, err)
			assert.Nil(t, k)
			assert.True(t, errors.Is(err, kintypes.ErrInvalidConfig), "got %v", err)
		})
	}

	k, err := New(1e-4, 100, good, good)
	require.NoError(t, err)
	assert.NotNil(t, k)
}
