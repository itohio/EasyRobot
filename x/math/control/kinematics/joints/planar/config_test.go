package planar

import (
	"errors"
	"testing"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Config{} has Min == Max == 0 and silently pinned joints.
func TestConfig_Validate(t *testing.T) {
	nan, inf := math32.NaN(), math32.Inf(1)
	tests := []struct {
		name  string
		cfg   Config
		valid bool
	}{
		{"zero value", Config{}, false},
		{"length only", Config{Length: 0.045}, false},
		{"Min > Max", Config{Min: 1, Max: -1, Length: 1}, false},
		{"NaN Min", Config{Min: nan, Max: 1, Length: 1}, false},
		{"Inf Max", Config{Min: -1, Max: inf, Length: 1}, false},
		{"NaN Length", Config{Min: -1, Max: 1, Length: nan}, false},
		{"negative Length", Config{Min: -1, Max: 1, Length: -0.1}, false},
		{"valid", Config{Min: -pi, Max: pi, Length: 0.1}, true},
		{"valid zero length", Config{Min: -pi, Max: pi}, true},
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
	good := Config{Min: -pi, Max: pi, Length: 1}

	k2, err := New2DOF([2]Config{good, {Length: 1}})
	assert.Nil(t, k2)
	assert.True(t, errors.Is(err, kintypes.ErrInvalidConfig), "2DOF zero limits: got %v", err)

	k3, err := New3DOF([3]Config{{Length: 0.039}, {Length: 0.045}, {Length: 0.1}})
	assert.Nil(t, k3)
	assert.True(t, errors.Is(err, kintypes.ErrInvalidConfig), "3DOF zero limits: got %v", err)

	// The analytic 3-DOF solver divides by the femur and tibia lengths.
	k3, err = New3DOF([3]Config{good, {Min: -1, Max: 1}, good})
	assert.Nil(t, k3)
	assert.True(t, errors.Is(err, kintypes.ErrInvalidConfig), "3DOF zero femur: got %v", err)

	_, err = New2DOF([2]Config{good, good})
	assert.NoError(t, err)
	_, err = New3DOF([3]Config{good, good, good})
	assert.NoError(t, err)
}
