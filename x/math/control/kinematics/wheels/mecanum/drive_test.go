package mecanum

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/stretchr/testify/require"
)

const tol = 1e-5

func closeVec(t *testing.T, got []float32, want []float32) {
	t.Helper()
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > float64(tol) {
			t.Fatalf("component %d mismatch: got %.6f, want %.6f", i, got[i], want[i])
		}
	}
}

func TestDriveForward(t *testing.T) {
	tests := []struct {
		name   string
		params [4]float32
		want   [3]float32
	}{
		{
			name:   "pure forward",
			params: [4]float32{4, 4, 4, 4},
			want:   [3]float32{4, 0, 0},
		},
		{
			name:   "pure strafe",
			params: [4]float32{1, -1, -1, 1},
			want:   [3]float32{0, 1, 0},
		},
		{
			name:   "pure yaw",
			params: [4]float32{-1, 1, -1, 1},
			want:   [3]float32{0, 0, 0.5},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := New(1, 2, 2)
			state := mat.New(4, 1)
			destination := mat.New(3, 1)
			for i, v := range tc.params {
				state[i][0] = v
			}

			require.NoError(t, d.Forward(state, destination, nil))

			eff := destination.View().(mat.Matrix)
			got := []float32{eff[0][0], eff[1][0], eff[2][0]}
			closeVec(t, got, tc.want[:])
		})
	}
}

func TestDriveBackward(t *testing.T) {
	tests := []struct {
		name string
		eff  [3]float32
		want [4]float32
	}{
		{
			name: "forward",
			eff:  [3]float32{4, 0, 0},
			want: [4]float32{4, 4, 4, 4},
		},
		{
			name: "strafe",
			eff:  [3]float32{0, 1, 0},
			want: [4]float32{1, -1, -1, 1},
		},
		{
			name: "yaw",
			eff:  [3]float32{0, 0, 0.5},
			want: [4]float32{-1, 1, -1, 1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := New(1, 2, 2)
			destination := mat.New(3, 1)
			controls := mat.New(4, 1)
			for i, v := range tc.eff {
				destination[i][0] = v
			}

			require.NoError(t, d.Backward(nil, destination, controls))

			ctrl := controls.View().(mat.Matrix)
			got := []float32{ctrl[0][0], ctrl[1][0], ctrl[2][0], ctrl[3][0]}
			closeVec(t, got, tc.want[:])
		})
	}
}

func TestDriveRoundTrip(t *testing.T) {
	original := [4]float32{3.5, -1.25, 2.0, -0.75}
	d := New(1, 2, 2)

	state := mat.New(4, 1)
	destination := mat.New(3, 1)
	controls := mat.New(4, 1)
	result := mat.New(3, 1)

	for i, v := range original {
		state[i][0] = v
	}

	require.NoError(t, d.Forward(state, destination, nil))
	require.NoError(t, d.Backward(nil, destination, controls))
	require.NoError(t, d.Forward(controls, result, nil))

	eff := destination.View().(mat.Matrix)
	res := result.View().(mat.Matrix)
	closeVec(t, []float32{res[0][0], res[1][0], res[2][0]}, []float32{eff[0][0], eff[1][0], eff[2][0]})
}
