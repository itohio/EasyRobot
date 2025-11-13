package mat

import (
	"testing"

	"github.com/chewxy/math32"
)

func TestCalculateJacobianColumn(t *testing.T) {
	tests := []struct {
		name        string
		jointPos    [3]float32
		jointAxis   [3]float32
		eePos       [3]float32
		isRevolute  bool
		wantLinear  [3]float32
		wantAngular [3]float32
	}{
		{
			name:        "revolute joint at origin, end-effector at x=1",
			jointPos:    [3]float32{0, 0, 0},
			jointAxis:   [3]float32{0, 0, 1}, // Z-axis rotation
			eePos:       [3]float32{1, 0, 0},
			isRevolute:  true,
			wantLinear:  [3]float32{0, 1, 0}, // Z × (1,0,0) = (0,1,0)
			wantAngular: [3]float32{0, 0, 1}, // Z-axis
		},
		{
			name:        "revolute joint at origin, end-effector at y=1",
			jointPos:    [3]float32{0, 0, 0},
			jointAxis:   [3]float32{0, 0, 1}, // Z-axis rotation
			eePos:       [3]float32{0, 1, 0},
			isRevolute:  true,
			wantLinear:  [3]float32{-1, 0, 0}, // Z × (0,1,0) = (-1,0,0)
			wantAngular: [3]float32{0, 0, 1},  // Z-axis
		},
		{
			name:        "prismatic joint along X-axis",
			jointPos:    [3]float32{0, 0, 0},
			jointAxis:   [3]float32{1, 0, 0}, // X-axis translation
			eePos:       [3]float32{1, 0, 0},
			isRevolute:  false,
			wantLinear:  [3]float32{1, 0, 0}, // Translation axis
			wantAngular: [3]float32{0, 0, 0}, // Zero for prismatic
		},
		{
			name:        "revolute joint offset from origin",
			jointPos:    [3]float32{1, 0, 0},
			jointAxis:   [3]float32{0, 0, 1}, // Z-axis rotation
			eePos:       [3]float32{2, 0, 0},
			isRevolute:  true,
			wantLinear:  [3]float32{0, 1, 0}, // Z × (1,0,0) = (0,1,0)
			wantAngular: [3]float32{0, 0, 1}, // Z-axis
		},
		{
			name:        "revolute joint with X-axis rotation",
			jointPos:    [3]float32{0, 0, 0},
			jointAxis:   [3]float32{1, 0, 0}, // X-axis rotation
			eePos:       [3]float32{0, 1, 0},
			isRevolute:  true,
			wantLinear:  [3]float32{0, 0, 1}, // X × (0,1,0) = (0,0,1)
			wantAngular: [3]float32{1, 0, 0}, // X-axis
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CalculateJacobianColumn(tt.jointPos, tt.jointAxis, tt.eePos, tt.isRevolute)

			if !vectorsEqual(got.Linear[:], tt.wantLinear[:], 1e-5) {
				t.Errorf("Linear component = %v, want %v", got.Linear, tt.wantLinear)
			}

			if !vectorsEqual(got.Angular[:], tt.wantAngular[:], 1e-5) {
				t.Errorf("Angular component = %v, want %v", got.Angular, tt.wantAngular)
			}
		})
	}
}

func TestCalculateJacobianColumn_Properties(t *testing.T) {
	// Test that revolute joint at origin with end-effector at distance
	jointPos := [3]float32{0, 0, 0}
	jointAxis := [3]float32{0, 0, 1}
	eePos := [3]float32{2, 0, 0} // Distance 2 along X

	col := CalculateJacobianColumn(jointPos, jointAxis, eePos, true)

	// Linear should be perpendicular to both axis and r
	// Z × (2,0,0) = (0,2,0)
	if math32.Abs(col.Linear[0]) > 1e-5 || math32.Abs(col.Linear[1]-2.0) > 1e-5 || math32.Abs(col.Linear[2]) > 1e-5 {
		t.Errorf("Linear component should be (0,2,0), got %v", col.Linear)
	}

	// Angular should be joint axis
	if math32.Abs(col.Angular[0]) > 1e-5 || math32.Abs(col.Angular[1]) > 1e-5 || math32.Abs(col.Angular[2]-1.0) > 1e-5 {
		t.Errorf("Angular component should be (0,0,1), got %v", col.Angular)
	}
}

func TestCalculateJacobianColumn_PrismaticProperties(t *testing.T) {
	// Test prismatic joint properties
	jointPos := [3]float32{0, 0, 0}
	jointAxis := [3]float32{0, 0, 1} // Z-axis translation
	eePos := [3]float32{1, 2, 3}     // Any end-effector position

	col := CalculateJacobianColumn(jointPos, jointAxis, eePos, false)

	// Linear should be joint axis (regardless of end-effector position)
	if math32.Abs(col.Linear[0]) > 1e-5 || math32.Abs(col.Linear[1]) > 1e-5 || math32.Abs(col.Linear[2]-1.0) > 1e-5 {
		t.Errorf("Prismatic linear should equal joint axis, got %v", col.Linear)
	}

	// Angular should be zero
	if math32.Abs(col.Angular[0]) > 1e-5 || math32.Abs(col.Angular[1]) > 1e-5 || math32.Abs(col.Angular[2]) > 1e-5 {
		t.Errorf("Prismatic angular should be zero, got %v", col.Angular)
	}
}

// Helper function for vector comparison
func vectorsEqual(a, b []float32, eps float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math32.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}
