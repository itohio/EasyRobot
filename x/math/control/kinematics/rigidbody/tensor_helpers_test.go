package rigidbody

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
)

func TestInertiaHelpersProducePositiveDiagonal(t *testing.T) {
	box := InertiaBox(2, 1, 2, 3)
	cyl := InertiaSolidCylinder(1.5, 0.3, 0.8)
	sphere := InertiaSolidSphere(2, 0.4)

	for _, tensor := range []mat.Matrix3x3{box, cyl, sphere} {
		for i := 0; i < 3; i++ {
			if tensor[i][i] <= 0 {
				t.Fatalf("expected positive diagonal, got %f", tensor[i][i])
			}
		}
	}

	rot := RotateInertia(sphere, mat.Matrix3x3{}.Eye().(mat.Matrix3x3))
	for i := 0; i < 3; i++ {
		if math32.Abs(rot[i][i]-sphere[i][i]) > 1e-6 {
			t.Fatalf("rotation should preserve inertia when identity: %f vs %f", rot[i][i], sphere[i][i])
		}
	}
}
