package rigid

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const eps = 1e-6

func SolveTwist(radius float32, wheels []float32, headings []float32, positions []vec.Vector2D) (float32, float32) {
	var (
		saa, sab, sbb float32
		sv, sw        float32
	)
	for i := range wheels {
		linear := wheels[i] * radius
		c := math32.Cos(headings[i])
		s := math32.Sin(headings[i])
		a := c
		b := -positions[i][1]*c + positions[i][0]*s
		saa += a * a
		sab += a * b
		sbb += b * b
		sv += a * linear
		sw += b * linear
	}
	det := saa*sbb - sab*sab
	if math32.Abs(det) < eps {
		return 0, 0
	}
	inv := 1 / det
	v := (sbb*sv - sab*sw) * inv
	omega := (saa*sw - sab*sv) * inv
	return v, omega
}

func AssignWheelRates(radius float32, wheels []float32, v, omega float32, headings []float32, positions []vec.Vector2D) {
	for i := range wheels {
		wheels[i] = WheelLinearVelocity(v, omega, positions[i], headings[i]) / radius
	}
}

func WheelLinearVelocity(v, omega float32, p vec.Vector2D, heading float32) float32 {
	vx := v - omega*p[1]
	vy := omega * p[0]
	return vx*math32.Cos(heading) + vy*math32.Sin(heading)
}
