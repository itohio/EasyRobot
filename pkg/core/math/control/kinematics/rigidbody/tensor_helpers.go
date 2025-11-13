package rigidbody

import (
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
)

func InertiaSolidSphere(mass, radius float32) mat.Matrix3x3 {
	value := 0.4 * mass * radius * radius
	return mat.Matrix3x3{{value, 0, 0}, {0, value, 0}, {0, 0, value}}
}

func InertiaSolidCylinder(mass, radius, height float32) mat.Matrix3x3 {
	iAxis := 0.5 * mass * radius * radius
	iRadial := (mass / 12) * (3*radius*radius + height*height)
	return mat.Matrix3x3{{iRadial, 0, 0}, {0, iRadial, 0}, {0, 0, iAxis}}
}

func InertiaBox(mass, width, height, depth float32) mat.Matrix3x3 {
	return mat.Matrix3x3{
		{(mass / 12) * (height*height + depth*depth), 0, 0},
		{0, (mass / 12) * (width*width + depth*depth), 0},
		{0, 0, (mass / 12) * (width*width + height*height)},
	}
}

func RotateInertia(inertia mat.Matrix3x3, orientation mat.Matrix3x3) mat.Matrix3x3 {
	tmp := mat.Matrix3x3{}.Mul(orientation, inertia).(mat.Matrix3x3)
	orientationT := orientation.Transpose(orientation).(mat.Matrix3x3)
	return mat.Matrix3x3{}.Mul(tmp, orientationT).(mat.Matrix3x3)
}
