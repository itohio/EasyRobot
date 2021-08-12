package ahrs

import "github.com/foxis/EasyRobot/pkg/core/math/vec"

type AHRS interface {
	Acceleration() vec.Vector
	Gyroscope() vec.Vector
	Magnetometer() vec.Vector
	Orientation() vec.Vector
	Reset() AHRS
	Update(samplePeriod float32) AHRS
	Calculate() AHRS
}
