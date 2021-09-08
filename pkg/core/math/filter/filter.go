package filter

import "github.com/foxis/EasyRobot/pkg/core/math/vec"

type Filter interface {
	Reset() Filter
	Update(timestep float32) Filter
	GetInput() vec.Vector
	GetOutput() vec.Vector
	GetTarget() vec.Vector
}

type Filter1D interface {
	Reset() Filter1D
	Update(timestep float32) Filter1D
	GetInput() *float32
	GetOutput() *float32
	GetTarget() *float32
}
