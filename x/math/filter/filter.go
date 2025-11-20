package filter

import "github.com/itohio/EasyRobot/x/math/vec"

type Resetter interface {
	Reset()
}

type Updater[T any] interface {
	Update(timestep float32, input T)
}

type Filter[T any] interface {
	Updater[T]
	Resetter
	Input() vec.Vector
	Output() vec.Vector
}

type Processor[T any] interface {
	Resetter
	Process(T) T
}
