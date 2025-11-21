package filter

type Resetter interface {
	Reset()
}

type Updater[T any] interface {
	Update(timestep float32, input T)
}

type Filter[T, K any] interface {
	Updater[T]
	Resetter
	Input() T
	Output() K
}

type Processor[T any] interface {
	Resetter
	Process(input T) T
}
