package chain

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/types"
)

type Sample interface {
	types.Number | types.Unsigned
}

// Streamer is a constraint for types that implement both Writer and Reader interfaces.
// Used for buffer types in Chain and Adapter processors.
type Streamer[T any] interface {
	buffer.Writer[T]
	buffer.Reader[T]
	buffer.Initializer
}

// Processor processes data from source type SRC to destination type DST.
// Processors are used by Chain and Adapter to transform data in pipelines.
type Processor[DST, SRC Sample] interface {
	// Process processes the data from SRC to DST.
	// Returns the number of samples written to dst.
	// If dst is nil, calculates theoretical maximum dst size.
	Process(dst buffer.Writer[DST], src buffer.Reader[SRC]) int

	// Reset resets the processor state.
	Reset()
}
