package audio

import (
	"github.com/itohio/EasyRobot/x/buffer"
)

var (
	// Global buffer pools for zero-allocation operations
	bytePool    buffer.Pool[byte]
	int16Pool   buffer.Pool[int16]
	int32Pool   buffer.Pool[int32]
	float32Pool buffer.Pool[float32]
	uint8Pool   buffer.Pool[uint8]
)

