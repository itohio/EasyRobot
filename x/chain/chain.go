package chain

import "github.com/itohio/EasyRobot/x/buffer"

// Chain chains multiple processors together where all processors have the same input and output type.
// Processing flow: src -> processor1 -> processor2 -> ... -> processorN -> dst
// This allows combining multiple processing layers (e.g., encryption -> reed solomon -> manchester).
//
// Type parameters:
//   - T: Sample type (input and output type for all processors)
//   - BUF: Buffer type for intermediate storage (must implement Streamer[T])
//
// Type constraints:
//   - All processors must be Processor[T, T] (same input and output type)
//   - BUF must implement Streamer[T] (both Writer and Reader interfaces)
//   - This ensures type compatibility: all intermediate buffers are T type,
//     and all processors can read T and write T
//
// Example usage with Ring buffer:
//   factory := func(cap int) *buffer.Ring[float32] {
//       return buffer.NewRing(make([]float32, 0, cap))
//   }
//   chain := New[float32, *buffer.Ring[float32]](factory, processor1, processor2, processor3)
//
// Example usage with BitsRing:
//   factory := func(cap int) *buffer.BitsRing[byte] {
//       return buffer.NewBitsRing(make([]byte, 0, cap))
//   }
//   chain := New[byte, *buffer.BitsRing[byte]](factory, processor1, processor2, processor3)
type Chain[T Sample, BUF Streamer[T]] struct {
	processors    []Processor[T, T]
	buffers       []BUF         // Intermediate buffers for chaining (user-specified type)
	bufferFactory func(int) BUF // Factory function to create new buffers with given capacity
}

// New creates a new chain of processors that process data sequentially.
// Processing flow: src -> processor1 -> processor2 -> ... -> processorN -> dst
//
// This allows combining multiple processing layers (e.g., encryption -> reed solomon -> manchester).
// All processors must have the same input and output type (Processor[T, T]).
//
// Type constraints:
//   - All processors must be Processor[T, T] (same input and output type)
//   - BUF must implement Streamer[T] (both Writer and Reader interfaces)
//   - This ensures type compatibility: intermediate buffers are T type,
//     and all processors can read T and write T
//
// Example usage with Ring buffer:
//   factory := func(cap int) *buffer.Ring[float32] {
//       return buffer.NewRing(make([]float32, 0, cap))
//   }
//   chain := New[float32, *buffer.Ring[float32]](factory, processor1, processor2, processor3)
//
// Parameters:
//   - bufferFactory: Function to create new buffers with given capacity (cap int) -> BUF
//   - processors: Variable number of processors to chain together (all must be Processor[T, T])
//
// Returns:
//   - Chain ready for use
func New[T Sample, BUF Streamer[T]](
	bufferFactory func(int) BUF,
	processors ...Processor[T, T],
) Chain[T, BUF] {
	if bufferFactory == nil {
		panic("Chain: bufferFactory cannot be nil")
	}
	return Chain[T, BUF]{
		processors:    processors,
		bufferFactory: bufferFactory,
	}
}

// Process processes samples through all chained processors in sequence.
// Processing flow: src -> processor1 -> processor2 -> ... -> processorN -> dst
//
// If dst is nil, calculates the theoretical maximum output size by propagating
// size calculations through all processors. This is useful for buffer sizing.
//
// If dst is not nil, performs actual processing:
//   - First processor reads from src and writes to first intermediate buffer
//   - Each subsequent processor reads from previous buffer and writes to next buffer
//   - Last processor writes to dst
//
// Returns the number of samples written to dst (or calculated size if dst is nil).
//
// Note: Intermediate buffers are allocated on first call with non-nil dst.
// If src size changes, call Process with nil dst to recalculate buffer sizes.
func (c *Chain[T, BUF]) Process(dst buffer.Writer[T], src buffer.Reader[T]) int {
	if len(c.processors) == 0 {
		return 0
	}

	// If dst is nil, calculate theoretical maximum output size
	if dst == nil {
		// Start with source size estimation (read available data to estimate)
		// For size calculation, we need to propagate through all processors
		var currentSize int

		// Try to estimate source size by reading a small chunk
		// If src is a buffer, we can check its length
		if srcBuf, ok := src.(interface{ Len() int }); ok {
			currentSize = srcBuf.Len()
		} else {
			// Fallback: use a reasonable default or propagate nil
			// Call first processor with nil to get its output size estimate
			currentSize = c.processors[0].Process(nil, src)
		}

		// Propagate size through all processors
		processors := c.processors[1:]
		for i := range processors {
			// Create a dummy reader that reports the size
			dummyReader := &sizeReader[T]{size: currentSize}
			currentSize = processors[i].Process(nil, dummyReader)
		}
		return currentSize
	}

	// Normal operation: process through all processors
	// Initialize buffers if needed
	if c.buffers == nil || len(c.buffers) < len(c.processors)-1 {
		c.initializeBuffers(src)
	}

	// Chain through all processors
	// All processors have the same input/output type T, so no type conversion needed.
	currentReader := src

	// Process through all processors except the last one
	for i := range c.buffers {
		// Use intermediate buffer for this stage
		buf := c.buffers[i]
		buf.Reset() // Clear buffer for new data

		// Process: read from current reader, write to intermediate buffer
		// BUF implements Streamer[T], so it's both Writer[T] and Reader[T]
		written := c.processors[i].Process(buf, currentReader)
		if written == 0 {
			// No data produced, stop chain
			return 0
		}

		// Next stage reads from this buffer
		// BUF implements buffer.Reader[T], which matches the processor's input type
		currentReader = buf
	}

	// Last processor writes to final destination
	return c.processors[len(c.processors)-1].Process(dst, currentReader)
}

// initializeBuffers initializes intermediate buffers based on processor output sizes.
func (c *Chain[T, BUF]) initializeBuffers(src buffer.Reader[T]) {
	// Calculate sizes by propagating through processors
	var currentSize int

	// Estimate source size
	if srcBuf, ok := src.(interface{ Len() int }); ok {
		currentSize = srcBuf.Len()
	} else {
		// Use first processor to estimate
		currentSize = c.processors[0].Process(nil, src)
	}

	// Allocate buffers with some extra space (20% overhead)
	numBuffers := len(c.processors) - 1
	c.buffers = make([]BUF, numBuffers)

	for i := 0; i < numBuffers; i++ {
		// Calculate output size for this processor
		if i == 0 {
			currentSize = c.processors[i].Process(nil, src)
		} else {
			dummyReader := &sizeReader[T]{size: currentSize}
			currentSize = c.processors[i].Process(nil, dummyReader)
		}

		// Allocate buffer with extra space using factory
		bufSize := currentSize + currentSize/5 // 20% overhead
		if bufSize < 64 {
			bufSize = 64 // Minimum size
		}
		c.buffers[i] = c.bufferFactory(bufSize)
	}
}

// Reset resets all chained processors and buffers.
func (c *Chain[T, BUF]) Reset() {
	for _, processor := range c.processors {
		processor.Reset()
	}
	// Reset buffers (keep allocated capacity)
	for i := range c.buffers {
		c.buffers[i].Reset()
	}
}
