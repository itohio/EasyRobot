package chain

import "github.com/itohio/EasyRobot/x/buffer"

// Adapter composes two processors together with different intermediate types.
// Processing flow: src (SRC) -> src.Process() -> intermediate buffer (MID) -> dst.Process() -> dst (DST)
//
// Adapter is used when you need to connect processors with different types:
//   - First processor: SRC -> MID (reads SRC, writes MID)
//   - Second processor: MID -> DST (reads MID, writes DST)
//   - The intermediate buffer stores MID type data, bridging the type gap
//
// This is different from Chain, which requires all processors to have the same input/output type.
// Use Adapter when you need type conversion in your processing pipeline.
//
// Type parameters:
//   - DST: Destination sample type (output type of second processor)
//   - MID: Intermediate sample type (output of first processor, input to second processor)
//   - SRC: Source sample type (input type of first processor)
//   - BUF: Buffer type for intermediate storage (must implement Streamer[MID])
//
// Example usage with Ring buffer:
//   factory := func(cap int) *buffer.Ring[[]byte] {
//       return buffer.NewRing(make([][]byte, 0, cap))
//   }
//   adapter := NewAdapter[float32, []byte, byte, *buffer.Ring[[]byte]](factory, dstProcessor, srcProcessor)
//
//   Flow: src (byte) -> srcProcessor -> buffer ([]byte) -> dstProcessor -> dst (float32)
//
// Example usage with BitsRing:
//   factory := func(cap int) *buffer.BitsRing[byte] {
//       return buffer.NewBitsRing(make([]byte, 0, cap))
//   }
//   adapter := NewAdapter[float32, byte, byte, *buffer.BitsRing[byte]](factory, dstProcessor, srcProcessor)
type Adapter[DST, MID, SRC Sample, BUF Streamer[MID]] struct {
	src           Processor[MID, SRC] // First processor: SRC -> MID
	dst           Processor[DST, MID] // Second processor: MID -> DST
	buf           BUF                 // Intermediate buffer (user-specified type)
	bufferFactory func(int) BUF       // Factory function to create new buffer with given capacity
}

// NewAdapter creates a composed processor that applies two processors in sequence.
// Processing flow: src (SRC) -> src.Process() -> intermediate buffer (MID) -> dst.Process() -> dst (DST)
//
// This allows combining two processing stages with different intermediate types.
// The intermediate buffer is automatically managed and allocated on first use.
//
// Use Adapter when you need to connect processors with different types.
// For processors with the same input/output type, use Chain instead.
//
// Example with Ring buffer:
//   factory := func(cap int) *buffer.Ring[[]byte] {
//       return buffer.NewRing(make([][]byte, 0, cap))
//   }
//   adapter := NewAdapter[float32, []byte, byte, *buffer.Ring[[]byte]](factory, dstProcessor, srcProcessor)
//
//   Flow: src (byte) -> srcProcessor -> buffer ([]byte) -> dstProcessor -> dst (float32)
//
// Parameters:
//   - bufferFactory: Function to create new buffer with given capacity (cap int) -> BUF
//   - dst: Second processor (MID -> DST)
//   - src: First processor (SRC -> MID)
//
// Returns:
//   - Adapter ready for use
func NewAdapter[DST, MID, SRC Sample, BUF Streamer[MID]](
	bufferFactory func(int) BUF,
	dst Processor[DST, MID],
	src Processor[MID, SRC],
) Adapter[DST, MID, SRC, BUF] {
	if bufferFactory == nil {
		panic("Adapter: bufferFactory cannot be nil")
	}
	return Adapter[DST, MID, SRC, BUF]{
		dst:           dst,
		src:           src,
		bufferFactory: bufferFactory,
	}
}

// Process processes samples through two composed processors in sequence.
// Processing flow: src (SRC) -> src.Process() -> intermediate buffer (MID) -> dst.Process() -> dst (DST)
//
// If dst is nil, calculates the theoretical maximum output size:
//   - Calls src processor with nil dst to get intermediate buffer size
//   - Calls dst processor with nil dst and estimated intermediate size
//   - Returns the calculated output size
//
// If dst is not nil, performs actual processing:
//   - src processor reads from src and writes to intermediate buffer
//   - dst processor reads from intermediate buffer and writes to dst
//
// Returns the number of samples written to dst (or calculated size if dst is nil).
//
// Note: Intermediate buffer is allocated on first call with non-nil dst.
// If src size changes, call Process with nil dst to recalculate buffer size.
func (c *Adapter[DST, MID, SRC, BUF]) Process(dst buffer.Writer[DST], src buffer.Reader[SRC]) int {
	// If dst is nil, calculate theoretical maximum output size
	if dst == nil {
		// First, get intermediate buffer size from src processor
		midSize := c.src.Process(nil, src)

		// Then, get output size from dst processor using estimated intermediate size
		dummyReader := &sizeReader[MID]{size: midSize}
		return c.dst.Process(nil, dummyReader)
	}

	// Normal operation: process through both processors
	// Initialize intermediate buffer if needed
	// Check if buffer is initialized by checking if it has capacity
	if c.buf.Cap() == 0 {
		// Calculate intermediate buffer size
		midSize := c.src.Process(nil, src)
		// Add 20% overhead for safety
		bufSize := midSize + midSize/5
		if bufSize < 64 {
			bufSize = 64 // Minimum size
		}
		c.buf = c.bufferFactory(bufSize)
	}

	// Reset intermediate buffer for new processing
	c.buf.Reset()

	// First processor: src (SRC) -> intermediate buffer (MID)
	midWritten := c.src.Process(c.buf, src)
	if midWritten == 0 {
		return 0
	}

	// Second processor: intermediate buffer (MID) -> dst (DST)
	return c.dst.Process(dst, c.buf)
}

func (c *Adapter[DST, MID, SRC, BUF]) Reset() {
	c.dst.Reset()
	c.src.Reset()
	if c.buf.Cap() > 0 {
		c.buf.Reset()
	}
}
