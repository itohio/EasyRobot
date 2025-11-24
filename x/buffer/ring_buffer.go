package buffer

import (
	"errors"
	"io"
)

var _ Writer[byte] = (*Ring[byte])(nil)
var _ Reader[byte] = (*Ring[byte])(nil)
var _ Initializer = (*Ring[byte])(nil)

// Ring is a generic ring buffer that uses a provided backing slice.
// When instantiated with byte type, it implements io.Reader and io.Writer.
// The buffer uses a circular array pattern with head (write) and tail (read) positions.
type Ring[T any] struct {
	buf  []T
	head int // write position (where new data is written)
	tail int // read position (where data is read from)
	len  int // number of elements currently in buffer
}

// NewRing creates a new RingBuffer using the provided backing slice.
// The capacity is determined by the slice's capacity, not its length.
// The backing slice should have sufficient capacity for the desired buffer size.
func NewRing[T any](backing []T) *Ring[T] {
	if cap(backing) == 0 {
		return &Ring[T]{buf: backing}
	}
	// Use the full capacity of the backing slice
	buf := backing[:cap(backing)]
	return &Ring[T]{buf: buf}
}

// Write writes elements from p into the ring buffer.
// Returns the number of elements written, which may be less than len(p) if the buffer is full.
func (rb *Ring[T]) Write(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if rb.available() == 0 {
		return 0, io.ErrShortWrite
	}

	cap := rb.capacity()
	avail := rb.available()
	toWrite := min(len(p), avail)

	// Write in one or two chunks (if wrapping around)
	firstChunk := min(toWrite, cap-rb.head)
	for i := range firstChunk {
		rb.buf[rb.head+i] = p[i]
	}
	n = firstChunk

	if toWrite > firstChunk {
		// Second chunk wraps around
		secondChunk := toWrite - firstChunk
		for i := range secondChunk {
			rb.buf[i] = p[firstChunk+i]
		}
		n = toWrite
	}

	rb.head = (rb.head + n) % cap
	rb.len += n
	return n, nil
}

// Read reads elements from the ring buffer into p.
// Returns the number of elements read, which may be less than len(p) if the buffer is empty.
// Returns io.EOF when the buffer is empty or when fewer items are read than requested.
func (rb *Ring[T]) Read(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if rb.len == 0 {
		return 0, io.EOF
	}

	cap := rb.capacity()
	toRead := min(len(p), rb.len)

	// Read in one or two chunks (if wrapping around)
	firstChunk := min(toRead, cap-rb.tail)
	for i := range firstChunk {
		p[i] = rb.buf[rb.tail+i]
	}
	n = firstChunk

	if toRead > firstChunk {
		// Second chunk wraps around
		secondChunk := toRead - firstChunk
		for i := range secondChunk {
			p[firstChunk+i] = rb.buf[i]
		}
		n = toRead
	}

	rb.tail = (rb.tail + n) % cap
	rb.len -= n

	// Return EOF if we read fewer items than requested and buffer is now empty
	// This indicates end of stream (no more data available)
	if n < len(p) && rb.len == 0 {
		return n, io.EOF
	}
	return n, nil
}

// WriteItem writes a single element to the ring buffer.
// Returns an error if the buffer is full.
func (rb *Ring[T]) WriteItem(c T) error {
	if rb.available() == 0 {
		return io.ErrShortWrite
	}
	cap := rb.capacity()
	rb.buf[rb.head] = c
	rb.head = (rb.head + 1) % cap
	rb.len++
	return nil
}

// ReadItem reads a single element from the ring buffer.
// Returns an error if the buffer is empty.
func (rb *Ring[T]) ReadItem() (T, error) {
	var zero T
	if rb.len == 0 {
		return zero, io.EOF
	}
	cap := rb.capacity()
	c := rb.buf[rb.tail]
	rb.tail = (rb.tail + 1) % cap
	rb.len--
	return c, nil
}

// Len returns the number of elements currently in the buffer.
func (rb *Ring[T]) Len() int {
	return rb.len
}

// Cap returns the capacity of the ring buffer.
func (rb *Ring[T]) Cap() int {
	return rb.capacity()
}

// Available returns the number of elements that can be written without overwriting unread data.
func (rb *Ring[T]) Available() int {
	return rb.available()
}

// Reset clears the buffer by resetting head, tail positions and length.
func (rb *Ring[T]) Reset() {
	rb.head = 0
	rb.tail = 0
	rb.len = 0
}

// Full returns true if the buffer is full (no space available).
func (rb *Ring[T]) Full() bool {
	return rb.available() == 0
}

// Empty returns true if the buffer is empty.
func (rb *Ring[T]) Empty() bool {
	return rb.len == 0
}

// capacity returns the capacity of the backing buffer.
func (rb *Ring[T]) capacity() int {
	return cap(rb.buf)
}

// available returns the number of available slots.
func (rb *Ring[T]) available() int {
	return rb.capacity() - rb.len
}

// Grow grows the buffer's capacity to guarantee space for n more elements.
// After Grow(n), at least n elements can be written without another allocation.
// Note: Growing a ring buffer requires reallocating and copying existing data.
func (rb *Ring[T]) Grow(n int) {
	if n < 0 {
		panic("Ring.Grow: negative count")
	}
	if n == 0 {
		return
	}

	currentCap := rb.capacity()
	if rb.available() >= n {
		return // Already have enough space
	}

	// Calculate new capacity
	newCap := currentCap + n
	if newCap < 4 {
		newCap = 4
	}

	// Allocate new buffer
	newBuf := make([]T, newCap)

	// Copy existing data preserving order
	if rb.len > 0 {
		oldCap := currentCap
		toRead := rb.len
		readPos := rb.tail

		// Copy in chunks (handle wrapping)
		firstChunk := min(toRead, oldCap-readPos)
		copy(newBuf, rb.buf[readPos:readPos+firstChunk])

		if toRead > firstChunk {
			secondChunk := toRead - firstChunk
			copy(newBuf[firstChunk:], rb.buf[:secondChunk])
		}
	}

	// Update buffer and positions
	rb.buf = newBuf
	rb.tail = 0
	rb.head = rb.len
}

// Init initializes the ring buffer with the given parameters.
// For Ring buffer, numBits is ignored (not applicable).
// Returns an error if bufLen or bufCap are invalid.
func (rb *Ring[T]) Init(numBits, bufLen, bufCap int) error {
	if bufLen < 0 {
		return errors.New("buffer.Init: length must be non-negative")
	}
	if bufCap < bufLen {
		return errors.New("buffer.Init: capacity must be >= length")
	}
	if bufCap == 0 {
		rb.buf = nil
	} else {
		rb.buf = make([]T, bufCap)
	}
	rb.head = 0
	rb.tail = 0
	rb.len = 0
	return nil
}

// Ensure RingBuffer[byte] implements io.Reader and io.Writer
var (
	_ io.Reader = (*Ring[byte])(nil)
	_ io.Writer = (*Ring[byte])(nil)
)
