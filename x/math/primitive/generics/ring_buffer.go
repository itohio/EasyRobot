package generics

import (
	"io"
)

// RingBuffer is a generic ring buffer that uses a provided backing slice.
// When instantiated with byte type, it implements io.Reader and io.Writer.
// The buffer uses a circular array pattern with read and write positions.
type RingBuffer[T any] struct {
	buf  []T
	read int // read position
	len  int // number of elements currently in buffer
}

// NewRing creates a new RingBuffer using the provided backing slice.
// The capacity is determined by the slice's capacity, not its length.
// The backing slice should have sufficient capacity for the desired buffer size.
func NewRing[T any](backing []T) *RingBuffer[T] {
	if cap(backing) == 0 {
		return &RingBuffer[T]{buf: backing}
	}
	// Use the full capacity of the backing slice
	buf := backing[:cap(backing)]
	return &RingBuffer[T]{buf: buf}
}

// Write writes elements from p into the ring buffer.
// Returns the number of elements written, which may be less than len(p) if the buffer is full.
func (rb *RingBuffer[T]) Write(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if rb.available() == 0 {
		return 0, io.ErrShortWrite
	}

	cap := rb.capacity()
	write := (rb.read + rb.len) % cap
	avail := rb.available()
	toWrite := min(len(p), avail)

	// Write in one or two chunks (if wrapping around)
	firstChunk := min(toWrite, cap-write)
	for i := range firstChunk {
		rb.buf[write+i] = p[i]
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

	rb.len += n
	return n, nil
}

// Read reads elements from the ring buffer into p.
// Returns the number of elements read, which may be less than len(p) if the buffer is empty.
func (rb *RingBuffer[T]) Read(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if rb.len == 0 {
		return 0, io.EOF
	}

	cap := rb.capacity()
	toRead := min(len(p), rb.len)

	// Read in one or two chunks (if wrapping around)
	firstChunk := min(toRead, cap-rb.read)
	for i := range firstChunk {
		p[i] = rb.buf[rb.read+i]
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

	rb.read = (rb.read + n) % cap
	rb.len -= n
	return n, nil
}

// WriteItem writes a single element to the ring buffer.
// Returns an error if the buffer is full.
func (rb *RingBuffer[T]) WriteItem(c T) error {
	if rb.available() == 0 {
		return io.ErrShortWrite
	}
	cap := rb.capacity()
	write := (rb.read + rb.len) % cap
	rb.buf[write] = c
	rb.len++
	return nil
}

// ReadItem reads a single element from the ring buffer.
// Returns an error if the buffer is empty.
func (rb *RingBuffer[T]) ReadItem() (T, error) {
	var zero T
	if rb.len == 0 {
		return zero, io.EOF
	}
	cap := rb.capacity()
	c := rb.buf[rb.read]
	rb.read = (rb.read + 1) % cap
	rb.len--
	return c, nil
}

// Len returns the number of elements currently in the buffer.
func (rb *RingBuffer[T]) Len() int {
	return rb.len
}

// Cap returns the capacity of the ring buffer.
func (rb *RingBuffer[T]) Cap() int {
	return rb.capacity()
}

// Available returns the number of elements that can be written without overwriting unread data.
func (rb *RingBuffer[T]) Available() int {
	return rb.available()
}

// Reset clears the buffer by resetting read position and length.
func (rb *RingBuffer[T]) Reset() {
	rb.read = 0
	rb.len = 0
}

// Full returns true if the buffer is full (no space available).
func (rb *RingBuffer[T]) Full() bool {
	return rb.available() == 0
}

// Empty returns true if the buffer is empty.
func (rb *RingBuffer[T]) Empty() bool {
	return rb.len == 0
}

// capacity returns the capacity of the backing buffer.
func (rb *RingBuffer[T]) capacity() int {
	return cap(rb.buf)
}

// available returns the number of available slots.
func (rb *RingBuffer[T]) available() int {
	return rb.capacity() - rb.len
}

// Ensure RingBuffer[byte] implements io.Reader and io.Writer
var (
	_ io.Reader = (*RingBuffer[byte])(nil)
	_ io.Writer = (*RingBuffer[byte])(nil)
)
