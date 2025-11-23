package buffer

import (
	"io"
)

// Linear is a generic buffer that mimics bytes.Buffer functionality.
// It uses a provided backing slice and grows it as needed.
// When instantiated with byte type, it implements io.Reader and io.Writer.
type Linear[T any] struct {
	buf []T
	off int // read/write offset
}

// NewBuffer creates a new Buffer using the provided backing slice.
// The buffer starts with the backing slice's length and capacity.
func NewBuffer[T any](backing []T) *Linear[T] {
	return &Linear[T]{buf: backing}
}

// Write appends the contents of p to the buffer, growing the buffer as needed.
// Returns the number of elements written, which is always len(p) and a nil error.
func (b *Linear[T]) Write(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	b.grow(len(p))
	n = copy(b.buf[len(b.buf):cap(b.buf)], p)
	if n < len(p) {
		// Need to extend the slice
		b.buf = append(b.buf, p[n:]...)
		n = len(p)
	} else {
		b.buf = b.buf[:len(b.buf)+n]
	}
	return n, nil
}

// Read reads the next len(p) elements from the buffer or until the buffer is drained.
// Returns the number of elements read and io.EOF if the buffer is empty.
func (b *Linear[T]) Read(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if b.off >= len(b.buf) {
		return 0, io.EOF
	}
	n = copy(p, b.buf[b.off:])
	b.off += n
	return n, nil
}

// WriteItem appends a single element to the buffer.
func (b *Linear[T]) WriteItem(c T) error {
	b.grow(1)
	if len(b.buf) < cap(b.buf) {
		b.buf = b.buf[:len(b.buf)+1]
		b.buf[len(b.buf)-1] = c
	} else {
		b.buf = append(b.buf, c)
	}
	return nil
}

// ReadItem reads and returns the next element from the buffer.
// Returns io.EOF if the buffer is empty.
func (b *Linear[T]) ReadItem() (T, error) {
	var zero T
	if b.off >= len(b.buf) {
		return zero, io.EOF
	}
	c := b.buf[b.off]
	b.off++
	return c, nil
}

// Len returns the number of unread elements in the buffer.
func (b *Linear[T]) Len() int {
	return len(b.buf) - b.off
}

// Cap returns the capacity of the buffer's underlying slice.
func (b *Linear[T]) Cap() int {
	return cap(b.buf)
}

// Reset resets the buffer to be empty, but it retains the underlying storage.
func (b *Linear[T]) Reset() {
	b.buf = b.buf[:0]
	b.off = 0
}

// Grow grows the buffer's capacity, if necessary, to guarantee space for
// another n elements. After Grow(n), at least n elements can be written to the
// buffer without another allocation.
func (b *Linear[T]) Grow(n int) {
	if n < 0 {
		panic("Buffer.Grow: negative count")
	}
	b.grow(n)
}

// grow ensures the buffer has capacity for at least n more elements.
func (b *Linear[T]) grow(n int) {
	if cap(b.buf)-len(b.buf) >= n {
		return
	}
	// Double the capacity, but ensure we have at least n more elements
	newCap := 2*cap(b.buf) + n
	if newCap < 4 {
		newCap = 4
	}
	newBuf := make([]T, len(b.buf), newCap)
	copy(newBuf, b.buf)
	b.buf = newBuf
}

// Truncate discards all but the first n unread elements from the buffer
// but continues to use the same allocated storage.
// It panics if n is negative or greater than the length of the buffer.
func (b *Linear[T]) Truncate(n int) {
	if n < 0 || n > b.Len() {
		panic("Buffer.Truncate: invalid count")
	}
	b.buf = b.buf[:b.off+n]
}

// Items returns a slice of length b.Len() holding the unread portion of the buffer.
// The slice is valid for use only until the next buffer modification.
func (b *Linear[T]) Items() []T {
	return b.buf[b.off:]
}

// Next returns a slice containing the next n elements from the buffer,
// advancing the buffer as if the elements had been returned by Read.
// If there are fewer than n elements in the buffer, Next returns the entire buffer.
// The slice is valid only until the next call to a read or write method.
func (b *Linear[T]) Next(n int) []T {
	m := b.Len()
	if n > m {
		n = m
	}
	data := b.buf[b.off : b.off+n]
	b.off += n
	return data
}

// UnreadByte unreads the last element returned by the most recent successful
// read operation that returned at least one element.
func (b *Linear[T]) UnreadByte() error {
	if b.off == 0 {
		return io.ErrUnexpectedEOF
	}
	b.off--
	return nil
}

// Ensure Buffer[byte] implements io.Reader and io.Writer
var (
	_ io.Reader = (*Linear[byte])(nil)
	_ io.Writer = (*Linear[byte])(nil)
)
