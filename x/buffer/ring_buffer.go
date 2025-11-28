package buffer

import (
	"errors"
	"io"
	"sync"
)

var _ Writer[byte] = (*Ring[byte])(nil)
var _ Reader[byte] = (*Ring[byte])(nil)
var _ Initializer = (*Ring[byte])(nil)

// Ring is a generic ring buffer that uses a provided backing slice.
// When instantiated with byte type, it implements io.Reader and io.Writer.
// The buffer uses a circular array pattern with head (write) and tail (read) positions.
// Ring is thread-safe for concurrent read/write operations using mutex synchronization.
type Ring[T any] struct {
	mu   sync.RWMutex // Protects head, tail, and len
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
// Uses bulk copy operations for better performance.
// Returns the number of elements written, which may be less than len(p) if the buffer is full.
// Thread-safe: uses mutex for synchronization.
func (rb *Ring[T]) Write(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	capacity := cap(rb.buf)
	if capacity == 0 {
		return 0, io.ErrShortWrite
	}

	rb.mu.Lock()
	defer rb.mu.Unlock()

	// Check available space
	available := capacity - rb.len
	if available == 0 {
		return 0, io.ErrShortWrite
	}

	toWrite := len(p)
	if toWrite > available {
		toWrite = available
	}

	head := rb.head
	spaceUntilWrap := capacity - head

	if toWrite <= spaceUntilWrap {
		// Single contiguous copy
		copy(rb.buf[head:head+toWrite], p[:toWrite])
		head += toWrite
		if head >= capacity {
			head = 0
		}
		rb.head = head
		rb.len += toWrite
		return toWrite, nil
	}

	// Two-part copy (wraps around)
	firstPart := spaceUntilWrap
	secondPart := toWrite - firstPart
	copy(rb.buf[head:], p[:firstPart])
	copy(rb.buf[:secondPart], p[firstPart:toWrite])
	rb.head = secondPart
	rb.len += toWrite
	return toWrite, nil
}

// Read reads elements from the ring buffer into p.
// Uses bulk copy operations for better performance.
// Returns the number of elements read, which may be less than len(p) if the buffer is empty.
// Returns io.EOF when the buffer is empty or when fewer items are read than requested.
// Thread-safe: uses mutex for synchronization.
func (rb *Ring[T]) Read(p []T) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	capacity := cap(rb.buf)
	if capacity == 0 {
		return 0, io.EOF
	}

	rb.mu.Lock()
	defer rb.mu.Unlock()

	// Check if buffer is empty
	if rb.len == 0 {
		return 0, io.EOF
	}

	toRead := len(p)
	if toRead > rb.len {
		toRead = rb.len
	}

	tail := rb.tail
	availableUntilWrap := capacity - tail

	if toRead <= availableUntilWrap {
		// Single contiguous copy
		copy(p[:toRead], rb.buf[tail:tail+toRead])
		tail += toRead
		if tail >= capacity {
			tail = 0
		}
		rb.tail = tail
		rb.len -= toRead
		if toRead < len(p) {
			return toRead, io.EOF
		}
		return toRead, nil
	}

	// Two-part copy (wraps around)
	firstPart := availableUntilWrap
	secondPart := toRead - firstPart
	copy(p[:firstPart], rb.buf[tail:])
	copy(p[firstPart:toRead], rb.buf[:secondPart])
	rb.tail = secondPart
	rb.len -= toRead
	if toRead < len(p) {
		return toRead, io.EOF
	}
	return toRead, nil
}

// WriteItem writes a single element to the ring buffer.
// Returns an error if the buffer is full.
// Thread-safe: uses mutex.
func (rb *Ring[T]) WriteItem(c T) error {
	capacity := cap(rb.buf)
	if capacity == 0 {
		return io.ErrShortWrite
	}

	rb.mu.Lock()
	defer rb.mu.Unlock()

	// Check if buffer is full
	if rb.len >= capacity {
		return io.ErrShortWrite
	}

	rb.buf[rb.head] = c
	rb.head++
	if rb.head >= capacity {
		rb.head = 0
	}
	rb.len++
	return nil
}

// ReadItem reads a single element from the ring buffer.
// Returns an error if the buffer is empty.
// Thread-safe: uses mutex.
func (rb *Ring[T]) ReadItem() (T, error) {
	var zero T
	capacity := cap(rb.buf)
	if capacity == 0 {
		return zero, io.EOF
	}

	rb.mu.Lock()
	defer rb.mu.Unlock()

	// Check if buffer is empty
	if rb.len == 0 {
		return zero, io.EOF
	}

	c := rb.buf[rb.tail]
	rb.tail++
	if rb.tail >= capacity {
		rb.tail = 0
	}
	rb.len--
	return c, nil
}

// Len returns the number of elements currently in the buffer.
// Thread-safe: uses read lock.
func (rb *Ring[T]) Len() int {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return rb.len
}

// Cap returns the capacity of the ring buffer.
// Thread-safe: reads from cap(buf) which is constant.
func (rb *Ring[T]) Cap() int {
	return cap(rb.buf)
}

// Available returns the number of elements that can be written without overwriting unread data.
// Thread-safe: uses read lock.
func (rb *Ring[T]) Available() int {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	capacity := cap(rb.buf)
	return capacity - rb.len
}

// Reset clears the buffer by resetting head, tail positions and length.
// Thread-safe: uses mutex.
func (rb *Ring[T]) Reset() {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	rb.head = 0
	rb.tail = 0
	rb.len = 0
}

// Full returns true if the buffer is full (no space available).
// Thread-safe: uses read lock.
func (rb *Ring[T]) Full() bool {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	capacity := cap(rb.buf)
	if capacity == 0 {
		return true
	}
	return rb.len >= capacity
}

// Empty returns true if the buffer is empty.
// Thread-safe: uses read lock.
func (rb *Ring[T]) Empty() bool {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return rb.len == 0
}

// capacity returns the capacity of the backing buffer.
func (rb *Ring[T]) capacity() int {
	return cap(rb.buf)
}

// available returns the number of available slots.
// Must be called with lock held.
func (rb *Ring[T]) available() int {
	capacity := cap(rb.buf)
	return capacity - rb.len
}

// Grow grows the buffer's capacity to guarantee space for n more elements.
// After Grow(n), at least n elements can be written without another allocation.
// Note: Growing a ring buffer requires reallocating and copying existing data.
//
// WARNING: Grow is NOT thread-safe. It must not be called concurrently with
// Read or Write operations. Use external synchronization if needed.
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
	rb.mu.Lock()
	currentLen := rb.len
	readPos := rb.tail
	rb.mu.Unlock()

	if currentLen > 0 {
		oldCap := currentCap
		toRead := currentLen

		// Copy in chunks (handle wrapping)
		firstChunk := min(toRead, oldCap-readPos)
		copy(newBuf, rb.buf[readPos:readPos+firstChunk])

		if toRead > firstChunk {
			secondChunk := toRead - firstChunk
			copy(newBuf[firstChunk:], rb.buf[:secondChunk])
		}
	}

	// Update buffer and positions
	// Note: Not thread-safe - caller must ensure no concurrent access
	rb.mu.Lock()
	rb.buf = newBuf
	rb.tail = 0
	rb.head = currentLen
	rb.len = currentLen
	rb.mu.Unlock()
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
