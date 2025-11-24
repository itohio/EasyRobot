package chain

import "io"

// sizeReader is a helper that reports a fixed size for size calculations.
// Used by Chain and Adapter when calculating theoretical output sizes.
type sizeReader[T any] struct {
	size int
}

// Read implements buffer.Reader interface.
// Always returns EOF immediately - this reader is only used for size calculations.
func (r *sizeReader[T]) Read(p []T) (n int, err error) {
	return 0, io.EOF
}

// Len returns the reported size (if the reader implements Len interface).
// This allows size calculations to work with processors that check source length.
func (r *sizeReader[T]) Len() int {
	return r.size
}
