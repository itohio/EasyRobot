package types

// Core defines the core tensor properties and metadata access.
// This interface contains fundamental operations for tensor identity and element access.
type Core interface {
	// ID returns a unique identifier for the tensor.
	ID() uintptr

	// DataType returns the tensor's data type (e.g., DTFP32, DTINT8).
	DataType() DataType

	// Data returns the underlying data storage as any.
	// For FP32 tensors, returns []float32. For INT8 tensors, returns []int8.
	// Use DataType() to determine the actual type before type assertion.
	// Direct data access bypasses tensor abstractions; prefer Elements() for iteration when possible.
	Data() any

	// Shape returns a copy of the tensor's shape (dimensions).
	Shape() Shape

	// Rank returns the number of dimensions in the tensor.
	Rank() int

	// Size returns the total number of elements in the tensor.
	Size() int

	// Empty returns true if the tensor is empty (no shape or data).
	Empty() bool

	// Strides returns the tensor's strides, computing if necessary.
	// If dst is provided and has sufficient capacity, strides are written to dst.
	// If dst is nil or too small, a stack-allocated array may be used internally.
	// Returns the slice containing the strides.
	//
	// For contiguous tensors (strides == nil), computes canonical row-major strides from shape.
	// For non-contiguous tensors, returns stored strides.
	// Returns nil for empty tensors or scalars (rank 0).
	Strides(dst []int) []int

	// IsContiguous reports whether the tensor is contiguous (dense row-major layout).
	// A tensor is contiguous if:
	// - strides == nil (always contiguous, uses canonical strides)
	// - OR stored strides match canonical row-major strides for the shape
	IsContiguous() bool

	// Offset returns the base offset into the data slice.
	// Returns 0 for tensors that start at the beginning of data.
	// Non-zero offset is used for views (e.g., slices) that reference a portion of larger array.
	Offset() int

	// DataWithOffset returns the data slice adjusted by the tensor's offset.
	// For tensors with offset == 0, returns the data as-is.
	// For tensors with offset > 0, returns a slice starting at the offset.
	// This is a type-agnostic helper that works with all data types.
	// Returns nil if the tensor has no data or if offset is out of bounds.
	DataWithOffset() any

	// At returns the element at the given multi-dimensional indices.
	// When only one index is provided and tensor rank > 1, uses linear indexing (direct data access).
	// Otherwise, indices must match the tensor's dimensions for multi-dimensional access.
	// Panics if indices are out of bounds or incorrect number of indices provided.
	At(indices ...int) float64

	// SetAt sets the element at the given multi-dimensional indices to the specified value.
	// When only one index is provided and tensor rank > 1, uses linear indexing (direct data access).
	// Otherwise, indices must match the tensor's dimensions for multi-dimensional access.
	// Panics if indices are out of bounds or incorrect number of indices provided.
	SetAt(value float64, indices ...int)

	// Elements creates an iterator over tensor elements (Go 1.22+ range-over-function).
	// Returns a function that can be used in range loops.
	// fixedAxisValuePairs are pairs of axis index and fixed value: axis1, value1, axis2, value2, ...
	// If no pairs provided, iterates over all elements. Iterates in row-major order.
	// Returns Element objects with Get() and Set() methods for element access.
	Elements(fixedAxisValuePairs ...int) func(func(Element) bool)

	// Release allows tensors backed by pooled storage to return their buffers.
	// Implementations may no-op when no pooling is used or when tensor views
	// share storage. Calling Release multiple times should be safe.
	Release()
}
