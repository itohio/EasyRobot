package tensor

// DataType represents the underlying element type stored by a tensor.
type DataType uint8

const (
	// DTFP32 represents 32-bit floating point tensors (default).
	DTFP32 DataType = iota
)
