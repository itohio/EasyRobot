// go:build linux
package v4l

import (
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// PooledTensor wraps uint8 data with buffer pool integration
type PooledTensor struct {
	data     []uint8
	pool     *helpers.Pool[uint8]
	width    int
	height   int
	channels int
}

// NewPooledTensor creates a tensor from pooled buffer
func NewPooledTensor(pool *helpers.Pool[uint8], width, height, channels int) *PooledTensor {
	size := width * height * channels
	data := pool.Get(size)

	return &PooledTensor{
		data:     data[:size],
		pool:     pool,
		width:    width,
		height:   height,
		channels: channels,
	}
}

// NewPooledTensorFromData creates a tensor from existing data with pool integration
func NewPooledTensorFromData(pool *helpers.Pool[uint8], data []uint8, width, height, channels int) *PooledTensor {
	return &PooledTensor{
		data:     data,
		pool:     pool,
		width:    width,
		height:   height,
		channels: channels,
	}
}

// ID returns a unique identifier for the tensor
func (t *PooledTensor) ID() uintptr {
	return uintptr(t.DataAddr())
}

// DataType returns the tensor's data type (UINT8)
func (t *PooledTensor) DataType() tensortypes.DataType {
	return tensortypes.UINT8
}

// Data returns the underlying data storage as any ([]uint8)
func (t *PooledTensor) Data() any {
	return t.data
}

// Shape returns a copy of the tensor's shape [height, width, channels]
func (t *PooledTensor) Shape() tensortypes.Shape {
	return tensortypes.NewShape(t.height, t.width, t.channels)
}

// Rank returns the number of dimensions (3 for HWC format)
func (t *PooledTensor) Rank() int {
	return 3
}

// Size returns the total number of elements
func (t *PooledTensor) Size() int {
	return t.width * t.height * t.channels
}

// Empty returns true if the tensor is empty
func (t *PooledTensor) Empty() bool {
	return len(t.data) == 0 || t.width == 0 || t.height == 0 || t.channels == 0
}

// Strides returns the tensor's strides for row-major layout
func (t *PooledTensor) Strides(dst []int) []int {
	if t.Empty() {
		return nil
	}

	strides := make([]int, 3)
	strides[2] = 1                    // channel stride
	strides[1] = t.channels           // width stride
	strides[0] = t.channels * t.width // height stride

	if dst != nil && len(dst) >= 3 {
		copy(dst, strides)
		return dst[:3]
	}
	return strides
}

// IsContiguous reports whether the tensor is contiguous (always true for pooled tensors)
func (t *PooledTensor) IsContiguous() bool {
	return true
}

// Offset returns the base offset (always 0 for pooled tensors)
func (t *PooledTensor) Offset() int {
	return 0
}

// DataWithOffset returns the data slice (same as Data() for pooled tensors)
func (t *PooledTensor) DataWithOffset() any {
	return t.data
}

// At returns the element at the given indices (HWC format)
func (t *PooledTensor) At(indices ...int) float64 {
	if len(indices) != 3 {
		panic("PooledTensor.At: requires exactly 3 indices (height, width, channel)")
	}

	h, w, c := indices[0], indices[1], indices[2]
	if h < 0 || h >= t.height || w < 0 || w >= t.width || c < 0 || c >= t.channels {
		panic("PooledTensor.At: index out of bounds")
	}

	idx := h*t.width*t.channels + w*t.channels + c
	return float64(t.data[idx])
}

// SetAt sets the element at the given indices to the specified value (HWC format)
func (t *PooledTensor) SetAt(value float64, indices ...int) {
	if len(indices) != 3 {
		panic("PooledTensor.SetAt: requires exactly 3 indices (height, width, channel)")
	}

	h, w, c := indices[0], indices[1], indices[2]
	if h < 0 || h >= t.height || w < 0 || w >= t.width || c < 0 || c >= t.channels {
		panic("PooledTensor.SetAt: index out of bounds")
	}

	idx := h*t.width*t.channels + w*t.channels + c
	t.data[idx] = uint8(value)
}

// Elements creates an iterator over tensor elements (Go 1.22+ range-over-function)
func (t *PooledTensor) Elements(fixedAxisValuePairs ...int) func(func(tensortypes.Element) bool) {
	// For now, return a simple iterator over all elements
	// A full implementation would handle fixed axis values
	return func(yield func(tensortypes.Element) bool) {
		for i := 0; i < len(t.data); i++ {
			h := i / (t.width * t.channels)
			w := (i % (t.width * t.channels)) / t.channels
			c := i % t.channels

			elem := &pooledElement{
				tensor: t,
				index:  i,
				coords: [3]int{h, w, c},
			}

			if !yield(elem) {
				return
			}
		}
	}
}

// Release returns the buffer to the pool
func (t *PooledTensor) Release() {
	if t.data != nil && t.pool != nil {
		// Return the underlying buffer to the pool
		t.pool.Put(t.data)
		t.data = nil
	}
}

// DataAddr returns the address of the data slice for ID generation
func (t *PooledTensor) DataAddr() *uint8 {
	if len(t.data) == 0 {
		return nil
	}
	return &t.data[0]
}

// pooledElement implements tensortypes.Element for pooled tensors
type pooledElement struct {
	tensor *PooledTensor
	index  int
	coords [3]int
}

func (e *pooledElement) Get() float64 {
	return float64(e.tensor.data[e.index])
}

func (e *pooledElement) Set(value float64) {
	e.tensor.data[e.index] = uint8(value)
}

// defaultTensorFactory creates pooled tensors
func defaultTensorFactory(pool *helpers.Pool[uint8]) func([]uint8, int, int, int) types.Tensor {
	return func(data []uint8, width, height, channels int) types.Tensor {
		return NewPooledTensorFromData(pool, data, width, height, channels)
	}
}
