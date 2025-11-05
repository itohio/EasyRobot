package tensor

import (
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

type Tensor = types.Tensor
type Shape = types.Shape
type DataType = types.DataType

const (
	DT_UNKNOWN DataType = types.DT_UNKNOWN
	DTINT48    DataType = types.DTINT48
	DTINT8     DataType = types.DTINT8
	DTINT16    DataType = types.DTINT16
	DTFP16     DataType = types.DTFP16
	DTFP32     DataType = types.DTFP32
	DTFP64     DataType = types.DTFP64
)

func NewShape(dimensions ...int) Shape {
	return types.NewShape(dimensions...)
}

func New(dtype types.DataType, shape types.Shape) eager_tensor.Tensor {
	return eager_tensor.New(dtype, shape)
}

// NewAs creates a new tensor with the same data type and shape as the given tensor.
func NewAs(t types.Tensor) types.Tensor {
	return eager_tensor.NewAs(t)
}

// Empty creates an empty tensor of given data type.
func Empty(dt types.DataType) eager_tensor.Tensor {
	return eager_tensor.Empty(dt)
}

// EmptyLike creates an empty tensor with the same data type as the given tensor.
func EmptyLike(t types.Tensor) eager_tensor.Tensor {
	return eager_tensor.EmptyLike(t)
}

func FromArray[T types.DataElementType](shape types.Shape, data []T) eager_tensor.Tensor {
	return eager_tensor.FromArray(shape, data)
}

// FromFloat32 constructs an FP32 tensor from an existing backing slice.
// If data is nil, a new buffer is allocated. The slice is used directly (no copy).
func FromFloat32(shape types.Shape, data []float32) eager_tensor.Tensor {
	return eager_tensor.FromFloat32(shape, data)
}

// ZerosLike creates a new tensor with the same shape as t, filled with zeros.
func ZerosLike(t types.Tensor) types.Tensor {
	return eager_tensor.ZerosLike(t)
}

// OnesLike creates a new tensor with the same shape as t, filled with ones.
func OnesLike(t types.Tensor) types.Tensor {
	return eager_tensor.OnesLike(t)
}

// FullLike creates a new tensor with the same shape as t, filled with the given value.
func FullLike(t types.Tensor, value float32) types.Tensor {
	return eager_tensor.FullLike(t, value)
}

// IsNil checks whether the tensor is nil or empty.
// Returns true if t is nil, or if t.Shape() is nil, or if t.Empty() is true.
func IsNil(t types.Tensor) bool {
	return eager_tensor.IsNil(t)
}

// RNG interface for random number generation used in initialization.
// This extends types.RNG with NormFloat64() for normal distribution initialization.
type RNG = types.RNG

// XavierUniform creates a new tensor with the given data type and shape,
// and initializes it with Xavier/Glorot uniform initialization.
// For weights, limit = sqrt(6 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierUniform(dtype DataType, shape Shape, fanIn, fanOut int, rng RNG) Tensor {
	return eager_tensor.XavierUniform(dtype, shape, fanIn, fanOut, rng)
}

// XavierNormal creates a new tensor with the given data type and shape,
// and initializes it with Xavier/Glorot normal initialization.
// For weights, stddev = sqrt(2 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierNormal(dtype DataType, shape Shape, fanIn, fanOut int, rng RNG) Tensor {
	return eager_tensor.XavierNormal(dtype, shape, fanIn, fanOut, rng)
}

// XavierUniformLike creates a new tensor with the same data type and shape as the reference tensor,
// and initializes it with Xavier/Glorot uniform initialization.
// For weights, limit = sqrt(6 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierUniformLike(ref Tensor, fanIn, fanOut int, rng RNG) Tensor {
	return eager_tensor.XavierUniformLike(ref, fanIn, fanOut, rng)
}

// XavierNormalLike creates a new tensor with the same data type and shape as the reference tensor,
// and initializes it with Xavier/Glorot normal initialization.
// For weights, stddev = sqrt(2 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierNormalLike(ref Tensor, fanIn, fanOut int, rng RNG) Tensor {
	return eager_tensor.XavierNormalLike(ref, fanIn, fanOut, rng)
}
