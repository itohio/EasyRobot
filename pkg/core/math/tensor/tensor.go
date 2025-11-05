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

// Empty creates an empty tensor of given data type.
func EmptyAs(t types.Tensor) eager_tensor.Tensor {
	return eager_tensor.EmptyAs(t)
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
