package tensor

import (
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

type Tensor = types.Tensor
type Shape = types.Shape
type DataType = types.DataType

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
