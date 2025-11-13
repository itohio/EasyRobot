package eager_tensor

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// XavierUniform creates a new tensor with the given data type and shape,
// and initializes it with Xavier/Glorot uniform initialization.
// For weights, limit = sqrt(6 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierUniform(dtype types.DataType, shape types.Shape, fanIn, fanOut int, rng types.RNG) types.Tensor {
	if rng == nil || shape == nil || len(shape) == 0 {
		return nil
	}

	result := New(dtype, shape)
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	for elem := range result.Elements() {
		elem.Set((rng.Float64()*2 - 1) * limit)
	}
	return result
}

// XavierNormal creates a new tensor with the given data type and shape,
// and initializes it with Xavier/Glorot normal initialization.
// For weights, stddev = sqrt(2 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierNormal(dtype types.DataType, shape types.Shape, fanIn, fanOut int, rng types.RNG) types.Tensor {
	if rng == nil || shape == nil || len(shape) == 0 {
		return nil
	}

	result := New(dtype, shape)
	stddev := math.Sqrt(2.0 / float64(fanIn+fanOut))
	for elem := range result.Elements() {
		elem.Set(rng.NormFloat64() * stddev)
	}
	return result
}

// XavierUniformLike creates a new tensor with the same data type and shape as the reference tensor,
// and initializes it with Xavier/Glorot uniform initialization.
// For weights, limit = sqrt(6 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierUniformLike(ref types.Tensor, fanIn, fanOut int, rng types.RNG) types.Tensor {
	if ref == nil || rng == nil {
		return nil
	}
	return XavierUniform(ref.DataType(), ref.Shape(), fanIn, fanOut, rng)
}

// XavierNormalLike creates a new tensor with the same data type and shape as the reference tensor,
// and initializes it with Xavier/Glorot normal initialization.
// For weights, stddev = sqrt(2 / (fanIn + fanOut))
// Returns a new initialized tensor.
func XavierNormalLike(ref types.Tensor, fanIn, fanOut int, rng types.RNG) types.Tensor {
	if ref == nil || rng == nil {
		return nil
	}
	return XavierNormal(ref.DataType(), ref.Shape(), fanIn, fanOut, rng)
}
