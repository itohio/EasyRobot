package gocv

import "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"

func (t Tensor) Sum(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("Sum")
	return nil
}

func (t Tensor) ReduceSum(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("ReduceSum")
	return nil
}

func (t Tensor) Mean(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("Mean")
	return nil
}

func (t Tensor) ReduceMean(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("ReduceMean")
	return nil
}

func (t Tensor) Max(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("Max")
	return nil
}

func (t Tensor) ReduceMax(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("ReduceMax")
	return nil
}

func (t Tensor) Min(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("Min")
	return nil
}

func (t Tensor) ReduceMin(dst types.Tensor, dims []int) types.Tensor {
	panicUnsupported("ReduceMin")
	return nil
}

func (t Tensor) ArgMax(dst types.Tensor, dim int) types.Tensor {
	panicUnsupported("ArgMax")
	return nil
}

func (t Tensor) ArgMin(dst types.Tensor, dim int) types.Tensor {
	panicUnsupported("ArgMin")
	return nil
}

func (t Tensor) MatMul(dst types.Tensor, other types.Tensor) types.Tensor {
	panicUnsupported("MatMul")
	return nil
}

func (t Tensor) MatMulTransposed(dst types.Tensor, other types.Tensor, transposeA, transposeB bool) types.Tensor {
	panicUnsupported("MatMulTransposed")
	return nil
}

func (t Tensor) MatVecMulTransposed(dst types.Tensor, matrix, vector types.Tensor, alpha, beta float64) types.Tensor {
	panicUnsupported("MatVecMulTransposed")
	return nil
}

func (t Tensor) Dot(other types.Tensor) float64 {
	panicUnsupported("Dot")
	return 0
}

func (t Tensor) Tensordot(other types.Tensor) float64 {
	panicUnsupported("Tensordot")
	return 0
}

func (t Tensor) Norm(ord int) float64 {
	panicUnsupported("Norm")
	return 0
}

func (t Tensor) L2Normalize(dst types.Tensor, dim int) types.Tensor {
	panicUnsupported("L2Normalize")
	return nil
}

func (t Tensor) Normalize(dst types.Tensor, dim int) types.Tensor {
	panicUnsupported("Normalize")
	return nil
}

func (t Tensor) AddScaled(dst types.Tensor, other types.Tensor, alpha float64) types.Tensor {
	panicUnsupported("AddScaled")
	return nil
}

func (t Tensor) ScatterAdd(dst types.Tensor, index, value types.Tensor) types.Tensor {
	panicUnsupported("ScatterAdd")
	return nil
}

