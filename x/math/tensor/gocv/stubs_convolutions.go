package gocv

import "github.com/itohio/EasyRobot/x/math/tensor/types"

func (t Tensor) Conv1D(dst types.Tensor, kernel, bias types.Tensor, stride, padding int) types.Tensor {
	panicUnsupported("Conv1D")
	return nil
}

func (t Tensor) Conv2D(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	panicUnsupported("Conv2D")
	return nil
}

func (t Tensor) Conv2DTransposed(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	panicUnsupported("Conv2DTransposed")
	return nil
}

func (t Tensor) Conv2DKernelGrad(dst types.Tensor, outputGrad, kernel types.Tensor, stride, padding []int) types.Tensor {
	panicUnsupported("Conv2DKernelGrad")
	return nil
}

func (t Tensor) Conv1DKernelGrad(dst types.Tensor, outputGrad, kernel types.Tensor, stride, padding int) types.Tensor {
	panicUnsupported("Conv1DKernelGrad")
	return nil
}

func (t Tensor) Im2Col(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	panicUnsupported("Im2Col")
	return nil
}

func (t Tensor) Col2Im(dst types.Tensor, outputShape, kernelSize, stride, padding []int) types.Tensor {
	panicUnsupported("Col2Im")
	return nil
}
