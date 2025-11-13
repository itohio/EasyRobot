package gocv

import "github.com/itohio/EasyRobot/x/math/tensor/types"

func (t Tensor) MaxPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	panicUnsupported("MaxPool2D")
	return nil
}

func (t Tensor) MaxPool2DWithIndices(dst types.Tensor, indicesDst types.Tensor, kernelSize, stride, padding []int) (types.Tensor, types.Tensor) {
	panicUnsupported("MaxPool2DWithIndices")
	return nil, nil
}

func (t Tensor) MaxPool2DBackward(dst types.Tensor, gradOutput, indices types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	panicUnsupported("MaxPool2DBackward")
	return nil
}

func (t Tensor) AvgPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	panicUnsupported("AvgPool2D")
	return nil
}

func (t Tensor) AvgPool2DBackward(dst types.Tensor, gradOutput types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	panicUnsupported("AvgPool2DBackward")
	return nil
}

func (t Tensor) GlobalAvgPool2D(dst types.Tensor) types.Tensor {
	panicUnsupported("GlobalAvgPool2D")
	return nil
}

func (t Tensor) AdaptiveAvgPool2D(dst types.Tensor, outputSize []int) types.Tensor {
	panicUnsupported("AdaptiveAvgPool2D")
	return nil
}
