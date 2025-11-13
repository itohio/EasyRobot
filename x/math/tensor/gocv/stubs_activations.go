package gocv

import "github.com/itohio/EasyRobot/x/math/tensor/types"

func (t Tensor) ReLU(dst types.Tensor) types.Tensor {
	panicUnsupported("ReLU")
	return nil
}

func (t Tensor) Sigmoid(dst types.Tensor) types.Tensor {
	panicUnsupported("Sigmoid")
	return nil
}

func (t Tensor) Tanh(dst types.Tensor) types.Tensor {
	panicUnsupported("Tanh")
	return nil
}

func (t Tensor) Softmax(dim int, dst types.Tensor) types.Tensor {
	panicUnsupported("Softmax")
	return nil
}

func (t Tensor) ReLU6(dst types.Tensor) types.Tensor {
	panicUnsupported("ReLU6")
	return nil
}

func (t Tensor) LeakyReLU(dst types.Tensor, alpha float64) types.Tensor {
	panicUnsupported("LeakyReLU")
	return nil
}

func (t Tensor) ELU(dst types.Tensor, alpha float64) types.Tensor {
	panicUnsupported("ELU")
	return nil
}

func (t Tensor) Softplus(dst types.Tensor) types.Tensor {
	panicUnsupported("Softplus")
	return nil
}

func (t Tensor) Swish(dst types.Tensor) types.Tensor {
	panicUnsupported("Swish")
	return nil
}

func (t Tensor) GELU(dst types.Tensor) types.Tensor {
	panicUnsupported("GELU")
	return nil
}

func (t Tensor) ReLUGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	panicUnsupported("ReLUGrad")
	return nil
}

func (t Tensor) SigmoidGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	panicUnsupported("SigmoidGrad")
	return nil
}

func (t Tensor) TanhGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	panicUnsupported("TanhGrad")
	return nil
}

func (t Tensor) SoftmaxGrad(dst types.Tensor, gradOutput types.Tensor, dim int) types.Tensor {
	panicUnsupported("SoftmaxGrad")
	return nil
}
