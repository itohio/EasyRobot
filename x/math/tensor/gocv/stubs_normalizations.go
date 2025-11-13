package gocv

import "github.com/itohio/EasyRobot/x/math/tensor/types"

func (t Tensor) BatchNormForward(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	panicUnsupported("BatchNormForward")
	return nil
}

func (t Tensor) BatchNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panicUnsupported("BatchNormGrad")
	return nil, nil, nil
}

func (t Tensor) LayerNormForward(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	panicUnsupported("LayerNormForward")
	return nil
}

func (t Tensor) LayerNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panicUnsupported("LayerNormGrad")
	return nil, nil, nil
}

func (t Tensor) RMSNormForward(dst types.Tensor, gamma types.Tensor, eps float64) types.Tensor {
	panicUnsupported("RMSNormForward")
	return nil
}

func (t Tensor) RMSNormGrad(gradInputDst, gradGammaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor) {
	panicUnsupported("RMSNormGrad")
	return nil, nil
}

func (t Tensor) InstanceNorm2D(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	panicUnsupported("InstanceNorm2D")
	return nil
}

func (t Tensor) InstanceNorm2DGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panicUnsupported("InstanceNorm2DGrad")
	return nil, nil, nil
}

func (t Tensor) GroupNormForward(dst types.Tensor, gamma, beta types.Tensor, numGroups int, eps float64) types.Tensor {
	panicUnsupported("GroupNormForward")
	return nil
}

func (t Tensor) GroupNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, numGroups int, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panicUnsupported("GroupNormGrad")
	return nil, nil, nil
}
