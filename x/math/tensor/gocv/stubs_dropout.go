package gocv

import "github.com/itohio/EasyRobot/x/math/tensor/types"

func (t Tensor) DropoutForward(dst types.Tensor, mask types.Tensor) types.Tensor {
	panicUnsupported("DropoutForward")
	return nil
}

func (t Tensor) DropoutMask(p, scale float64, rng types.RNG) types.Tensor {
	panicUnsupported("DropoutMask")
	return nil
}

func (t Tensor) DropoutBackward(dst types.Tensor, gradOutput, mask types.Tensor) types.Tensor {
	panicUnsupported("DropoutBackward")
	return nil
}
