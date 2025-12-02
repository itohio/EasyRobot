package types

import (
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// ImageWithMask holds an image tensor and a mask tensor.
// Used for processors that produce masks (histogram backprojection, background separation).
type ImageWithMask struct {
	Image types.Tensor
	Mask  types.Tensor // Grayscale mask (0-255)
}

// ImageWithMasks holds an image tensor and multiple mask tensors.
// Used when multiple masks need to be produced (e.g., multiple histogram backprojections).
type ImageWithMasks struct {
	Image types.Tensor
	Masks []types.Tensor // Multiple masks
}
