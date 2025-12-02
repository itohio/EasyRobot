package format

import (
	"image"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// ImageWithTensor holds an image.Image and its source tensor.
// Used for format conversion where we need to preserve both representations.
type ImageWithTensor struct {
	Image  image.Image
	Source types.Tensor // Optional: reference to source tensor if available
}

// StereoPair holds a pair of tensors for stereo processing.
type StereoPair struct {
	Left  types.Tensor
	Right types.Tensor
}
