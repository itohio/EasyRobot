package destination

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

// TensorToMat converts a tensor to gocv.Mat.
// This is an exported helper function for use by other packages.
func TensorToMat(tensor types.Tensor) (cv.Mat, error) {
	return tensorToMat(tensor)
}

// tensorToMat converts a tensor to gocv.Mat
func tensorToMat(tensor types.Tensor) (cv.Mat, error) {
	accessor, ok := tensor.(tensorgocv.Accessor)
	if !ok {
		return cv.Mat{}, fmt.Errorf("tensor does not expose GoCV accessor (expected tensorgocv.Accessor)")
	}
	mat, err := accessor.MatClone()
	if err != nil {
		return cv.Mat{}, err
	}
	return mat, nil
}

