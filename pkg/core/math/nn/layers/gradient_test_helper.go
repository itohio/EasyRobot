package layers

import (
	"math"

	nntypes "github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// numericalGradient computes numerical gradient using finite differences
// f: function that takes input and returns output
// input: input tensor
// epsilon: small perturbation value
// Returns: numerical gradient tensor
func numericalGradient(f func(types.Tensor) (types.Tensor, error), input types.Tensor, epsilon float64) (types.Tensor, error) {
	inputShape := input.Shape()
	inputSize := input.Size()

	// Create output gradient tensor
	grad := tensor.New(input.DataType(), inputShape)

	// Compute gradient for each element using finite differences
	for i := 0; i < inputSize; i++ {
		// Forward difference: f(x + epsilon) - f(x)
		inputPlus := input.Clone()
		val := input.At(i)
		inputPlus.SetAt(val+epsilon, i)

		fPlus, err := f(inputPlus)
		if err != nil {
			return nil, err
		}

		fOrig, err := f(input)
		if err != nil {
			return nil, err
		}

		// Compute gradient: (f(x + epsilon) - f(x)) / epsilon
		// For scalar output, use the single value
		// For tensor output, use sum of all elements
		var gradVal float64
		if fPlus.Size() == 1 {
			gradVal = (fPlus.At(0) - fOrig.At(0)) / epsilon
		} else {
			// Sum all elements of the output
			sumPlusTensor := fPlus.Sum()
			sumOrigTensor := fOrig.Sum()
			sumPlus := sumPlusTensor.At(0)
			sumOrig := sumOrigTensor.At(0)
			gradVal = (sumPlus - sumOrig) / epsilon
		}

		grad.SetAt(gradVal, i)
	}

	return grad, nil
}

// checkGradientAccuracy compares analytical gradient with numerical gradient
// analyticalGrad: gradient computed by backward pass
// numericalGrad: gradient computed using finite differences
// tolerance: allowed difference between gradients
// Returns: true if gradients match within tolerance
func checkGradientAccuracy(analyticalGrad, numericalGrad types.Tensor, tolerance float64) (bool, float64, error) {
	if analyticalGrad.Size() != numericalGrad.Size() {
		return false, 0, nil
	}

	maxDiff := 0.0
	for i := 0; i < analyticalGrad.Size(); i++ {
		analytical := analyticalGrad.At(i)
		numerical := numericalGrad.At(i)

		diff := math.Abs(analytical - numerical)
		if diff > maxDiff {
			maxDiff = diff
		}

		// Check relative error for non-zero values
		if math.Abs(numerical) > 1e-8 {
			relDiff := diff / math.Abs(numerical)
			if relDiff > tolerance {
				return false, maxDiff, nil
			}
		} else if diff > tolerance {
			// For near-zero values, check absolute difference
			return false, maxDiff, nil
		}
	}

	return true, maxDiff, nil
}

// computeLoss creates a simple loss function that sums all elements
// This is useful for gradient checking
func computeLoss(output types.Tensor) float64 {
	sumTensor := output.Sum()
	return sumTensor.At(0)
}

// perturbParameter perturbs a parameter by epsilon at the given index
func perturbParameter(param *nntypes.Parameter, index int, epsilon float64) {
	val := param.Data.At(index)
	param.Data.SetAt(val+epsilon, index)
}

// restoreParameter restores a parameter value
func restoreParameter(param *nntypes.Parameter, index int, originalVal float64) {
	param.Data.SetAt(originalVal, index)
}
