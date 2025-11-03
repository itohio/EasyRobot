package nn

import (
	"math"
	"math/rand"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Parameter represents a trainable parameter (weight or bias).
type Parameter struct {
	Data         tensor.Tensor // Parameter values
	Grad         tensor.Tensor // Gradients (lazy allocation)
	RequiresGrad bool          // Whether to compute gradients (deprecated, use layer CanLearn)
}

// ZeroGrad zeros the gradient tensor.
// Allocates gradient tensor if needed and zeroes all values.
func (p *Parameter) ZeroGrad() {
	if p == nil || !p.RequiresGrad {
		return
	}

	if len(p.Grad.Dim) == 0 {
		// Lazy allocation: create gradient tensor with same shape as data
		p.Grad = tensor.Tensor{
			Dim:  make([]int, len(p.Data.Dim)),
			Data: make([]float32, p.Data.Size()),
		}
		copy(p.Grad.Dim, p.Data.Dim)
	}

	// Zero all gradient values
	for i := range p.Grad.Data {
		p.Grad.Data[i] = 0
	}
}

// InitXavier initializes parameter with Xavier/Glorot uniform initialization.
// For weights, limit = sqrt(6 / (fanIn + fanOut))
// For biases, limit = fanOut
func InitXavier(param *Parameter, fanIn, fanOut int, rng *rand.Rand) {
	if param == nil || rng == nil {
		return
	}

	limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))

	for i := range param.Data.Data {
		param.Data.Data[i] = (rng.Float32()*2 - 1) * limit
	}
}

// InitXavierNormal initializes parameter with Xavier/Glorot normal initialization.
// For weights, stddev = sqrt(2 / (fanIn + fanOut))
func InitXavierNormal(param *Parameter, fanIn, fanOut int, rng *rand.Rand) {
	if param == nil || rng == nil {
		return
	}

	stddev := float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))

	for i := range param.Data.Data {
		param.Data.Data[i] = float32(rng.NormFloat64()) * stddev
	}
}
