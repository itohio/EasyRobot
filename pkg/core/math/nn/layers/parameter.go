package layers

import (
	"math"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Parameter represents a trainable parameter (weight or bias).
type Parameter struct {
	Data         types.Tensor // Parameter values
	Grad         types.Tensor // Gradients (lazy allocation)
	RequiresGrad bool         // Whether to compute gradients (deprecated, use layer CanLearn)
}

// ZeroGrad zeros the gradient tensor.
// Allocates gradient tensor if needed and zeroes all values.
func (p *Parameter) ZeroGrad() {
	if p == nil || !p.RequiresGrad {
		return
	}

	if p.Grad.Shape().Rank() == 0 {
		// Lazy allocation: create gradient tensor with same shape as data
		p.Grad = tensor.New(tensor.DTFP32, p.Data.Shape())
	}

	// Zero all gradient values
	p.Grad.Scale(0)
}

// InitXavier initializes parameter with Xavier/Glorot uniform initialization.
// For weights, limit = sqrt(6 / (fanIn + fanOut))
// For biases, limit = fanOut
func InitXavier(param *Parameter, fanIn, fanOut int, rng RNG) {
	if param == nil || rng == nil {
		return
	}

	limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
	for idx := range param.Data.Shape().Iterator() {
		// Convert float32 to float64 for SetAt interface
		param.Data.SetAt(float64((rng.Float32()*2-1)*limit), idx...)
	}
}

// InitXavierNormal initializes parameter with Xavier/Glorot normal initialization.
// For weights, stddev = sqrt(2 / (fanIn + fanOut))
type RNG interface {
	NormFloat64() float64
	Float32() float32
}

func InitXavierNormal(param *Parameter, fanIn, fanOut int, rng RNG) {
	if param == nil || rng == nil {
		return
	}

	stddev := float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))

	for idx := range param.Data.Shape().Iterator() {
		// Convert float32 to float64 for SetAt interface
		param.Data.SetAt(float64(float32(rng.NormFloat64())*stddev), idx...)
	}
}
