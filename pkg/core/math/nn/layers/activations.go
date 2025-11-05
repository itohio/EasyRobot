package layers

import (
	"fmt"
	"math/rand"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// ReLU layer implements ReLU activation as a Layer.
type ReLU struct {
	Base
}

// NewReLU creates a new ReLU layer.
func NewReLU(name string) *ReLU {
	relu := &ReLU{
		Base: NewBase("relu"),
	}
	relu.Base.ParseOptions(WithName(name))
	return relu
}

// Init initializes the layer.
func (r *ReLU) Init(inputShape []int) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("ReLU.Init: empty input shape")
	}
	// Output shape is same as input for ReLU
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	r.Base.AllocOutput(inputShape, outputSize)
	return nil
}

// Forward computes ReLU: output = max(0, input).
func (r *ReLU) Forward(input types.Tensor) (types.Tensor, error) {
	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("ReLU.Forward: empty input")
	}

	// Store input
	r.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := r.Base.Output()
	if output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("ReLU.Forward: output not allocated, must call Init first")
	}

	// Apply ReLU directly to output
	output = input.ReLU(output)

	r.Base.StoreOutput(output)
	return output, nil
}

// Backward computes ReLU gradient: gradInput = gradOutput * (input > 0 ? 1 : 0).
func (r *ReLU) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if gradOutput == nil || gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("ReLU.Backward: empty gradOutput")
	}

	input := r.Base.Input()
	if input == nil || input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("ReLU.Backward: input not stored, must call Forward first")
	}

	// ReLU gradient using primitives: gradInput = gradOutput * (input > 0 ? 1 : 0)
	// Create mask: 1.0 where input > 0, 0.0 otherwise
	zeros := tensor.ZerosLike(input)
	mask := input.GreaterThan(zeros)

	// Element-wise multiply: gradOutput * mask
	gradInput := gradOutput.Clone().Mul(mask)

	r.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for ReLU).
func (r *ReLU) OutputShape(inputShape []int) ([]int, error) {
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, nil
}

// Sigmoid layer implements sigmoid activation as a Layer.
type Sigmoid struct {
	Base
}

// NewSigmoid creates a new Sigmoid layer.
func NewSigmoid(name string) *Sigmoid {
	sigmoid := &Sigmoid{
		Base: NewBase("sigmoid"),
	}
	sigmoid.Base.ParseOptions(WithName(name))
	return sigmoid
}

// Init initializes the layer.
func (s *Sigmoid) Init(inputShape []int) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Sigmoid.Init: empty input shape")
	}
	// Output shape is same as input for Sigmoid
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	s.Base.AllocOutput(inputShape, outputSize)
	return nil
}

// Forward computes sigmoid: output = 1 / (1 + exp(-input)).
func (s *Sigmoid) Forward(input types.Tensor) (types.Tensor, error) {
	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Sigmoid.Forward: empty input")
	}

	// Store input
	s.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := s.Base.Output()
	if output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Sigmoid.Forward: output not allocated, must call Init first")
	}

	// Apply sigmoid directly to output
	output = input.Sigmoid(output)

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes sigmoid gradient: gradInput = gradOutput * output * (1 - output).
func (s *Sigmoid) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if gradOutput == nil || gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Sigmoid.Backward: empty gradOutput")
	}

	output := s.Base.Output()
	if output == nil || output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Sigmoid.Backward: output not stored, must call Forward first")
	}

	// Sigmoid gradient using primitives: gradInput = gradOutput * output * (1 - output)
	ones := tensor.OnesLike(output)
	term1 := ones.Clone().Sub(output)          // (1 - output) - clone ones first to avoid modifying it
	term2 := output.Clone().Mul(term1)         // output * (1 - output) - clone output to avoid modifying stored tensor
	gradInput := gradOutput.Clone().Mul(term2) // gradOutput * output * (1 - output) - clone gradOutput

	s.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Sigmoid).
func (s *Sigmoid) OutputShape(inputShape []int) ([]int, error) {
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, nil
}

// Tanh layer implements tanh activation as a Layer.
type Tanh struct {
	Base
}

// NewTanh creates a new Tanh layer.
func NewTanh(name string) *Tanh {
	tanh := &Tanh{
		Base: NewBase("tanh"),
	}
	tanh.Base.ParseOptions(WithName(name))
	return tanh
}

// Init initializes the layer.
func (t *Tanh) Init(inputShape []int) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Tanh.Init: empty input shape")
	}
	// Output shape is same as input for Tanh
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	t.Base.AllocOutput(inputShape, outputSize)
	return nil
}

// Forward computes tanh: output = tanh(input).
func (t *Tanh) Forward(input types.Tensor) (types.Tensor, error) {
	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Tanh.Forward: empty input")
	}

	// Store input
	t.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := t.Base.Output()
	if output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Tanh.Forward: output not allocated, must call Init first")
	}

	// Apply tanh directly to output
	output = input.Tanh(output)

	t.Base.StoreOutput(output)
	return output, nil
}

// Backward computes tanh gradient: gradInput = gradOutput * (1 - output^2).
func (t *Tanh) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if gradOutput == nil || gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Tanh.Backward: empty gradOutput")
	}

	output := t.Base.Output()
	if output == nil || output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Tanh.Backward: output not stored, must call Forward first")
	}

	// Tanh gradient using primitives: gradInput = gradOutput * (1 - output^2)
	squared := output.Clone().Mul(output) // output^2
	ones := tensor.OnesLike(output)
	term := ones.Clone().Sub(squared)         // (1 - output^2)
	gradInput := gradOutput.Clone().Mul(term) // gradOutput * (1 - output^2)

	t.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Tanh).
func (t *Tanh) OutputShape(inputShape []int) ([]int, error) {
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, nil
}

// Softmax layer implements softmax activation as a Layer.
type Softmax struct {
	Base
	dim int // Dimension along which to apply softmax
}

// NewSoftmax creates a new Softmax layer for the given dimension.
func NewSoftmax(name string, dim int) *Softmax {
	softmax := &Softmax{
		Base: NewBase("softmax"),
		dim:  dim,
	}
	softmax.Base.ParseOptions(WithName(name))
	return softmax
}

// Init initializes the layer.
func (s *Softmax) Init(inputShape []int) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Softmax.Init: empty input shape")
	}
	// Output shape is same as input for Softmax
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	s.Base.AllocOutput(inputShape, outputSize)
	return nil
}

// Forward computes softmax: output = softmax(input, dim).
func (s *Softmax) Forward(input types.Tensor) (types.Tensor, error) {
	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Softmax.Forward: empty input")
	}

	// Store input
	s.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := s.Base.Output()
	if output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Softmax.Forward: output not allocated, must call Init first")
	}

	// Apply softmax directly to output
	output = input.Softmax(s.dim, output)

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes softmax gradient: gradInput = output * (gradOutput - sum(gradOutput * output)).
func (s *Softmax) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if gradOutput == nil || gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Softmax.Backward: empty gradOutput")
	}

	output := s.Base.Output()
	if output == nil || output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Softmax.Backward: output not stored, must call Forward first")
	}

	// Softmax gradient using primitives: gradInput = output * (gradOutput - sum(gradOutput * output))
	// Step 1: Compute element-wise product: gradOutput * output
	prod := gradOutput.Clone().Mul(output)

	// Step 2: Sum along the softmax dimension
	sumTerm := prod.Sum(s.dim)

	// Step 3: Broadcast sum back to original shape
	sumBroadcast, err := sumTerm.BroadcastTo(output.Shape())
	if err != nil {
		return nil, fmt.Errorf("Softmax.Backward: broadcast failed: %w", err)
	}

	// Step 4: Compute: gradOutput - sumTerm
	diff := gradOutput.Clone().Sub(sumBroadcast)

	// Step 5: Compute final gradient: output * (gradOutput - sumTerm)
	gradInput := output.Clone().Mul(diff)

	s.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Softmax).
func (s *Softmax) OutputShape(inputShape []int) ([]int, error) {
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, nil
}

// DropoutOption represents a configuration option for Dropout layer.
type DropoutOption func(*Dropout)

// WithDropoutRate returns a DropoutOption that sets the dropout rate.
func WithDropoutRate(rate float32) DropoutOption {
	return func(d *Dropout) {
		if rate < 0 || rate >= 1 {
			return // Invalid rate, ignore
		}
		d.p = rate
	}
}

// WithTrainingMode returns a DropoutOption that sets the training mode.
func WithTrainingMode(isTraining bool) DropoutOption {
	return func(d *Dropout) {
		d.isTraining = isTraining
	}
}

// WithDropoutRNG returns a DropoutOption that sets the random number generator.
func WithDropoutRNG(rng *rand.Rand) DropoutOption {
	return func(d *Dropout) {
		if rng != nil {
			d.rng = rng
		}
	}
}

// Dropout layer implements dropout regularization as a Layer.
// During training: randomly sets elements to zero with probability p, scales by 1/(1-p).
// During inference: passes input through unchanged.
type Dropout struct {
	Base
	p          float32      // Dropout rate (probability of dropping, 0.0 to 1.0)
	isTraining bool         // Whether in training mode
	mask       types.Tensor // Mask stored from forward pass (for backward)
	rng        *rand.Rand   // Random number generator
}

// NewDropout creates a new Dropout layer.
// Default dropout rate is 0.5, default training mode is false (inference).
func NewDropout(name string, opts ...DropoutOption) *Dropout {
	d := &Dropout{
		Base:       NewBase("dropout"),
		p:          0.5,
		isTraining: false,
		rng:        rand.New(rand.NewSource(rand.Int63())),
	}
	d.Base.ParseOptions(WithName(name))

	// Apply options
	for _, opt := range opts {
		opt(d)
	}

	return d
}

// Init initializes the layer.
func (d *Dropout) Init(inputShape []int) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Dropout.Init: empty input shape")
	}
	// Output shape is same as input for Dropout
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	d.Base.AllocOutput(inputShape, outputSize)
	return nil
}

// Forward computes dropout: during training, randomly zero elements with probability p and scale by 1/(1-p).
// During inference, passes input through unchanged.
func (d *Dropout) Forward(input types.Tensor) (types.Tensor, error) {
	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dropout.Forward: empty input")
	}

	// Store input
	d.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := d.Base.Output()
	if output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dropout.Forward: output not allocated, must call Init first")
	}

	inputSize := input.Size()
	scale := float32(1.0) / (1.0 - d.p)

	if d.isTraining && d.p > 0 {
		// Allocate or reuse mask tensor
		if d.mask == nil || d.mask.Shape().Rank() == 0 || d.mask.Size() != inputSize {
			d.mask = tensor.New(tensor.DTFP32, input.Shape())
		}

		// Generate mask using tensor method
		// Convert float32 to float64 for interface
		d.mask = d.mask.DropoutMask(float64(d.p), float64(scale), d.rng)

		// Copy input to output and apply dropout using tensor method
		// Use Clone to copy input to output, then apply dropout
		output = input.Clone()
		output = output.DropoutForward(d.mask)
	} else {
		// Inference mode: pass through unchanged
		output = input.Clone()
		// Clear mask (not needed in inference)
		d.mask = nil
	}

	d.Base.StoreOutput(output)
	return output, nil
}

// Backward computes dropout gradient: gradInput = gradOutput * mask.
func (d *Dropout) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dropout.Backward: empty gradOutput")
	}

	input := d.Base.Input()
	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dropout.Backward: input not stored, must call Forward first")
	}

	var gradInput types.Tensor
	if d.isTraining && d.p > 0 && d.mask != nil && d.mask.Shape().Rank() > 0 {
		// Training mode: gradInput = gradOutput * mask
		gradInput = gradOutput.Clone().Mul(d.mask)
	} else {
		// Inference mode or no dropout: pass gradient through unchanged
		gradInput = gradOutput.Clone()
	}

	d.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Dropout).
func (d *Dropout) OutputShape(inputShape []int) ([]int, error) {
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, nil
}

// SetTrainingMode sets whether the layer is in training mode.
func (d *Dropout) SetTrainingMode(isTraining bool) {
	d.isTraining = isTraining
}

// TrainingMode returns whether the layer is in training mode.
func (d *Dropout) TrainingMode() bool {
	return d.isTraining
}

// Rate returns the dropout rate.
func (d *Dropout) Rate() float32 {
	return d.p
}
