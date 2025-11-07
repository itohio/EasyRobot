package layers

import (
	"fmt"
	"math/rand"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Alias for tensor.Tensor to match Layer interface
type Tensor = tensor.Tensor

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
func (r *ReLU) Init(inputShape tensor.Shape) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("ReLU.Init: empty input shape")
	}
	// Output shape is same as input for ReLU
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	r.Base.AllocOutput(inputShape, outputSize)

	// No need to pre-allocate mask tensor - ReLUGrad now uses optimized primitive directly

	return nil
}

// Forward computes ReLU: output = max(0, input).
func (r *ReLU) Forward(input Tensor) (Tensor, error) {
	if r == nil {
		return nil, fmt.Errorf("ReLU.Forward: nil layer")
	}
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("ReLU.Forward: empty input")
	}

	// Store input
	r.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := r.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("ReLU.Forward: output not allocated, must call Init first")
	}

	// Apply ReLU directly to output
	output = input.ReLU(output)

	r.Base.StoreOutput(output)
	return output, nil
}

// Backward computes ReLU gradient: gradInput = gradOutput * (input > 0 ? 1 : 0).
func (r *ReLU) Backward(gradOutput Tensor) (Tensor, error) {
	if r == nil {
		return nil, fmt.Errorf("ReLU.Backward: nil layer")
	}
	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("ReLU.Backward: empty gradOutput")
	}

	input := r.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("ReLU.Backward: input not stored, must call Forward first")
	}

	// Use optimized ReLU gradient primitive: gradInput = gradOutput * (input > 0 ? 1 : 0)
	// This directly calls fp32.ReLUGrad, eliminating intermediate mask tensor and extra multiply
	gradInput := r.Base.Grad() // Use pre-allocated grad tensor if available
	if tensor.IsNil(gradInput) {
		gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
	}
	input.ReLUGrad(gradInput, gradOutput)

	r.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for ReLU).
func (r *ReLU) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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
func (s *Sigmoid) Init(inputShape tensor.Shape) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Sigmoid.Init: empty input shape")
	}
	// Output shape is same as input for Sigmoid
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	s.Base.AllocOutput(inputShape, outputSize)

	// No need to pre-allocate scratch tensors - SigmoidGrad now uses optimized primitive directly

	return nil
}

// Forward computes sigmoid: output = 1 / (1 + exp(-input)).
func (s *Sigmoid) Forward(input Tensor) (Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("Sigmoid.Forward: nil layer")
	}
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Sigmoid.Forward: empty input")
	}

	// Store input
	s.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := s.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Sigmoid.Forward: output not allocated, must call Init first")
	}

	// Apply sigmoid directly to output
	output = input.Sigmoid(output)

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes sigmoid gradient: gradInput = gradOutput * output * (1 - output).
func (s *Sigmoid) Backward(gradOutput Tensor) (Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("Sigmoid.Backward: nil layer")
	}
	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Sigmoid.Backward: empty gradOutput")
	}

	input := s.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Sigmoid.Backward: input not stored, must call Forward first")
	}

	output := s.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Sigmoid.Backward: output not stored, must call Forward first")
	}

	// Use optimized sigmoid gradient primitive: gradInput = gradOutput * output * (1 - output)
	// This directly calls fp32.SigmoidGrad, eliminating intermediate tensors and multiple passes
	gradInput := s.Base.Grad() // Use pre-allocated grad tensor if available
	if tensor.IsNil(gradInput) {
		gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
	}
	output.SigmoidGrad(gradInput, gradOutput)

	s.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Sigmoid).
func (s *Sigmoid) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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
func (t *Tanh) Init(inputShape tensor.Shape) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Tanh.Init: empty input shape")
	}
	// Output shape is same as input for Tanh
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	t.Base.AllocOutput(inputShape, outputSize)

	// No need to pre-allocate scratch tensors - TanhGrad now uses optimized primitive directly

	return nil
}

// Forward computes tanh: output = tanh(input).
func (t *Tanh) Forward(input Tensor) (Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("Tanh.Forward: nil layer")
	}
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Tanh.Forward: empty input")
	}

	// Store input
	t.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := t.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Tanh.Forward: output not allocated, must call Init first")
	}

	// Apply tanh directly to output
	output = input.Tanh(output)

	t.Base.StoreOutput(output)
	return output, nil
}

// Backward computes tanh gradient: gradInput = gradOutput * (1 - output^2).
func (t *Tanh) Backward(gradOutput Tensor) (Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("Tanh.Backward: nil layer")
	}
	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Tanh.Backward: empty gradOutput")
	}

	input := t.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Tanh.Backward: input not stored, must call Forward first")
	}

	output := t.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Tanh.Backward: output not stored, must call Forward first")
	}

	// Use optimized tanh gradient primitive: gradInput = gradOutput * (1 - output^2)
	// This directly calls fp32.TanhGrad, eliminating intermediate tensors and multiple passes
	gradInput := t.Base.Grad() // Use pre-allocated grad tensor if available
	if tensor.IsNil(gradInput) {
		gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
	}
	output.TanhGrad(gradInput, gradOutput)

	t.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Tanh).
func (t *Tanh) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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
func (s *Softmax) Init(inputShape tensor.Shape) error {
	if len(inputShape) == 0 {
		return fmt.Errorf("Softmax.Init: empty input shape")
	}
	// Validate dimension
	if s.dim < 0 || s.dim >= len(inputShape) {
		return fmt.Errorf("Softmax.Init: invalid dimension %d for input shape %v", s.dim, inputShape)
	}
	// Output shape is same as input for Softmax
	outputSize := 1
	for _, dim := range inputShape {
		outputSize *= dim
	}
	s.Base.AllocOutput(inputShape, outputSize)

	// No need to pre-allocate scratch tensors - SoftmaxGrad now uses optimized primitive directly

	return nil
}

// Forward computes softmax: output = softmax(input, dim).
func (s *Softmax) Forward(input Tensor) (Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("Softmax.Forward: nil layer")
	}
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Softmax.Forward: empty input")
	}

	// Store input
	s.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := s.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Softmax.Forward: output not allocated, must call Init first")
	}

	// Apply softmax directly to output
	output = input.Softmax(s.dim, output)

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes softmax gradient: gradInput = output * (gradOutput - sum(gradOutput * output)).
func (s *Softmax) Backward(gradOutput Tensor) (Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("Softmax.Backward: nil layer")
	}
	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Softmax.Backward: empty gradOutput")
	}

	input := s.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Softmax.Backward: input not stored, must call Forward first")
	}

	output := s.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Softmax.Backward: output not stored, must call Forward first")
	}

	// Use optimized softmax gradient primitive: gradInput = output * (gradOutput - sum(gradOutput * output))
	// This directly calls fp32.Softmax*Grad, eliminating all intermediate tensors and operations
	gradInput := s.Base.Grad()
	if tensor.IsNil(gradInput) {
		gradInput = tensor.New(output.DataType(), output.Shape())
	}
	output.SoftmaxGrad(gradInput, gradOutput, s.dim)

	s.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Softmax).
func (s *Softmax) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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
	p          float32            // Dropout rate (probability of dropping, 0.0 to 1.0)
	isTraining bool               // Whether in training mode
	mask       tensorTypes.Tensor // Mask stored from forward pass (for backward)
	rng        *rand.Rand         // Random number generator
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
func (d *Dropout) Init(inputShape tensor.Shape) error {
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
func (d *Dropout) Forward(input Tensor) (Tensor, error) {
	if d == nil {
		return nil, fmt.Errorf("Dropout.Forward: nil layer")
	}
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Dropout.Forward: empty input")
	}

	// Store input
	d.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := d.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Dropout.Forward: output not allocated, must call Init first")
	}

	inputSize := input.Size()
	scale := float32(1.0) / (1.0 - d.p)

	if d.isTraining && d.p > 0 {
		// Allocate or reuse mask tensor
		// Use layer's data type for intermediate mask tensor
		if tensor.IsNil(d.mask) || d.mask.Size() != inputSize {
			d.mask = tensor.New(d.Base.DataType(), input.Shape())
		}

		// Generate mask using tensor method
		// Convert float32 to float64 for interface
		d.mask = d.mask.DropoutMask(float64(d.p), float64(scale), d.rng)

		// Copy input to output and apply dropout using tensor method
		// Use Copy instead of Clone (more efficient when destination exists)
		output.Copy(input)
		output = output.DropoutForward(output, d.mask)
	} else {
		// Inference mode: pass through unchanged
		// Use Copy instead of Clone (more efficient when destination exists)
		output.Copy(input)
		// Clear mask (not needed in inference)
		d.mask = nil
	}

	d.Base.StoreOutput(output)
	return output, nil
}

// Backward computes dropout gradient: gradInput = gradOutput * mask.
func (d *Dropout) Backward(gradOutput Tensor) (Tensor, error) {
	if d == nil {
		return nil, fmt.Errorf("Dropout.Backward: nil layer")
	}
	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Dropout.Backward: empty gradOutput")
	}

	input := d.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Dropout.Backward: input not stored, must call Forward first")
	}

	var gradInput Tensor
	if d.isTraining && d.p > 0 && !tensor.IsNil(d.mask) {
		// Training mode: gradInput = gradOutput * mask
		// Use pre-allocated grad tensor if available, otherwise create new one
		gradInput = d.Base.Grad()
		if tensor.IsNil(gradInput) {
			gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
		}
		gradOutput.Multiply(gradInput, d.mask)
	} else {
		// Inference mode or no dropout: pass gradient through unchanged
		// Use pre-allocated grad tensor if available, otherwise use Copy
		gradInput = d.Base.Grad()
		if tensor.IsNil(gradInput) {
			gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
			gradInput.Copy(gradOutput)
		} else {
			gradInput.Copy(gradOutput)
		}
	}

	d.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Dropout).
func (d *Dropout) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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
