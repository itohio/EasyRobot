package layers

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

const float32ExpMax = 88.0 // max value for exp to avoid overflow

// ReLU layer implements ReLU activation as a Layer.
type ReLU struct {
	Base
}

// NewReLU creates a new ReLU layer.
func NewReLU(name string) *ReLU {
	return &ReLU{
		Base: NewBase("relu", WithName(name)),
	}
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
func (r *ReLU) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("ReLU.Forward: empty input")
	}

	// Store input
	r.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := r.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("ReLU.Forward: output not allocated, must call Init first")
	}

	// ReLU: output = max(0, input)
	for i := range output.Data {
		if input.Data[i] > 0 {
			output.Data[i] = input.Data[i]
		} else {
			output.Data[i] = 0
		}
	}

	r.Base.StoreOutput(output)
	return output, nil
}

// Backward computes ReLU gradient: gradInput = gradOutput * (input > 0 ? 1 : 0).
func (r *ReLU) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("ReLU.Backward: empty gradOutput")
	}

	input := r.Base.Input()
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("ReLU.Backward: input not stored, must call Forward first")
	}

	// Allocate gradient tensor
	gradSize := gradOutput.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(gradOutput.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, gradOutput.Dim)

	// ReLU derivative: 1 if input > 0, else 0
	for i := range gradInput.Data {
		if input.Data[i] > 0 {
			gradInput.Data[i] = gradOutput.Data[i]
		} else {
			gradInput.Data[i] = 0
		}
	}

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
	return &Sigmoid{
		Base: NewBase("sigmoid", WithName(name)),
	}
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
func (s *Sigmoid) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Sigmoid.Forward: empty input")
	}

	// Store input
	s.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := s.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Sigmoid.Forward: output not allocated, must call Init first")
	}

	// Sigmoid: output = 1 / (1 + exp(-input))
	for i := range output.Data {
		x := -input.Data[i]
		if x > float32ExpMax {
			output.Data[i] = 0.0
		} else if x < -float32ExpMax {
			output.Data[i] = 1.0
		} else {
			output.Data[i] = 1.0 / (1.0 + float32(math.Exp(float64(x))))
		}
	}

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes sigmoid gradient: gradInput = gradOutput * output * (1 - output).
func (s *Sigmoid) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Sigmoid.Backward: empty gradOutput")
	}

	output := s.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Sigmoid.Backward: output not stored, must call Forward first")
	}

	// Allocate gradient tensor
	gradSize := gradOutput.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(gradOutput.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, gradOutput.Dim)

	// Sigmoid derivative: output * (1 - output)
	for i := range gradInput.Data {
		gradInput.Data[i] = gradOutput.Data[i] * output.Data[i] * (1 - output.Data[i])
	}

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
	return &Tanh{
		Base: NewBase("tanh", WithName(name)),
	}
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
func (t *Tanh) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Tanh.Forward: empty input")
	}

	// Store input
	t.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := t.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Tanh.Forward: output not allocated, must call Init first")
	}

	// Tanh: output = tanh(input)
	for i := range output.Data {
		x := float64(input.Data[i])
		output.Data[i] = float32(math.Tanh(x))
	}

	t.Base.StoreOutput(output)
	return output, nil
}

// Backward computes tanh gradient: gradInput = gradOutput * (1 - output^2).
func (t *Tanh) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Tanh.Backward: empty gradOutput")
	}

	output := t.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Tanh.Backward: output not stored, must call Forward first")
	}

	// Allocate gradient tensor
	gradSize := gradOutput.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(gradOutput.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, gradOutput.Dim)

	// Tanh derivative: 1 - output^2
	for i := range gradInput.Data {
		gradInput.Data[i] = gradOutput.Data[i] * (1 - output.Data[i]*output.Data[i])
	}

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
	return &Softmax{
		Base: NewBase("softmax", WithName(name)),
		dim:  dim,
	}
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
func (s *Softmax) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Softmax.Forward: empty input")
	}

	// Store input
	s.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := s.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Softmax.Forward: output not allocated, must call Init first")
	}

	// Copy input to output first
	copy(output.Data, input.Data)

	// Apply softmax directly
	softmaxTensor(&output, s.dim)

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes softmax gradient: gradInput = output * (gradOutput - sum(gradOutput * output)).
func (s *Softmax) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Softmax.Backward: empty gradOutput")
	}

	output := s.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Softmax.Backward: output not stored, must call Forward first")
	}

	// Allocate gradient tensor
	gradSize := gradOutput.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(gradOutput.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, gradOutput.Dim)

	// Softmax backward: output * (gradOutput - sum(gradOutput * output))
	if len(output.Dim) == 1 {
		// 1D case: compute sum over all elements
		var sum float32
		for i := range gradOutput.Data {
			sum += gradOutput.Data[i] * output.Data[i]
		}
		// Apply: output * (gradOutput - sum)
		for i := range gradInput.Data {
			gradInput.Data[i] = output.Data[i] * (gradOutput.Data[i] - sum)
		}
	} else if len(output.Dim) == 2 {
		if s.dim == 1 {
			// Softmax along columns (dim 1): each row independently
			M, N := output.Dim[0], output.Dim[1]
			for i := 0; i < M; i++ {
				rowStart := i * N
				var sum float32
				// Compute sum for this row
				for j := 0; j < N; j++ {
					idx := rowStart + j
					sum += gradOutput.Data[idx] * output.Data[idx]
				}
				// Apply gradient for this row
				for j := 0; j < N; j++ {
					idx := rowStart + j
					gradInput.Data[idx] = output.Data[idx] * (gradOutput.Data[idx] - sum)
				}
			}
		} else if s.dim == 0 {
			// Softmax along rows (dim 0): each column independently
			M, N := output.Dim[0], output.Dim[1]
			for j := 0; j < N; j++ {
				var sum float32
				// Compute sum for this column
				for i := 0; i < M; i++ {
					idx := i*N + j
					sum += gradOutput.Data[idx] * output.Data[idx]
				}
				// Apply gradient for this column
				for i := 0; i < M; i++ {
					idx := i*N + j
					gradInput.Data[idx] = output.Data[idx] * (gradOutput.Data[idx] - sum)
				}
			}
		}
	}

	s.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape (same as input shape for Softmax).
func (s *Softmax) OutputShape(inputShape []int) ([]int, error) {
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	return outputShape, nil
}

// softmaxTensor applies softmax along specified dimension in-place.
func softmaxTensor(t *tensor.Tensor, dim int) {
	if t == nil || len(t.Dim) == 0 {
		return
	}

	if dim < 0 || dim >= len(t.Dim) {
		return
	}

	if len(t.Dim) == 1 {
		softmax1D(t)
		return
	}

	if len(t.Dim) == 2 {
		if dim == 0 {
			softmax2DRows(t)
			return
		} else if dim == 1 {
			softmax2DCols(t)
			return
		}
	}
}

func softmax1D(t *tensor.Tensor) {
	if len(t.Data) == 0 {
		return
	}
	maxVal := t.Data[0]
	for i := 1; i < len(t.Data); i++ {
		if t.Data[i] > maxVal {
			maxVal = t.Data[i]
		}
	}

	var sum float32
	for i := range t.Data {
		t.Data[i] = math32.Exp(t.Data[i] - maxVal)
		sum += t.Data[i]
	}

	if sum > 0 {
		for i := range t.Data {
			t.Data[i] /= sum
		}
	}
}

func softmax2DRows(t *tensor.Tensor) {
	if len(t.Dim) != 2 {
		return
	}
	M, N := t.Dim[0], t.Dim[1]

	for j := 0; j < N; j++ {
		maxVal := t.Data[j]
		for i := 1; i < M; i++ {
			val := t.Data[i*N+j]
			if val > maxVal {
				maxVal = val
			}
		}

		var sum float32
		for i := 0; i < M; i++ {
			val := t.Data[i*N+j] - maxVal
			t.Data[i*N+j] = math32.Exp(val)
			sum += t.Data[i*N+j]
		}

		if sum > 0 {
			for i := 0; i < M; i++ {
				t.Data[i*N+j] /= sum
			}
		}
	}
}

func softmax2DCols(t *tensor.Tensor) {
	if len(t.Dim) != 2 {
		return
	}
	M, N := t.Dim[0], t.Dim[1]

	for i := 0; i < M; i++ {
		rowStart := i * N
		maxVal := t.Data[rowStart]
		for j := 1; j < N; j++ {
			val := t.Data[rowStart+j]
			if val > maxVal {
				maxVal = val
			}
		}

		var sum float32
		for j := 0; j < N; j++ {
			val := t.Data[rowStart+j] - maxVal
			t.Data[rowStart+j] = math32.Exp(val)
			sum += t.Data[rowStart+j]
		}

		if sum > 0 {
			for j := 0; j < N; j++ {
				t.Data[rowStart+j] /= sum
			}
		}
	}
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
	p          float32       // Dropout rate (probability of dropping, 0.0 to 1.0)
	isTraining bool          // Whether in training mode
	mask       tensor.Tensor // Mask stored from forward pass (for backward)
	rng        *rand.Rand    // Random number generator
}

// NewDropout creates a new Dropout layer.
// Default dropout rate is 0.5, default training mode is false (inference).
func NewDropout(name string, opts ...DropoutOption) *Dropout {
	d := &Dropout{
		Base:       NewBase("dropout", WithName(name)),
		p:          0.5,
		isTraining: false,
		rng:        rand.New(rand.NewSource(rand.Int63())),
	}

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
func (d *Dropout) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dropout.Forward: empty input")
	}

	// Store input
	d.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := d.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dropout.Forward: output not allocated, must call Init first")
	}

	inputSize := input.Size()
	scale := float32(1.0) / (1.0 - d.p)

	if d.isTraining && d.p > 0 {
		// Allocate or reuse mask tensor
		if len(d.mask.Dim) == 0 || d.mask.Size() != inputSize {
			d.mask = tensor.Tensor{
				Dim:  make([]int, len(input.Dim)),
				Data: make([]float32, inputSize),
			}
			copy(d.mask.Dim, input.Dim)
		}

		// Generate mask and apply dropout
		for i := range output.Data {
			if d.rng.Float32() < d.p {
				// Drop this element
				d.mask.Data[i] = 0
				output.Data[i] = 0
			} else {
				// Keep this element, scale by 1/(1-p)
				d.mask.Data[i] = scale
				output.Data[i] = input.Data[i] * scale
			}
		}
	} else {
		// Inference mode: pass through unchanged
		copy(output.Data, input.Data)
		// Clear mask (not needed in inference)
		d.mask = tensor.Tensor{}
	}

	d.Base.StoreOutput(output)
	return output, nil
}

// Backward computes dropout gradient: gradInput = gradOutput * mask.
func (d *Dropout) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dropout.Backward: empty gradOutput")
	}

	input := d.Base.Input()
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dropout.Backward: input not stored, must call Forward first")
	}

	// Allocate gradient tensor
	gradSize := gradOutput.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(gradOutput.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, gradOutput.Dim)

	if d.isTraining && d.p > 0 && len(d.mask.Data) > 0 {
		// Training mode: multiply by mask (which contains 0 or scale)
		for i := range gradInput.Data {
			gradInput.Data[i] = gradOutput.Data[i] * d.mask.Data[i]
		}
	} else {
		// Inference mode or no dropout: pass gradient through unchanged
		copy(gradInput.Data, gradOutput.Data)
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
