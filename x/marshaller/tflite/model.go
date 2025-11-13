package tflite

import (
	"fmt"

	nnTypes "github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tflite "github.com/mattn/go-tflite"
)

// Model wraps a TFLite interpreter and implements nn.types.Model interface.
// It provides inference-only capabilities using the TFLite runtime directly.
//
// This approach:
// - Leverages TFLite's optimized runtime
// - Supports all TFLite operators
// - Supports quantized models
// - Can use hardware acceleration (delegates)
// - Requires less code maintenance
type Model struct {
	name        string
	interpreter *tflite.Interpreter
	model       *tflite.Model
	options     *tflite.InterpreterOptions
	inputShape  tensor.Shape
	outputShape tensor.Shape
	canLearn    bool // Always false for TFLite models

	// Cached tensors from last Forward pass
	lastInput  tensor.Tensor
	lastOutput tensor.Tensor

	// Keep model data alive
	modelData []byte
}

// NewModel creates a new TFLite model wrapper from model bytes.
// The model is loaded into a TFLite interpreter and ready for inference.
//
// Options can be used to configure the interpreter (threads, delegates, etc.)
func NewModel(modelData []byte, name string, opts ...InterpreterOption) (*Model, error) {
	if len(modelData) == 0 {
		return nil, fmt.Errorf("tflite.NewModel: empty model data")
	}

	m := &Model{
		name:      name,
		modelData: modelData, // Keep alive
		canLearn:  false,     // TFLite models are inference-only
	}

	// Create TFLite model
	m.model = tflite.NewModel(modelData)
	if m.model == nil {
		return nil, fmt.Errorf("tflite.NewModel: failed to create TFLite model")
	}

	// Create interpreter options
	m.options = tflite.NewInterpreterOptions()
	if m.options == nil {
		m.model.Delete()
		return nil, fmt.Errorf("tflite.NewModel: failed to create interpreter options")
	}

	// Apply options (threads, delegates, etc.)
	for _, opt := range opts {
		if err := opt(m); err != nil {
			m.Close()
			return nil, fmt.Errorf("tflite.NewModel: failed to apply option: %w", err)
		}
	}

	// Create interpreter
	m.interpreter = tflite.NewInterpreter(m.model, m.options)
	if m.interpreter == nil {
		m.Close()
		return nil, fmt.Errorf("tflite.NewModel: failed to create interpreter")
	}

	// Allocate tensors
	if status := m.interpreter.AllocateTensors(); status != tflite.OK {
		m.Close()
		return nil, fmt.Errorf("tflite.NewModel: failed to allocate tensors: status %d", status)
	}

	// Validate and cache input/output shapes
	if err := m.cacheShapes(); err != nil {
		m.Close()
		return nil, err
	}

	return m, nil
}

// cacheShapes validates and caches input/output tensor shapes.
func (m *Model) cacheShapes() error {
	// Validate input count
	inputCount := m.interpreter.GetInputTensorCount()
	if inputCount != 1 {
		return fmt.Errorf("tflite.Model: only single-input models supported, got %d inputs", inputCount)
	}

	// Validate output count
	outputCount := m.interpreter.GetOutputTensorCount()
	if outputCount != 1 {
		return fmt.Errorf("tflite.Model: only single-output models supported, got %d outputs", outputCount)
	}

	// Get input shape
	inputTensor := m.interpreter.GetInputTensor(0)
	if inputTensor == nil {
		return fmt.Errorf("tflite.Model: failed to get input tensor")
	}
	m.inputShape = tensor.NewShape(tensorDims(inputTensor)...)

	// Get output shape
	outputTensor := m.interpreter.GetOutputTensor(0)
	if outputTensor == nil {
		return fmt.Errorf("tflite.Model: failed to get output tensor")
	}
	m.outputShape = tensor.NewShape(tensorDims(outputTensor)...)

	return nil
}

// tensorDims creates a Go slice from a TFLite tensor's dimensions.
func tensorDims(t *tflite.Tensor) []int {
	if t == nil {
		return nil
	}
	dims := make([]int, t.NumDims())
	for i := range dims {
		dims[i] = t.Dim(i)
	}
	return dims
}

// Close releases TFLite resources.
// Must be called when done using the model.
func (m *Model) Close() {
	if m.interpreter != nil {
		m.interpreter.Delete()
		m.interpreter = nil
	}
	if m.options != nil {
		m.options.Delete()
		m.options = nil
	}
	if m.model != nil {
		m.model.Delete()
		m.model = nil
	}
}

// Name returns the model name.
func (m *Model) Name() string {
	return m.name
}

// Init initializes the model with the given input shape.
// For TFLite models, this validates that the shape matches the model's expected input.
func (m *Model) Init(inputShape tensor.Shape) error {
	if !m.inputShape.Equal(inputShape) {
		return fmt.Errorf("tflite.Model.Init: input shape mismatch: expected %v, got %v", m.inputShape, inputShape)
	}
	return nil
}

// Forward runs inference on the input tensor.
//
// Behavior:
// - Copies input data to TFLite interpreter (no zero-copy path yet)
// - Runs inference via TFLite runtime
// - Returns output as an eager tensor (copies data from the interpreter)
//
// The returned tensor is valid independently of further Forward calls.
func (m *Model) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if m.interpreter == nil {
		return nil, fmt.Errorf("tflite.Model.Forward: model closed")
	}

	if input == nil || input.Empty() {
		return nil, fmt.Errorf("tflite.Model.Forward: nil or empty input")
	}

	// Validate input shape
	if !m.inputShape.Equal(input.Shape()) {
		return nil, fmt.Errorf("tflite.Model.Forward: input shape mismatch: expected %v, got %v", m.inputShape, input.Shape())
	}

	// Get TFLite input tensor
	tfliteInput := m.interpreter.GetInputTensor(0)
	if tfliteInput == nil {
		return nil, fmt.Errorf("tflite.Model.Forward: failed to get input tensor")
	}

	// Copy input data to TFLite interpreter
	if err := copyToTFLiteTensor(tfliteInput, input.Data()); err != nil {
		return nil, fmt.Errorf("tflite.Model.Forward: failed to copy input: %w", err)
	}

	// Store input for later retrieval
	m.lastInput = input

	// Run inference
	if status := m.interpreter.Invoke(); status != tflite.OK {
		return nil, fmt.Errorf("tflite.Model.Forward: inference failed with status %d", status)
	}

	// Get output tensor
	tfliteOutput := m.interpreter.GetOutputTensor(0)
	if tfliteOutput == nil {
		return nil, fmt.Errorf("tflite.Model.Forward: failed to get output tensor")
	}

	// Copy output data into an eager tensor so it survives future invocations
	output := tensor.New(tensor.DTFP32, tensor.NewShape(tensorDims(tfliteOutput)...))
	if err := copyFromTFLiteTensor(tfliteOutput, output.Data()); err != nil {
		return nil, fmt.Errorf("tflite.Model.Forward: failed to read output: %w", err)
	}

	m.lastOutput = output

	return output, nil
}

// copyToTFLiteTensor copies data to a TFLite tensor based on the concrete slice type.
func copyToTFLiteTensor(dst *tflite.Tensor, srcData any) error {
	switch data := srcData.(type) {
	case []float32:
		return statusError(dst.CopyFromBuffer(data), "CopyFromBuffer float32")
	case []float64:
		buf := make([]float32, len(data))
		for i, v := range data {
			buf[i] = float32(v)
		}
		return statusError(dst.CopyFromBuffer(buf), "CopyFromBuffer float64")
	case []int32:
		return statusError(dst.CopyFromBuffer(data), "CopyFromBuffer int32")
	case []int64:
		return statusError(dst.CopyFromBuffer(data), "CopyFromBuffer int64")
	case []int8:
		return statusError(dst.CopyFromBuffer(data), "CopyFromBuffer int8")
	case []uint8:
		return statusError(dst.CopyFromBuffer(data), "CopyFromBuffer uint8")
	default:
		return fmt.Errorf("unsupported input data type %T", srcData)
	}
}

// copyFromTFLiteTensor copies data out of a TFLite tensor into the provided slice.
func copyFromTFLiteTensor(src *tflite.Tensor, dstData any) error {
	switch data := dstData.(type) {
	case []float32:
		return statusError(src.CopyToBuffer(data), "CopyToBuffer float32")
	case []int32:
		return statusError(src.CopyToBuffer(data), "CopyToBuffer int32")
	case []uint8:
		return statusError(src.CopyToBuffer(data), "CopyToBuffer uint8")
	default:
		return fmt.Errorf("unsupported output data type %T", dstData)
	}
}

func statusError(status tflite.Status, op string) error {
	if status != tflite.OK {
		return fmt.Errorf("%s failed with status %d", op, status)
	}
	return nil
}

// Backward is not supported for TFLite models (inference only).
func (m *Model) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	return nil, fmt.Errorf("tflite.Model.Backward: training not supported (inference only)")
}

// OutputShape returns the output shape for the given input shape.
func (m *Model) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if !m.inputShape.Equal(inputShape) {
		return nil, fmt.Errorf("tflite.Model.OutputShape: input shape mismatch: expected %v, got %v", m.inputShape, inputShape)
	}
	return m.outputShape, nil
}

// CanLearn returns false (TFLite models are inference-only).
func (m *Model) CanLearn() bool {
	return false
}

// SetCanLearn is a no-op for TFLite models.
func (m *Model) SetCanLearn(canLearn bool) {
	// Ignore - TFLite models cannot learn
}

// Input returns the input tensor from the last Forward pass.
func (m *Model) Input() tensor.Tensor {
	return m.lastInput
}

// Output returns the output tensor from the last Forward pass.
func (m *Model) Output() tensor.Tensor {
	return m.lastOutput
}

// Update is not supported for TFLite models (inference only).
func (m *Model) Update(optimizer nnTypes.Optimizer) error {
	return fmt.Errorf("tflite.Model.Update: training not supported (inference only)")
}

// ZeroGrad is a no-op for TFLite models (no parameters).
func (m *Model) ZeroGrad() {
	// No-op - no parameters to zero
}

// Parameter returns nil (TFLite models have no accessible parameters).
func (m *Model) Parameter(idx nnTypes.ParamIndex) (nnTypes.Parameter, bool) {
	return nnTypes.Parameter{}, false
}

// Parameters returns an empty map (TFLite models have no accessible parameters).
func (m *Model) Parameters() map[nnTypes.ParamIndex]nnTypes.Parameter {
	return make(map[nnTypes.ParamIndex]nnTypes.Parameter)
}

// LayerCount returns 1 (TFLite model is treated as a single layer).
func (m *Model) LayerCount() int {
	return 1
}

// GetLayer returns the model itself if index is 0.
func (m *Model) GetLayer(index int) nnTypes.Layer {
	if index == 0 {
		return m
	}
	return nil
}

// GetInputTensor returns the TFLite interpreter's input tensor.
// This allows direct access to the input buffer for zero-copy operations.
func (m *Model) GetInputTensor() *tflite.Tensor {
	if m.interpreter == nil {
		return nil
	}
	return m.interpreter.GetInputTensor(0)
}

// GetOutputTensor returns the TFLite interpreter's output tensor.
// This allows direct access to the output buffer for zero-copy operations.
func (m *Model) GetOutputTensor() *tflite.Tensor {
	if m.interpreter == nil {
		return nil
	}
	return m.interpreter.GetOutputTensor(0)
}

// Invoke runs inference directly using the TFLite interpreter.
// Use this for manual control when you've already populated the input tensor.
func (m *Model) Invoke() error {
	if m.interpreter == nil {
		return fmt.Errorf("tflite.Model.Invoke: model closed")
	}
	if status := m.interpreter.Invoke(); status != tflite.OK {
		return fmt.Errorf("tflite.Model.Invoke: inference failed with status %d", status)
	}
	return nil
}

// InputShape returns the model's expected input shape.
func (m *Model) InputShape() tensor.Shape {
	return m.inputShape
}

// OutputShapeValue returns the model's output shape.
func (m *Model) OutputShapeValue() tensor.Shape {
	return m.outputShape
}

// InterpreterOption is a function that configures a TFLite model.
type InterpreterOption func(*Model) error

// WithNumThreads sets the number of threads for TFLite inference.
func WithNumThreads(numThreads int) InterpreterOption {
	return func(m *Model) error {
		if numThreads < 1 {
			return fmt.Errorf("numThreads must be >= 1, got %d", numThreads)
		}
		m.options.SetNumThread(numThreads)
		return nil
	}
}

// WithErrorReporter sets a custom error reporter.
func WithErrorReporter(reporter func(string)) InterpreterOption {
	return func(m *Model) error {
		if reporter == nil {
			return nil
		}
		m.options.SetErrorReporter(func(msg string, _ interface{}) {
			reporter(msg)
		}, nil)
		return nil
	}
}

// Example usage:
//
//  // Load model
//  modelData, _ := os.ReadFile("model.tflite")
//  model, _ := tflite.NewModel(modelData, "mnist",
//      tflite.WithNumThreads(4),
//  )
//  defer model.Close()
//
//  // Run inference with eager tensor
//  input := tensor.New(tensor.DTFP32, tensor.NewShape(1, 28, 28, 1))
//  output, _ := model.Forward(input)
//
//  // Or use TFLite tensors directly (zero-copy)
//  tfliteInput := model.GetInputTensor()
//  // ... populate tfliteInput ...
//  model.Invoke()
//  tfliteOutput := model.GetOutputTensor()
