package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// LSTM represents a Long Short-Term Memory (LSTM) layer.
// Supports single timestep and batch processing.
// Input: [input_size] or [batch_size, input_size]
// Output: [hidden_size] or [batch_size, hidden_size]
// Also maintains hidden state and cell state internally.
type LSTM struct {
	Base
	inputSize  int
	hiddenSize int
	hasBias    bool
	// Internal state
	hiddenState tensorTypes.Tensor // [hidden_size] or [batch_size, hidden_size]
	cellState   tensorTypes.Tensor // [hidden_size] or [batch_size, hidden_size]
}

// NewLSTM creates a new LSTM layer with the given input and hidden sizes.
// Accepts Base Option types.
func NewLSTM(inputSize, hiddenSize int, opts ...Option) (*LSTM, error) {
	if inputSize <= 0 {
		return nil, fmt.Errorf("LSTM: inputSize must be positive, got %d", inputSize)
	}
	if hiddenSize <= 0 {
		return nil, fmt.Errorf("LSTM: hiddenSize must be positive, got %d", hiddenSize)
	}

	// Create Base without options first
	base := NewBase("lstm")

	lstm := &LSTM{
		Base:       base,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		hasBias:    true, // Always create bias by default
	}

	// Parse options first to get data type (if specified) and any pre-set weights/biases
	lstm.Base.ParseOptions(opts...)

	// Initialize parameters using Parameter.Init()
	dtype := lstm.Base.DataType()

	// Initialize weight_ih parameter: [4*hidden_size, input_size]
	// Order: input gate, forget gate, cell gate, output gate
	lstm.Base.initParam(types.ParamLSTMWeightIH)
	weightIHParam, _ := lstm.Base.Parameter(types.ParamLSTMWeightIH)
	weightIHParam.Init(dtype, tensor.NewShape(4*hiddenSize, inputSize),
		types.ParamLSTMWeightIH, inputSize, 4*hiddenSize, lstm.Base.rng, lstm.Base.CanLearn())
	lstm.Base.SetParam(types.ParamLSTMWeightIH, weightIHParam)

	// Initialize weight_hh parameter: [4*hidden_size, hidden_size]
	lstm.Base.initParam(types.ParamLSTMWeightHH)
	weightHHParam, _ := lstm.Base.Parameter(types.ParamLSTMWeightHH)
	weightHHParam.Init(dtype, tensor.NewShape(4*hiddenSize, hiddenSize),
		types.ParamLSTMWeightHH, hiddenSize, 4*hiddenSize, lstm.Base.rng, lstm.Base.CanLearn())
	lstm.Base.SetParam(types.ParamLSTMWeightHH, weightHHParam)

	// Initialize bias parameter: [4*hidden_size]
	lstm.Base.initParam(types.ParamLSTMBias)
	biasParam, _ := lstm.Base.Parameter(types.ParamLSTMBias)
	biasParam.Init(dtype, tensor.NewShape(4*hiddenSize),
		types.ParamLSTMBias, 1, 4*hiddenSize, lstm.Base.rng, lstm.Base.CanLearn())
	lstm.Base.SetParam(types.ParamLSTMBias, biasParam)

	return lstm, nil
}

// Init initializes the layer, creating internal computation tensors.
func (l *LSTM) Init(inputShape tensor.Shape) error {
	if l == nil {
		return fmt.Errorf("LSTM.Init: nil layer")
	}

	// Validate input shape
	if len(inputShape) == 1 {
		if inputShape[0] != l.inputSize {
			return fmt.Errorf("LSTM.Init: input shape %v incompatible with inputSize %d", inputShape, l.inputSize)
		}
	} else if len(inputShape) == 2 {
		if inputShape[1] != l.inputSize {
			return fmt.Errorf("LSTM.Init: input shape %v incompatible with inputSize %d", inputShape, l.inputSize)
		}
	} else {
		return fmt.Errorf("LSTM.Init: input must be 1D or 2D, got %dD", len(inputShape))
	}

	// Compute output shape (same as hidden state shape)
	var outputShape tensor.Shape
	if len(inputShape) == 1 {
		outputShape = tensor.NewShape(l.hiddenSize)
	} else {
		outputShape = tensor.NewShape(inputShape[0], l.hiddenSize)
	}

	// Allocate output tensor
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	l.Base.AllocOutput(outputShape, outputSize)

	// Initialize hidden state and cell state to zeros
	l.hiddenState = tensor.New(l.Base.DataType(), outputShape)
	l.cellState = tensor.New(l.Base.DataType(), outputShape)

	return nil
}

// Forward computes the forward pass of LSTM.
// Uses stored hidden state and cell state from previous timesteps.
func (l *LSTM) Forward(input tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if l == nil {
		return nil, fmt.Errorf("LSTM.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("LSTM.Forward: empty input")
	}

	// Store input
	l.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := l.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("LSTM.Forward: output not allocated, must call Init first")
	}

	// Get parameters
	weightIHParam, ok := l.Base.Parameter(types.ParamLSTMWeightIH)
	if !ok || tensor.IsNil(weightIHParam.Data) {
		return nil, fmt.Errorf("LSTM.Forward: weight_ih parameter not initialized")
	}
	weightHHParam, ok := l.Base.Parameter(types.ParamLSTMWeightHH)
	if !ok || tensor.IsNil(weightHHParam.Data) {
		return nil, fmt.Errorf("LSTM.Forward: weight_hh parameter not initialized")
	}
	biasParam, ok := l.Base.Parameter(types.ParamLSTMBias)
	if !ok || tensor.IsNil(biasParam.Data) {
		return nil, fmt.Errorf("LSTM.Forward: bias parameter not initialized")
	}

	// Initialize hidden and cell states if not already initialized
	inputShape := input.Shape()
	if tensor.IsNil(l.hiddenState) {
		if len(inputShape) == 1 {
			l.hiddenState = tensor.New(l.Base.DataType(), tensor.NewShape(l.hiddenSize))
			l.cellState = tensor.New(l.Base.DataType(), tensor.NewShape(l.hiddenSize))
		} else {
			l.hiddenState = tensor.New(l.Base.DataType(), tensor.NewShape(inputShape[0], l.hiddenSize))
			l.cellState = tensor.New(l.Base.DataType(), tensor.NewShape(inputShape[0], l.hiddenSize))
		}
	}

	// Compute forward pass
	if err := l.computeForward(input, weightIHParam.Data, weightHHParam.Data, biasParam.Data,
		l.hiddenState, l.cellState, output); err != nil {
		return nil, fmt.Errorf("LSTM.Forward: computation failed: %w", err)
	}

	// Update internal state
	l.hiddenState = output.Clone()
	l.cellState = l.cellState.Clone() // cellState is updated in computeForward

	// Store output
	l.Base.StoreOutput(output)
	return output, nil
}

// computeForward computes the LSTM forward pass using only tensor operations.
// Input: [input_size] or [batch_size, input_size]
// WeightIH: [4*hidden_size, input_size]
// WeightHH: [4*hidden_size, hidden_size]
// Bias: [4*hidden_size]
// HiddenState: [hidden_size] or [batch_size, hidden_size]
// CellState: [hidden_size] or [batch_size, hidden_size]
// Output: [hidden_size] or [batch_size, hidden_size] (updated hidden state)
func (l *LSTM) computeForward(input, weightIH, weightHH, bias,
	hiddenState, cellState, output tensorTypes.Tensor) error {
	inputShape := input.Shape()
	isBatch := len(inputShape) == 2

	// Compute gates: gates = input @ weight_ih.T + hidden @ weight_hh.T + bias
	// gates shape: [batch_size, 4*hidden_size] or [4*hidden_size]
	var gates tensorTypes.Tensor
	if isBatch {
		// input: [batch_size, input_size], weight_ih: [4*hidden_size, input_size]
		// input @ weight_ih.T = [batch_size, 4*hidden_size]
		gates = input.MatMulTransposed(weightIH, false, true, nil)
	} else {
		// input: [input_size], weight_ih: [4*hidden_size, input_size]
		// Reshape input to [1, input_size] for matrix multiplication
		inputReshaped := input.Reshape(tensor.NewShape(1, l.inputSize))
		gatesTemp := inputReshaped.MatMulTransposed(weightIH, false, true, nil)
		// Reshape back to [4*hidden_size]
		gates = gatesTemp.Reshape(tensor.NewShape(4 * l.hiddenSize))
	}

	// Add hidden contribution: hidden @ weight_hh.T
	var hiddenContribution tensorTypes.Tensor
	if isBatch {
		// hiddenState: [batch_size, hidden_size], weight_hh: [4*hidden_size, hidden_size]
		// hiddenState @ weight_hh.T = [batch_size, 4*hidden_size]
		hiddenContribution = hiddenState.MatMulTransposed(weightHH, false, true, nil)
	} else {
		// hiddenState: [hidden_size], weight_hh: [4*hidden_size, hidden_size]
		hiddenStateReshaped := hiddenState.Reshape(tensor.NewShape(1, l.hiddenSize))
		hiddenContributionTemp := hiddenStateReshaped.MatMulTransposed(weightHH, false, true, nil)
		hiddenContribution = hiddenContributionTemp.Reshape(tensor.NewShape(4 * l.hiddenSize))
	}

	// Add contributions: gates = gates + hiddenContribution
	gates.Add(hiddenContribution)

	// Add bias (broadcast if needed)
	if isBatch {
		// gates: [batch_size, 4*hidden_size], bias: [4*hidden_size]
		// Broadcast bias to [batch_size, 4*hidden_size]
		biasBroadcast := bias.Reshape(tensor.NewShape(1, 4*l.hiddenSize))
		biasFull, err := biasBroadcast.BroadcastTo(gates.Shape())
		if err != nil {
			return fmt.Errorf("LSTM.computeForward: bias broadcast failed: %w", err)
		}
		gates.Add(biasFull)
	} else {
		// gates: [4*hidden_size], bias: [4*hidden_size]
		gates.Add(bias)
	}

	// Split gates into i, f, g, o using Slice operation
	// gates shape: [batch_size, 4*hidden_size] or [4*hidden_size]
	// Extract each gate by slicing dimension 1 (or 0 for 1D)
	sliceDim := 1
	if !isBatch {
		sliceDim = 0
	}

	iGate := gates.Slice(sliceDim, 0, l.hiddenSize)
	fGate := gates.Slice(sliceDim, l.hiddenSize, l.hiddenSize)
	gGate := gates.Slice(sliceDim, 2*l.hiddenSize, l.hiddenSize)
	oGate := gates.Slice(sliceDim, 3*l.hiddenSize, l.hiddenSize)

	// Apply activations
	// iGate = sigmoid(iGate)
	iGateSigmoid := iGate.Clone().Sigmoid(nil)
	iGate = iGateSigmoid

	// fGate = sigmoid(fGate)
	fGateSigmoid := fGate.Clone().Sigmoid(nil)
	fGate = fGateSigmoid

	// gGate = tanh(gGate)
	gGateTanh := gGate.Clone().Tanh(nil)
	gGate = gGateTanh

	// oGate = sigmoid(oGate)
	oGateSigmoid := oGate.Clone().Sigmoid(nil)
	oGate = oGateSigmoid

	// Update cell state: cell = fGate * cellState + iGate * gGate
	cellNew := cellState.Clone()
	cellNew.Mul(fGate)                 // cellNew = fGate * cellState
	iGateG := iGate.Clone().Mul(gGate) // iGate * gGate
	cellNew.Add(iGateG)                // cellNew = fGate * cellState + iGate * gGate

	// Update hidden state: hidden = oGate * tanh(cellNew)
	cellNewTanh := cellNew.Clone().Tanh(nil)
	outputNew := oGate.Clone().Mul(cellNewTanh) // output = oGate * tanh(cellNew)

	// Copy to output and cellState
	output.Copy(outputNew)
	cellState.Copy(cellNew)

	return nil
}

// ResetState resets the hidden state and cell state to zeros.
func (l *LSTM) ResetState() {
	if l == nil {
		return
	}
	if !tensor.IsNil(l.hiddenState) {
		l.hiddenState.Scale(0)
	}
	if !tensor.IsNil(l.cellState) {
		l.cellState.Scale(0)
	}
}

// SetState sets the hidden state and cell state.
func (l *LSTM) SetState(hiddenState, cellState tensorTypes.Tensor) error {
	if l == nil {
		return fmt.Errorf("LSTM.SetState: nil layer")
	}
	if tensor.IsNil(hiddenState) || tensor.IsNil(cellState) {
		return fmt.Errorf("LSTM.SetState: empty state tensor")
	}
	if !hiddenState.Shape().Equal(l.hiddenState.Shape()) {
		return fmt.Errorf("LSTM.SetState: hidden state shape mismatch")
	}
	if !cellState.Shape().Equal(l.cellState.Shape()) {
		return fmt.Errorf("LSTM.SetState: cell state shape mismatch")
	}
	l.hiddenState = hiddenState.Clone()
	l.cellState = cellState.Clone()
	return nil
}

// GetState returns the current hidden state and cell state.
func (l *LSTM) GetState() (hiddenState, cellState tensorTypes.Tensor) {
	if l == nil {
		return nil, nil
	}
	if tensor.IsNil(l.hiddenState) || tensor.IsNil(l.cellState) {
		return nil, nil
	}
	return l.hiddenState.Clone(), l.cellState.Clone()
}

// OutputShape returns the output shape for given input shape.
func (l *LSTM) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if l == nil {
		return nil, fmt.Errorf("LSTM.OutputShape: nil layer")
	}

	if len(inputShape) == 1 {
		if inputShape[0] != l.inputSize {
			return nil, fmt.Errorf("LSTM.OutputShape: input shape %v incompatible with inputSize %d", inputShape, l.inputSize)
		}
		return tensor.NewShape(l.hiddenSize), nil
	} else if len(inputShape) == 2 {
		if inputShape[1] != l.inputSize {
			return nil, fmt.Errorf("LSTM.OutputShape: input shape %v incompatible with inputSize %d", inputShape, l.inputSize)
		}
		return tensor.NewShape(inputShape[0], l.hiddenSize), nil
	}
	return nil, fmt.Errorf("LSTM.OutputShape: input must be 1D or 2D, got %dD", len(inputShape))
}

// Backward computes gradients w.r.t. input and parameters.
// Note: Full LSTM backward pass is complex and requires storing intermediate values.
// This is a simplified version that computes basic gradients.
func (l *LSTM) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if l == nil {
		return nil, fmt.Errorf("LSTM.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("LSTM.Backward: empty gradOutput")
	}

	// TODO: Implement full backward pass
	// This requires storing intermediate values from forward pass
	// For now, return error indicating not yet implemented
	return nil, fmt.Errorf("LSTM.Backward: not yet implemented")
}
