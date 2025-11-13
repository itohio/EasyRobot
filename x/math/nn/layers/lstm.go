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
	// Pre-allocated gate activation tensors for optimization
	iGateSigmoid tensorTypes.Tensor // Input gate activation [hidden_size] or [batch_size, hidden_size]
	fGateSigmoid tensorTypes.Tensor // Forget gate activation [hidden_size] or [batch_size, hidden_size]
	gGateTanh    tensorTypes.Tensor // Cell gate activation [hidden_size] or [batch_size, hidden_size]
	oGateSigmoid tensorTypes.Tensor // Output gate activation [hidden_size] or [batch_size, hidden_size]
	// Pre-allocated intermediate computation tensors for optimization
	gatesTmp              tensorTypes.Tensor // For input @ weight_ih.T result [batch_size, 4*hidden_size] or [4*hidden_size]
	hiddenContributionTmp tensorTypes.Tensor // For hiddenState @ weight_hh.T result [batch_size, 4*hidden_size] or [4*hidden_size]
	biasFull              tensorTypes.Tensor // For broadcast bias result [batch_size, 4*hidden_size]
	cellNew               tensorTypes.Tensor // For updated cell state [hidden_size] or [batch_size, hidden_size]
	iGateG                tensorTypes.Tensor // For iGate * gGate [hidden_size] or [batch_size, hidden_size]
	cellNewTanhTmp        tensorTypes.Tensor // For tanh(cellNew) [hidden_size] or [batch_size, hidden_size]
	outputNew             tensorTypes.Tensor // For oGate * tanh(cellNew) [hidden_size] or [batch_size, hidden_size]
	// Pre-allocated tensors for non-batch case
	inputReshaped          tensorTypes.Tensor // For input reshape [1, input_size]
	gatesTemp              tensorTypes.Tensor // For MatMul result [1, 4*hidden_size]
	gates1D                tensorTypes.Tensor // For gates result [4*hidden_size]
	hiddenStateReshaped    tensorTypes.Tensor // For hiddenState reshape [1, hidden_size]
	hiddenContributionTemp tensorTypes.Tensor // For MatMul result [1, 4*hidden_size]
	hiddenContribution1D   tensorTypes.Tensor // For hiddenContribution result [4*hidden_size]
	gatesResult1D          tensorTypes.Tensor // For gates + hiddenContribution result [4*hidden_size]
	gatesResult1DBias      tensorTypes.Tensor // For gates + bias result [4*hidden_size]
	biasReshaped           tensorTypes.Tensor // For bias reshape [1, 4*hidden_size]
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

	// Pre-allocate gate activation tensors to avoid Clone() operations
	// These tensors will be reused across forward passes
	// Gate slices have the same shape as output (since they're slices along the gates dimension)
	dtype := l.Base.DataType()
	l.iGateSigmoid = tensor.New(dtype, outputShape)
	l.fGateSigmoid = tensor.New(dtype, outputShape)
	l.gGateTanh = tensor.New(dtype, outputShape)
	l.oGateSigmoid = tensor.New(dtype, outputShape)

	// Pre-allocate intermediate computation tensors to avoid allocations in computeForward
	// Handle both batch and non-batch cases
	// Bias reshape tensor is needed for both cases
	biasReshapedShape := tensor.NewShape(1, 4*l.hiddenSize)
	l.biasReshaped = tensor.New(dtype, biasReshapedShape)

	if len(inputShape) == 2 {
		// Batch case: gates have shape [batch_size, 4*hidden_size]
		gatesShape := tensor.NewShape(inputShape[0], 4*l.hiddenSize)
		l.gatesTmp = tensor.New(dtype, gatesShape)
		l.hiddenContributionTmp = tensor.New(dtype, gatesShape)
		l.biasFull = tensor.New(dtype, gatesShape)
	} else {
		// Non-batch case: pre-allocate all tensors needed for 1D processing
		// Gates shape: [4*hidden_size]
		gates1DShape := tensor.NewShape(4 * l.hiddenSize)
		l.gates1D = tensor.New(dtype, gates1DShape)
		l.hiddenContribution1D = tensor.New(dtype, gates1DShape)
		l.gatesResult1D = tensor.New(dtype, gates1DShape)
		l.gatesResult1DBias = tensor.New(dtype, gates1DShape)

		// Reshape intermediates: [1, input_size] and [1, hidden_size]
		inputReshapedShape := tensor.NewShape(1, l.inputSize)
		l.inputReshaped = tensor.New(dtype, inputReshapedShape)
		gatesTempShape := tensor.NewShape(1, 4*l.hiddenSize)
		l.gatesTemp = tensor.New(dtype, gatesTempShape)

		hiddenReshapedShape := tensor.NewShape(1, l.hiddenSize)
		l.hiddenStateReshaped = tensor.New(dtype, hiddenReshapedShape)
		hiddenContributionTempShape := tensor.NewShape(1, 4*l.hiddenSize)
		l.hiddenContributionTemp = tensor.New(dtype, hiddenContributionTempShape)
	}

	// Pre-allocate state update tensors (same shape as output)
	l.cellNew = tensor.New(dtype, outputShape)
	l.iGateG = tensor.New(dtype, outputShape)
	l.cellNewTanhTmp = tensor.New(dtype, outputShape)
	l.outputNew = tensor.New(dtype, outputShape)

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

	// Hidden and cell states should already be initialized in Init()
	if tensor.IsNil(l.hiddenState) {
		return nil, fmt.Errorf("LSTM.Forward: hidden state not initialized, must call Init first")
	}
	if !l.hiddenState.Shape().Equal(output.Shape()) {
		return nil, fmt.Errorf("LSTM.Forward: hidden state shape mismatch, expected %v, got %v", output.Shape(), l.hiddenState.Shape())
	}

	// Compute forward pass
	if err := l.computeForward(input, weightIHParam.Data, weightHHParam.Data, biasParam.Data,
		l.hiddenState, l.cellState, output); err != nil {
		return nil, fmt.Errorf("LSTM.Forward: computation failed: %w", err)
	}

	// Update internal state using Copy (hiddenState is pre-allocated in Init)
	l.hiddenState.Copy(output)

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
		// Use pre-allocated gatesTmp (allocated in Init)
		gates = input.MatMulTransposed(l.gatesTmp, weightIH, false, true)
	} else {
		// input: [input_size], weight_ih: [4*hidden_size, input_size]
		// Reshape input to [1, input_size] for matrix multiplication (use pre-allocated destination)
		input.Reshape(l.inputReshaped, l.inputReshaped.Shape())
		// Use pre-allocated gatesTemp for MatMul result
		l.inputReshaped.MatMulTransposed(l.gatesTemp, weightIH, false, true)
		// Reshape back to [4*hidden_size] (use pre-allocated destination)
		l.gatesTemp.Reshape(l.gates1D, l.gates1D.Shape())
		gates = l.gates1D
	}

	// Add hidden contribution: hidden @ weight_hh.T
	var hiddenContribution tensorTypes.Tensor
	if isBatch {
		// hiddenState: [batch_size, hidden_size], weight_hh: [4*hidden_size, hidden_size]
		// hiddenState @ weight_hh.T = [batch_size, 4*hidden_size]
		// Use pre-allocated hiddenContributionTmp (allocated in Init)
		hiddenContribution = hiddenState.MatMulTransposed(l.hiddenContributionTmp, weightHH, false, true)
	} else {
		// hiddenState: [hidden_size], weight_hh: [4*hidden_size, hidden_size]
		// Reshape hiddenState to [1, hidden_size] (use pre-allocated destination)
		hiddenState.Reshape(l.hiddenStateReshaped, l.hiddenStateReshaped.Shape())
		// Use pre-allocated hiddenContributionTemp for MatMul result
		l.hiddenStateReshaped.MatMulTransposed(l.hiddenContributionTemp, weightHH, false, true)
		// Reshape back to [4*hidden_size] (use pre-allocated destination)
		l.hiddenContributionTemp.Reshape(l.hiddenContribution1D, l.hiddenContribution1D.Shape())
		hiddenContribution = l.hiddenContribution1D
	}

	// Add contributions: gates = gates + hiddenContribution
	if isBatch {
		// gates is already in gatesTmp, so we can add in-place to gatesTmp
		gates.Add(gates, hiddenContribution)
	} else {
		// Use pre-allocated gatesResult1D for result
		gates.Add(l.gatesResult1D, hiddenContribution)
		gates = l.gatesResult1D
	}

	// Add bias (broadcast if needed)
	if isBatch {
		// gates: [batch_size, 4*hidden_size], bias: [4*hidden_size]
		// Broadcast bias to [batch_size, 4*hidden_size]
		// Use pre-allocated biasFull (allocated in Init)
		// Reshape bias to [1, 4*hidden_size] (use pre-allocated destination)
		bias.Reshape(l.biasReshaped, l.biasReshaped.Shape())
		// Broadcast to gates shape using pre-allocated destination
		l.biasReshaped.BroadcastTo(l.biasFull, gates.Shape())
		// Add bias to gates (gates is already in gatesTmp, so add in-place)
		gates.Add(gates, l.biasFull)
	} else {
		// gates: [4*hidden_size], bias: [4*hidden_size]
		// Use pre-allocated gatesResult1DBias for result
		gates.Add(l.gatesResult1DBias, bias)
		gates = l.gatesResult1DBias
	}

	// Split gates into i, f, g, o using Slice operation
	// gates shape: [batch_size, 4*hidden_size] or [4*hidden_size]
	// Extract each gate by slicing dimension 1 (or 0 for 1D)
	sliceDim := 1
	if !isBatch {
		sliceDim = 0
	}

	iGate := gates.Slice(nil, sliceDim, 0, l.hiddenSize)
	fGate := gates.Slice(nil, sliceDim, l.hiddenSize, l.hiddenSize)
	gGate := gates.Slice(nil, sliceDim, 2*l.hiddenSize, l.hiddenSize)
	oGate := gates.Slice(nil, sliceDim, 3*l.hiddenSize, l.hiddenSize)

	// Apply activations using pre-allocated tensors (eliminates Clone() operations)
	// Gate slices have the same shape as the pre-allocated tensors (outputShape)
	// iGate = sigmoid(iGate)
	iGate.Sigmoid(l.iGateSigmoid)
	iGate = l.iGateSigmoid

	// fGate = sigmoid(fGate)
	fGate.Sigmoid(l.fGateSigmoid)
	fGate = l.fGateSigmoid

	// gGate = tanh(gGate)
	gGate.Tanh(l.gGateTanh)
	gGate = l.gGateTanh

	// oGate = sigmoid(oGate)
	oGate.Sigmoid(l.oGateSigmoid)
	oGate = l.oGateSigmoid

	// Update cell state: cell = fGate * cellState + iGate * gGate
	// Use pre-allocated tensors (cellNew and iGateG)
	cellState.Multiply(l.cellNew, fGate) // cellNew = cellState * fGate
	iGate.Multiply(l.iGateG, gGate)      // iGateG = iGate * gGate
	l.cellNew.Add(l.cellNew, l.iGateG)   // cellNew = cellNew + iGateG (in-place)

	// Update hidden state: hidden = oGate * tanh(cellNew)
	// Use pre-allocated tensors (cellNewTanhTmp and outputNew)
	cellNewTanh := l.cellNew.Tanh(l.cellNewTanhTmp)
	oGate.Multiply(l.outputNew, cellNewTanh) // outputNew = oGate * cellNewTanh

	// Copy to output and cellState
	output.Copy(l.outputNew)
	cellState.Copy(l.cellNew)

	return nil
}

// ResetState resets the hidden state and cell state to zeros.
func (l *LSTM) ResetState() {
	if l == nil {
		return
	}
	if !tensor.IsNil(l.hiddenState) {
		l.hiddenState.ScalarMul(l.hiddenState, 0)
	}
	if !tensor.IsNil(l.cellState) {
		l.cellState.ScalarMul(l.cellState, 0)
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
