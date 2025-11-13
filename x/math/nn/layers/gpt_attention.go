package layers

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// GPTAttention represents a multi-head self-attention layer for GPT models.
// Implements causal self-attention for sequence processing in robot navigation.
type GPTAttention struct {
	Base
	embedDim  int     // Input embedding dimension
	numHeads  int     // Number of attention heads
	headDim   int     // Dimension per head (embedDim / numHeads)
	maxSeqLen int     // Maximum sequence length for causal masking
	dropout   float64 // Dropout probability (0.0 = no dropout)

	// Pre-computed causal mask for efficiency
	causalMask tensorTypes.Tensor // [maxSeqLen, maxSeqLen] boolean mask

	// Pre-allocated tensors for attention computation
	qkvBuffer   tensorTypes.Tensor // Buffer for Q, K, V projections [batch, seq, 3*embedDim]
	attnWeights tensorTypes.Tensor // Attention weights [batch, numHeads, seq, seq]
	attnOutput  tensorTypes.Tensor // Attention output before final projection [batch, seq, embedDim]
}

// NewGPTAttention creates a new GPT multi-head attention layer.
// embedDim: input embedding dimension (must be divisible by numHeads)
// numHeads: number of attention heads
// maxSeqLen: maximum sequence length for causal masking
// dropout: dropout probability for attention weights (0.0 = no dropout)
func NewGPTAttention(embedDim, numHeads, maxSeqLen int, dropout float64, opts ...Option) (*GPTAttention, error) {
	if embedDim <= 0 {
		return nil, fmt.Errorf("GPTAttention: embedDim must be positive, got %d", embedDim)
	}
	if numHeads <= 0 {
		return nil, fmt.Errorf("GPTAttention: numHeads must be positive, got %d", numHeads)
	}
	if embedDim%numHeads != 0 {
		return nil, fmt.Errorf("GPTAttention: embedDim %d must be divisible by numHeads %d", embedDim, numHeads)
	}
	if maxSeqLen <= 0 {
		return nil, fmt.Errorf("GPTAttention: maxSeqLen must be positive, got %d", maxSeqLen)
	}
	if dropout < 0.0 || dropout >= 1.0 {
		return nil, fmt.Errorf("GPTAttention: dropout must be in [0, 1), got %f", dropout)
	}

	headDim := embedDim / numHeads

	// Create Base without options first
	base := NewBase("gpt_attention")

	attention := &GPTAttention{
		Base:      base,
		embedDim:  embedDim,
		numHeads:  numHeads,
		headDim:   headDim,
		maxSeqLen: maxSeqLen,
		dropout:   dropout,
	}

	// Parse options first to get data type and any pre-set weights
	attention.Base.ParseOptions(opts...)

	// Initialize parameters
	dtype := attention.Base.DataType()

	// QKV projection matrix: [embedDim, 3*embedDim] (projects input to Q, K, V)
	attention.Base.initParam(types.ParamWeights)
	qkvParam, _ := attention.Base.Parameter(types.ParamWeights)
	qkvParam.Init(dtype, tensor.NewShape(embedDim, 3*embedDim), types.ParamWeights, embedDim, 3*embedDim, attention.Base.rng, attention.Base.CanLearn())
	attention.Base.SetParam(types.ParamWeights, qkvParam)

	// Output projection matrix: [embedDim, embedDim] (projects concatenated heads back to embedDim)
	attention.Base.initParam(types.ParamBiases)
	outParam, _ := attention.Base.Parameter(types.ParamBiases)
	outParam.Init(dtype, tensor.NewShape(embedDim, embedDim), types.ParamBiases, embedDim, embedDim, attention.Base.rng, attention.Base.CanLearn())
	attention.Base.SetParam(types.ParamBiases, outParam)

	// Create causal mask: [maxSeqLen, maxSeqLen]
	// mask[i, j] = 1.0 if j <= i (can attend), -inf otherwise
	attention.causalMask = tensor.New(dtype, tensor.NewShape(maxSeqLen, maxSeqLen))
	for i := 0; i < maxSeqLen; i++ {
		for j := 0; j < maxSeqLen; j++ {
			if j <= i {
				attention.causalMask.SetAt(0.0, i, j) // Can attend
			} else {
				attention.causalMask.SetAt(-math.MaxFloat32, i, j) // Cannot attend to future
			}
		}
	}

	return attention, nil
}

// Init initializes the layer, creating internal computation tensors.
// Input shape should be [batchSize, seqLen, embedDim] from positional encoding.
func (a *GPTAttention) Init(inputShape tensor.Shape) error {
	if a == nil {
		return fmt.Errorf("GPTAttention.Init: nil layer")
	}

	// Validate input shape: [batchSize, seqLen, embedDim] or [seqLen, embedDim]
	if len(inputShape) != 2 && len(inputShape) != 3 {
		return fmt.Errorf("GPTAttention.Init: input must be 2D or 3D, got %dD", len(inputShape))
	}

	var seqLen int
	if len(inputShape) == 2 {
		// Single sequence: [seqLen, embedDim]
		seqLen = inputShape[0]
		if inputShape[1] != a.embedDim {
			return fmt.Errorf("GPTAttention.Init: input embedDim %d doesn't match layer embedDim %d", inputShape[1], a.embedDim)
		}
	} else {
		// Batch of sequences: [batchSize, seqLen, embedDim]
		seqLen = inputShape[1]
		if inputShape[2] != a.embedDim {
			return fmt.Errorf("GPTAttention.Init: input embedDim %d doesn't match layer embedDim %d", inputShape[2], a.embedDim)
		}
	}

	// Validate sequence length
	if seqLen > a.maxSeqLen {
		return fmt.Errorf("GPTAttention.Init: sequence length %d exceeds maxSeqLen %d", seqLen, a.maxSeqLen)
	}

	// Output shape matches input shape
	var outputShape tensor.Shape
	if len(inputShape) == 2 {
		outputShape = tensor.NewShape(seqLen, a.embedDim)
	} else {
		outputShape = tensor.NewShape(inputShape[0], seqLen, a.embedDim)
	}

	// Allocate output tensor
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	a.Base.AllocOutput(outputShape, outputSize)

	// Pre-allocate computation buffers
	dtype := a.Base.DataType()
	if len(inputShape) == 3 {
		batchSize := inputShape[0]
		// QKV buffer: [batchSize, seqLen, 3*embedDim]
		a.qkvBuffer = tensor.New(dtype, tensor.NewShape(batchSize, seqLen, 3*a.embedDim))
		// Attention weights: [batchSize, numHeads, seqLen, seqLen]
		a.attnWeights = tensor.New(dtype, tensor.NewShape(batchSize, a.numHeads, seqLen, seqLen))
		// Attention output: [batchSize, seqLen, embedDim]
		a.attnOutput = tensor.New(dtype, tensor.NewShape(batchSize, seqLen, a.embedDim))
	} else {
		// Single sequence case
		a.qkvBuffer = tensor.New(dtype, tensor.NewShape(seqLen, 3*a.embedDim))
		a.attnWeights = tensor.New(dtype, tensor.NewShape(a.numHeads, seqLen, seqLen))
		a.attnOutput = tensor.New(dtype, tensor.NewShape(seqLen, a.embedDim))
	}

	return nil
}

// Forward computes the forward pass: multi-head self-attention with causal masking.
// Input: embeddings with shape [batchSize, seqLen, embedDim] or [seqLen, embedDim]
// Output: attention output with same shape
func (a *GPTAttention) Forward(input tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("GPTAttention.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GPTAttention.Forward: empty input")
	}

	// Store input
	a.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := a.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("GPTAttention.Forward: output not allocated, must call Init first")
	}

	// Get parameters
	qkvParam, ok := a.Base.Parameter(types.ParamWeights)
	if !ok || tensor.IsNil(qkvParam.Data) {
		return nil, fmt.Errorf("GPTAttention.Forward: QKV weights not initialized")
	}

	outParam, ok := a.Base.Parameter(types.ParamBiases)
	if !ok || tensor.IsNil(outParam.Data) {
		return nil, fmt.Errorf("GPTAttention.Forward: output weights not initialized")
	}

	// Compute multi-head attention
	if err := a.computeAttention(input, qkvParam.Data, outParam.Data, output); err != nil {
		return nil, fmt.Errorf("GPTAttention.Forward: attention computation failed: %w", err)
	}

	// Store output
	a.Base.StoreOutput(output)
	return output, nil
}

// computeAttention performs the complete multi-head attention computation.
func (a *GPTAttention) computeAttention(input, qkvWeights, outWeights, output tensorTypes.Tensor) error {
	inputShape := input.Shape()
	var batchSize, seqLen int

	if len(inputShape) == 2 {
		// Single sequence: [seqLen, embedDim]
		seqLen = inputShape[0]
		batchSize = 1
	} else {
		// Batch: [batchSize, seqLen, embedDim]
		batchSize = inputShape[0]
		seqLen = inputShape[1]
	}

	// 1. Linear projection to get Q, K, V
	if err := a.projectQKV(input, qkvWeights); err != nil {
		return fmt.Errorf("QKV projection failed: %w", err)
	}

	// 2. Split QKV into separate tensors and reshape for multi-head
	q, k, v, err := a.splitAndReshapeQKV(batchSize, seqLen)
	if err != nil {
		return fmt.Errorf("QKV split/reshape failed: %w", err)
	}

	// 3. Compute scaled dot-product attention for each head
	if err := a.computeScaledDotProductAttention(q, k, v, batchSize, seqLen); err != nil {
		return fmt.Errorf("scaled dot-product attention failed: %w", err)
	}

	// 4. Concatenate heads and apply output projection
	if err := a.concatenateAndProject(q, outWeights, output, batchSize, seqLen); err != nil {
		return fmt.Errorf("concatenation and projection failed: %w", err)
	}

	return nil
}

// projectQKV projects input to Q, K, V concatenated tensor.
func (a *GPTAttention) projectQKV(input, qkvWeights tensorTypes.Tensor) error {
	inputShape := input.Shape()

	if len(inputShape) == 2 {
		// Single sequence: [seqLen, embedDim] @ [embedDim, 3*embedDim] -> [seqLen, 3*embedDim]
		input.MatMul(a.qkvBuffer, qkvWeights)
	} else {
		// Batch: reshape input to [batch*seq, embedDim] for matrix multiplication
		batchSize := inputShape[0]
		seqLen := inputShape[1]
		inputReshaped := input.Reshape(nil, tensor.NewShape(batchSize*seqLen, a.embedDim))
		qkvReshaped := a.qkvBuffer.Reshape(nil, tensor.NewShape(batchSize*seqLen, 3*a.embedDim))
		inputReshaped.MatMul(qkvReshaped, qkvWeights)
		// Reshape back
		a.qkvBuffer = qkvReshaped.Reshape(nil, tensor.NewShape(batchSize, seqLen, 3*a.embedDim))
	}

	return nil
}

// splitAndReshapeQKV splits QKV buffer and reshapes for multi-head attention.
func (a *GPTAttention) splitAndReshapeQKV(batchSize, seqLen int) (q, k, v tensorTypes.Tensor, err error) {
	// Split QKV buffer into Q, K, V along last dimension
	// QKV buffer: [batch, seq, 3*embedDim] or [seq, 3*embedDim]
	var qSlice, kSlice, vSlice tensorTypes.Tensor

	if batchSize == 1 && len(a.qkvBuffer.Shape()) == 2 {
		// Single sequence case
		qSlice = a.qkvBuffer.Slice(nil, 1, 0, a.embedDim)              // [seq, embedDim]
		kSlice = a.qkvBuffer.Slice(nil, 1, a.embedDim, 2*a.embedDim)   // [seq, embedDim]
		vSlice = a.qkvBuffer.Slice(nil, 1, 2*a.embedDim, 3*a.embedDim) // [seq, embedDim]

		// Reshape for multi-head: [seq, embedDim] -> [numHeads, seq, headDim]
		q = qSlice.Reshape(nil, tensor.NewShape(a.numHeads, seqLen, a.headDim))
		k = kSlice.Reshape(nil, tensor.NewShape(a.numHeads, seqLen, a.headDim))
		v = vSlice.Reshape(nil, tensor.NewShape(a.numHeads, seqLen, a.headDim))
	} else {
		// Batch case
		qSlice = a.qkvBuffer.Slice(nil, 2, 0, a.embedDim)              // [batch, seq, embedDim]
		kSlice = a.qkvBuffer.Slice(nil, 2, a.embedDim, 2*a.embedDim)   // [batch, seq, embedDim]
		vSlice = a.qkvBuffer.Slice(nil, 2, 2*a.embedDim, 3*a.embedDim) // [batch, seq, embedDim]

		// Reshape for multi-head: [batch, seq, embedDim] -> [batch, numHeads, seq, headDim]
		q = qSlice.Reshape(nil, tensor.NewShape(batchSize, a.numHeads, seqLen, a.headDim))
		k = kSlice.Reshape(nil, tensor.NewShape(batchSize, a.numHeads, seqLen, a.headDim))
		v = vSlice.Reshape(nil, tensor.NewShape(batchSize, a.numHeads, seqLen, a.headDim))
	}

	return q, k, v, nil
}

// computeScaledDotProductAttention computes attention for each head.
func (a *GPTAttention) computeScaledDotProductAttention(q, k, v tensorTypes.Tensor, batchSize, seqLen int) error {
	scale := 1.0 / math.Sqrt(float64(a.headDim))

	if batchSize == 1 && len(q.Shape()) == 3 {
		// Single sequence case: Q, K, V are [numHeads, seq, headDim]
		for h := 0; h < a.numHeads; h++ {
			// Extract head: Q_h, K_h, V_h [seq, headDim]
			qh := q.Slice(nil, 0, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim)) // [seq, headDim]
			kh := k.Slice(nil, 0, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim)) // [seq, headDim]
			vh := v.Slice(nil, 0, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim)) // [seq, headDim]

			// Attention weights: Q_h @ K_h^T / sqrt(headDim) -> [seq, seq]
			attnHead := a.attnWeights.Slice(nil, 0, h, h+1).Reshape(nil, tensor.NewShape(seqLen, seqLen))
			qh.MatMulTransposed(attnHead, kh, false, true) // Q @ K^T
			attnHead.ScalarMul(attnHead, scale)            // Scale

			// Apply causal mask
			causalMask := a.causalMask.Slice(nil, 0, 0, seqLen).Slice(nil, 1, 0, seqLen) // [seq, seq]
			attnHead.Add(attnHead, causalMask)

			// Softmax along last dimension (sequence dimension)
			attnHead.Softmax(1, attnHead)

			// Apply dropout if enabled
			if a.dropout > 0.0 {
				// TODO: Implement dropout
			}

			// Apply attention to values: attn @ V_h -> [seq, headDim]
			attnOutHead := a.attnOutput.Slice(nil, 0, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim))
			attnHead.MatMul(attnOutHead, vh)
		}
	} else {
		// Batch case: Q, K, V are [batch, numHeads, seq, headDim]
		for b := 0; b < batchSize; b++ {
			for h := 0; h < a.numHeads; h++ {
				// Extract head for this batch: Q_h, K_h, V_h [seq, headDim]
				qh := q.Slice(nil, 0, b, b+1).Slice(nil, 1, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim))
				kh := k.Slice(nil, 0, b, b+1).Slice(nil, 1, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim))
				vh := v.Slice(nil, 0, b, b+1).Slice(nil, 1, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim))

				// Attention weights: Q_h @ K_h^T / sqrt(headDim) -> [seq, seq]
				attnHead := a.attnWeights.Slice(nil, 0, b, b+1).Slice(nil, 1, h, h+1).Reshape(nil, tensor.NewShape(seqLen, seqLen))
				qh.MatMulTransposed(attnHead, kh, false, true) // Q @ K^T
				attnHead.ScalarMul(attnHead, scale)            // Scale

				// Apply causal mask
				causalMask := a.causalMask.Slice(nil, 0, 0, seqLen).Slice(nil, 1, 0, seqLen) // [seq, seq]
				attnHead.Add(attnHead, causalMask)

				// Softmax along last dimension (sequence dimension)
				attnHead.Softmax(1, attnHead)

				// Apply attention to values: attn @ V_h -> [seq, headDim]
				attnOutHead := a.attnOutput.Slice(nil, 0, b, b+1).Slice(nil, 1, h, h+1).Reshape(nil, tensor.NewShape(seqLen, a.headDim))
				attnHead.MatMul(attnOutHead, vh)
			}
		}
	}

	return nil
}

// concatenateAndProject concatenates attention heads and applies output projection.
func (a *GPTAttention) concatenateAndProject(q, outWeights, output tensorTypes.Tensor, batchSize, seqLen int) error {
	// Concatenate heads: [batch, numHeads, seq, headDim] -> [batch, seq, embedDim]
	// or [numHeads, seq, headDim] -> [seq, embedDim]
	if batchSize == 1 && len(a.attnOutput.Shape()) == 3 {
		// Single sequence: [numHeads, seq, headDim] -> [seq, embedDim]
		// Transpose and reshape: [numHeads, seq, headDim] -> [seq, numHeads*headDim]
		attnTransposed := a.attnOutput.Transpose(nil, []int{1, 0, 2}) // [seq, numHeads, headDim]
		concatHeads := attnTransposed.Reshape(nil, tensor.NewShape(seqLen, a.embedDim))
		// Apply output projection: [seq, embedDim] @ [embedDim, embedDim] -> [seq, embedDim]
		concatHeads.MatMul(output, outWeights)
	} else {
		// Batch: [batch, numHeads, seq, headDim] -> [batch, seq, embedDim]
		// Transpose and reshape: [batch, numHeads, seq, headDim] -> [batch, seq, numHeads*headDim]
		attnTransposed := a.attnOutput.Transpose(nil, []int{0, 2, 1, 3}) // [batch, seq, numHeads, headDim]
		concatHeads := attnTransposed.Reshape(nil, tensor.NewShape(batchSize, seqLen, a.embedDim))
		// Apply output projection: [batch, seq, embedDim] @ [embedDim, embedDim] -> [batch, seq, embedDim]
		// Reshape for matrix multiplication
		concatReshaped := concatHeads.Reshape(nil, tensor.NewShape(batchSize*seqLen, a.embedDim))
		outputReshaped := output.Reshape(nil, tensor.NewShape(batchSize*seqLen, a.embedDim))
		concatReshaped.MatMul(outputReshaped, outWeights)
	}

	return nil
}

// Backward computes gradients for multi-head attention.
// This is complex and involves computing gradients through all the attention operations.
func (a *GPTAttention) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("GPTAttention.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("GPTAttention.Backward: empty gradOutput")
	}

	input := a.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GPTAttention.Backward: input not stored, must call Forward first")
	}

	// For now, implement a simplified backward pass
	// In a full implementation, this would need to backpropagate through:
	// - Output projection
	// - Attention mechanism (including softmax and scaling)
	// - QKV projections
	// This is quite complex and would require careful implementation

	// Placeholder: pass gradient through unchanged for now
	gradInput := a.Base.Grad()
	if tensor.IsNil(gradInput) {
		gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
	}
	gradOutput.Copy(gradInput)
	a.Base.StoreGrad(gradInput)

	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
// Output shape matches input shape exactly.
func (a *GPTAttention) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if a == nil {
		return nil, fmt.Errorf("GPTAttention.OutputShape: nil layer")
	}

	if len(inputShape) != 2 && len(inputShape) != 3 {
		return nil, fmt.Errorf("GPTAttention.OutputShape: input must be 2D or 3D, got %dD", len(inputShape))
	}

	var embedDim int
	if len(inputShape) == 2 {
		embedDim = inputShape[1]
	} else {
		embedDim = inputShape[2]
	}

	if embedDim != a.embedDim {
		return nil, fmt.Errorf("GPTAttention.OutputShape: input embedDim %d doesn't match layer embedDim %d", embedDim, a.embedDim)
	}

	return inputShape.Clone(), nil
}

// QKVWeights returns the QKV projection weights tensor.
func (a *GPTAttention) QKVWeights() tensorTypes.Tensor {
	if a == nil {
		return tensor.Empty(tensor.DTFP32)
	}
	weightParam := a.Base.Weights()
	if tensor.IsNil(weightParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return weightParam.Data
}

// OutputWeights returns the output projection weights tensor.
func (a *GPTAttention) OutputWeights() tensorTypes.Tensor {
	if a == nil {
		return tensor.Empty(tensor.DTFP32)
	}
	biasParam := a.Base.Biases()
	if tensor.IsNil(biasParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return biasParam.Data
}
