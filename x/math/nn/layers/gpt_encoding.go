package layers

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// GPTEncoding represents a positional encoding layer for GPT models.
// Adds position information to token embeddings for sequence processing.
type GPTEncoding struct {
	Base
	maxSeqLen int  // Maximum sequence length supported
	embedDim  int  // Embedding dimension (must match embedding layer)
	learned   bool // Whether to use learned positional encodings instead of sinusoidal
}

// NewGPTEncoding creates a new GPT positional encoding layer.
// maxSeqLen: maximum sequence length this layer will handle
// embedDim: embedding dimension (must match token embeddings)
// learned: if true, uses learned positional embeddings; if false, uses sinusoidal
func NewGPTEncoding(maxSeqLen, embedDim int, learned bool, opts ...Option) (*GPTEncoding, error) {
	if maxSeqLen <= 0 {
		return nil, fmt.Errorf("GPTEncoding: maxSeqLen must be positive, got %d", maxSeqLen)
	}
	if embedDim <= 0 {
		return nil, fmt.Errorf("GPTEncoding: embedDim must be positive, got %d", embedDim)
	}

	// Create Base without options first
	base := NewBase("gpt_encoding")

	encoding := &GPTEncoding{
		Base:      base,
		maxSeqLen: maxSeqLen,
		embedDim:  embedDim,
		learned:   learned,
	}

	// Parse options first to get data type and any pre-set weights
	encoding.Base.ParseOptions(opts...)

	// Initialize positional encoding parameter
	// Shape: [maxSeqLen, embedDim] - each row is position encoding for that position
	dtype := encoding.Base.DataType()

	if encoding.learned {
		// Learned positional embeddings
		encoding.Base.initParam(types.ParamWeights)
		weightParam, _ := encoding.Base.Parameter(types.ParamWeights)
		weightParam.Init(dtype, tensor.NewShape(maxSeqLen, embedDim), types.ParamWeights, maxSeqLen, embedDim, encoding.Base.rng, encoding.Base.CanLearn())
		encoding.Base.SetParam(types.ParamWeights, weightParam)
	} else {
		// Fixed sinusoidal positional encodings (not trainable)
		sinusoidalEncodings := encoding.createSinusoidalEncodings(dtype)
		encoding.Base.initParam(types.ParamWeights)
		weightParam, _ := encoding.Base.Parameter(types.ParamWeights)
		weightParam.Data = sinusoidalEncodings
		weightParam.RequiresGrad = false // Fixed encodings are not trainable
		encoding.Base.SetParam(types.ParamWeights, weightParam)
	}

	return encoding, nil
}

// createSinusoidalEncodings creates the fixed sinusoidal positional encodings.
// Uses the standard GPT positional encoding formula:
// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
func (e *GPTEncoding) createSinusoidalEncodings(dtype tensor.DataType) tensorTypes.Tensor {
	encodings := tensor.New(dtype, tensor.NewShape(e.maxSeqLen, e.embedDim))

	// Fill with sinusoidal encodings
	for pos := 0; pos < e.maxSeqLen; pos++ {
		for i := 0; i < e.embedDim; i += 2 {
			// Even indices: sin(pos / 10000^(2i/d_model))
			denom := math.Pow(10000.0, float64(2*i)/float64(e.embedDim))
			sinVal := math.Sin(float64(pos) / denom)
			encodings.SetAt(sinVal, pos, i)

			// Odd indices: cos(pos / 10000^(2i/d_model))
			if i+1 < e.embedDim {
				cosVal := math.Cos(float64(pos) / denom)
				encodings.SetAt(cosVal, pos, i+1)
			}
		}
	}

	return encodings
}

// Init initializes the layer, creating internal computation tensors.
// Input shape should be [batchSize, seqLen, embedDim] from embedding layer.
func (e *GPTEncoding) Init(inputShape tensor.Shape) error {
	if e == nil {
		return fmt.Errorf("GPTEncoding.Init: nil layer")
	}

	// Validate input shape: [batchSize, seqLen, embedDim] or [seqLen, embedDim]
	if len(inputShape) != 2 && len(inputShape) != 3 {
		return fmt.Errorf("GPTEncoding.Init: input must be 2D or 3D, got %dD", len(inputShape))
	}

	var seqLen int
	if len(inputShape) == 2 {
		// Single sequence: [seqLen, embedDim]
		seqLen = inputShape[0]
		if inputShape[1] != e.embedDim {
			return fmt.Errorf("GPTEncoding.Init: input embedDim %d doesn't match layer embedDim %d", inputShape[1], e.embedDim)
		}
	} else {
		// Batch of sequences: [batchSize, seqLen, embedDim]
		seqLen = inputShape[1]
		if inputShape[2] != e.embedDim {
			return fmt.Errorf("GPTEncoding.Init: input embedDim %d doesn't match layer embedDim %d", inputShape[2], e.embedDim)
		}
	}

	// Validate sequence length
	if seqLen > e.maxSeqLen {
		return fmt.Errorf("GPTEncoding.Init: sequence length %d exceeds maxSeqLen %d", seqLen, e.maxSeqLen)
	}

	// Output shape matches input shape
	var outputShape tensor.Shape
	if len(inputShape) == 2 {
		outputShape = tensor.NewShape(seqLen, e.embedDim)
	} else {
		outputShape = tensor.NewShape(inputShape[0], seqLen, e.embedDim)
	}

	// Allocate output tensor
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	e.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass: adds positional encodings to input embeddings.
// Input: embeddings with shape [batchSize, seqLen, embedDim] or [seqLen, embedDim]
// Output: position-encoded embeddings with same shape
func (e *GPTEncoding) Forward(input tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if e == nil {
		return nil, fmt.Errorf("GPTEncoding.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GPTEncoding.Forward: empty input")
	}

	// Store input
	e.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := e.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("GPTEncoding.Forward: output not allocated, must call Init first")
	}

	// Get positional encodings
	weightParam, ok := e.Base.Parameter(types.ParamWeights)
	if !ok || tensor.IsNil(weightParam.Data) {
		return nil, fmt.Errorf("GPTEncoding.Forward: positional encodings not initialized")
	}

	// Add positional encodings to input
	if err := e.addPositionalEncodings(input, weightParam.Data, output); err != nil {
		return nil, fmt.Errorf("GPTEncoding.Forward: positional encoding addition failed: %w", err)
	}

	// Store output
	e.Base.StoreOutput(output)
	return output, nil
}

// addPositionalEncodings adds positional encodings to the input embeddings.
// For each position in the sequence, adds the corresponding positional encoding vector.
func (e *GPTEncoding) addPositionalEncodings(input, posEncodings, output tensorTypes.Tensor) error {
	inputShape := input.Shape()

	if len(inputShape) == 2 {
		// Single sequence: [seqLen, embedDim]
		seqLen := inputShape[0]

		// Copy input to output, then add positional encodings
		input.Copy(output)

		for pos := 0; pos < seqLen; pos++ {
			// Get positional encoding for this position: posEncodings[pos, :] -> [1, embedDim]
			posEncoding := posEncodings.Slice(nil, 0, pos, pos+1) // [1, embedDim]

			// Add to corresponding output position: output[pos, :] += posEncoding[0, :]
			outputSlice := output.Slice(nil, 0, pos, pos+1) // [1, embedDim]
			outputSlice.Add(outputSlice, posEncoding)
		}
	} else if len(inputShape) == 3 {
		// Batch of sequences: [batchSize, seqLen, embedDim]
		batchSize := inputShape[0]
		seqLen := inputShape[1]

		// Copy input to output, then add positional encodings
		input.Copy(output)

		for b := 0; b < batchSize; b++ {
			for pos := 0; pos < seqLen; pos++ {
				// Get positional encoding for this position: posEncodings[pos, :] -> [1, embedDim]
				posEncoding := posEncodings.Slice(nil, 0, pos, pos+1) // [1, embedDim]

				// Add to corresponding output position: output[b, pos, :] += posEncoding[0, :]
				// Reshape the slice to match posEncoding shape for addition
				outputSlice := output.Slice(nil, 1, pos, pos+1).Slice(nil, 0, b, b+1)  // [1, 1, embedDim]
				outputSlice = outputSlice.Reshape(nil, tensor.NewShape(1, e.embedDim)) // [1, embedDim]
				outputSlice.Add(outputSlice, posEncoding)
			}
		}
	}

	return nil
}

// Backward computes gradients.
// For sinusoidal encodings (not learned), no gradients are computed.
// For learned encodings, gradients are computed and accumulated.
func (e *GPTEncoding) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if e == nil {
		return nil, fmt.Errorf("GPTEncoding.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("GPTEncoding.Backward: empty gradOutput")
	}

	input := e.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GPTEncoding.Backward: input not stored, must call Forward first")
	}

	// For learned positional encodings, compute gradients
	if e.learned && e.Base.CanLearn() {
		weightParam, ok := e.Base.Parameter(types.ParamWeights)
		if !ok || tensor.IsNil(weightParam.Data) {
			return nil, fmt.Errorf("GPTEncoding.Backward: positional encodings not initialized")
		}

		if weightParam.RequiresGrad {
			if tensor.IsNil(weightParam.Grad) {
				weightParam.Grad = tensor.New(weightParam.Data.DataType(), weightParam.Data.Shape())
			}

			// Accumulate gradients for positional encodings
			if err := e.accumulatePositionalGradients(gradOutput, weightParam.Grad); err != nil {
				return nil, fmt.Errorf("GPTEncoding.Backward: gradient accumulation failed: %w", err)
			}

			e.Base.SetParam(types.ParamWeights, weightParam)
		}
	}

	// Gradient w.r.t. input is the same as gradOutput (positional encodings are additive)
	gradInput := e.Base.Grad()
	if tensor.IsNil(gradInput) {
		gradInput = tensor.New(gradOutput.DataType(), gradOutput.Shape())
	}
	gradOutput.Copy(gradInput)
	e.Base.StoreGrad(gradInput)

	return gradInput, nil
}

// accumulatePositionalGradients accumulates gradients for learned positional encodings.
// Since positional encodings are added to each position, we sum gradients across all sequences.
func (e *GPTEncoding) accumulatePositionalGradients(gradOutput, gradPosEncodings tensorTypes.Tensor) error {
	inputShape := e.Base.Input().Shape()

	if len(inputShape) == 2 {
		// Single sequence: [seqLen, embedDim]
		seqLen := inputShape[0]

		for pos := 0; pos < seqLen; pos++ {
			// Add gradient from this position: gradPosEncodings[pos, :] += gradOutput[pos, :]
			posGradSlice := gradPosEncodings.Slice(nil, 0, pos, pos+1) // [1, embedDim]
			outputGradSlice := gradOutput.Slice(nil, 0, pos, pos+1)    // [1, embedDim]
			posGradSlice.Add(posGradSlice, outputGradSlice)
		}
	} else if len(inputShape) == 3 {
		// Batch of sequences: [batchSize, seqLen, embedDim]
		batchSize := inputShape[0]
		seqLen := inputShape[1]

		for b := 0; b < batchSize; b++ {
			for pos := 0; pos < seqLen; pos++ {
				// Add gradient from this position: gradPosEncodings[pos, :] += gradOutput[b, pos, :]
				posGradSlice := gradPosEncodings.Slice(nil, 0, pos, pos+1)                     // [1, embedDim]
				outputGradSlice := gradOutput.Slice(nil, 1, pos, pos+1).Slice(nil, 0, b, b+1)  // [1, 1, embedDim]
				outputGradSlice = outputGradSlice.Reshape(nil, tensor.NewShape(1, e.embedDim)) // [1, embedDim]
				posGradSlice.Add(posGradSlice, outputGradSlice)
			}
		}
	}

	return nil
}

// OutputShape returns the output shape for given input shape.
// Output shape matches input shape exactly.
func (e *GPTEncoding) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if e == nil {
		return nil, fmt.Errorf("GPTEncoding.OutputShape: nil layer")
	}

	if len(inputShape) != 2 && len(inputShape) != 3 {
		return nil, fmt.Errorf("GPTEncoding.OutputShape: input must be 2D or 3D, got %dD", len(inputShape))
	}

	// Output shape matches input shape
	if len(inputShape) == 2 {
		if inputShape[1] != e.embedDim {
			return nil, fmt.Errorf("GPTEncoding.OutputShape: input embedDim %d doesn't match layer embedDim %d", inputShape[1], e.embedDim)
		}
	} else {
		if inputShape[2] != e.embedDim {
			return nil, fmt.Errorf("GPTEncoding.OutputShape: input embedDim %d doesn't match layer embedDim %d", inputShape[2], e.embedDim)
		}
	}

	return inputShape.Clone(), nil
}

// PositionalEncodings returns the positional encodings tensor.
func (e *GPTEncoding) PositionalEncodings() tensorTypes.Tensor {
	if e == nil {
		return tensor.Empty(tensor.DTFP32)
	}
	weightParam := e.Base.Weights()
	if tensor.IsNil(weightParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return weightParam.Data
}

// SetPositionalEncodings sets the positional encodings tensor.
// Only works for learned positional encodings.
func (e *GPTEncoding) SetPositionalEncodings(encodings tensorTypes.Tensor) error {
	if e == nil {
		return fmt.Errorf("GPTEncoding.SetPositionalEncodings: nil layer")
	}
	if !e.learned {
		return fmt.Errorf("GPTEncoding.SetPositionalEncodings: cannot set encodings for sinusoidal (non-learned) layer")
	}
	if tensor.IsNil(encodings) {
		return fmt.Errorf("GPTEncoding.SetPositionalEncodings: empty encodings tensor")
	}

	// Validate shape
	encShape := encodings.Shape()
	if encShape.Rank() != 2 || encShape[0] != e.maxSeqLen || encShape[1] != e.embedDim {
		return fmt.Errorf("GPTEncoding.SetPositionalEncodings: encodings shape %v doesn't match expected [%d, %d]",
			encShape, e.maxSeqLen, e.embedDim)
	}

	weightParam, ok := e.Base.Parameter(types.ParamWeights)
	if !ok || tensor.IsNil(weightParam.Data) {
		return fmt.Errorf("GPTEncoding.SetPositionalEncodings: encodings parameter not initialized")
	}

	weightParam.Data = encodings
	weightParam.RequiresGrad = e.Base.CanLearn()
	e.Base.SetParam(types.ParamWeights, weightParam)

	return nil
}
