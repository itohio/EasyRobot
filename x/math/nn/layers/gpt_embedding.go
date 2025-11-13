package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// GPTEmbedding represents a token embedding layer for GPT models.
// Maps token IDs to embedding vectors for robot navigation tasks.
type GPTEmbedding struct {
	Base
	vocabSize int // Number of tokens in vocabulary
	embedDim  int // Dimension of embedding vectors
}

// NewGPTEmbedding creates a new GPT embedding layer.
// vocabSize: number of unique tokens (sensor readings, actions, goals)
// embedDim: dimension of each embedding vector
func NewGPTEmbedding(vocabSize, embedDim int, opts ...Option) (*GPTEmbedding, error) {
	if vocabSize <= 0 {
		return nil, fmt.Errorf("GPTEmbedding: vocabSize must be positive, got %d", vocabSize)
	}
	if embedDim <= 0 {
		return nil, fmt.Errorf("GPTEmbedding: embedDim must be positive, got %d", embedDim)
	}

	// Create Base without options first
	base := NewBase("gpt_embedding")

	embedding := &GPTEmbedding{
		Base:      base,
		vocabSize: vocabSize,
		embedDim:  embedDim,
	}

	// Parse options first to get data type and any pre-set weights
	embedding.Base.ParseOptions(opts...)

	// Initialize embedding table parameter
	// Shape: [vocabSize, embedDim] - each row is an embedding vector
	dtype := embedding.Base.DataType()
	embedding.Base.initParam(types.ParamWeights)
	weightParam, _ := embedding.Base.Parameter(types.ParamWeights)
	weightParam.Init(dtype, tensor.NewShape(vocabSize, embedDim), types.ParamWeights, vocabSize, embedDim, embedding.Base.rng, embedding.Base.CanLearn())
	embedding.Base.SetParam(types.ParamWeights, weightParam)

	return embedding, nil
}

// Init initializes the layer, creating internal computation tensors.
// Input shape should be [batchSize, seqLen] for token IDs.
func (e *GPTEmbedding) Init(inputShape tensor.Shape) error {
	if e == nil {
		return fmt.Errorf("GPTEmbedding.Init: nil layer")
	}

	// Validate input shape: [batchSize, seqLen] or [seqLen]
	if len(inputShape) < 1 || len(inputShape) > 2 {
		return fmt.Errorf("GPTEmbedding.Init: input must be 1D or 2D, got %dD", len(inputShape))
	}

	// Compute output shape
	var outputShape tensor.Shape
	var seqLen int
	if len(inputShape) == 1 {
		// Single sequence: [seqLen] -> [seqLen, embedDim]
		seqLen = inputShape[0]
		outputShape = tensor.NewShape(seqLen, e.embedDim)
	} else {
		// Batch of sequences: [batchSize, seqLen] -> [batchSize, seqLen, embedDim]
		batchSize := inputShape[0]
		seqLen = inputShape[1]
		outputShape = tensor.NewShape(batchSize, seqLen, e.embedDim)
	}

	// Allocate output tensor
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	e.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass: looks up embeddings for token IDs.
// Input: token IDs with shape [batchSize, seqLen] or [seqLen]
// Output: embeddings with shape [batchSize, seqLen, embedDim] or [seqLen, embedDim]
func (e *GPTEmbedding) Forward(input tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if e == nil {
		return nil, fmt.Errorf("GPTEmbedding.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GPTEmbedding.Forward: empty input")
	}

	// Store input
	e.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := e.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("GPTEmbedding.Forward: output not allocated, must call Init first")
	}

	// Get embedding table
	weightParam, ok := e.Base.Parameter(types.ParamWeights)
	if !ok || tensor.IsNil(weightParam.Data) {
		return nil, fmt.Errorf("GPTEmbedding.Forward: embedding table not initialized")
	}

	// Perform embedding lookup
	if err := e.lookupEmbeddings(input, weightParam.Data, output); err != nil {
		return nil, fmt.Errorf("GPTEmbedding.Forward: embedding lookup failed: %w", err)
	}

	// Store output
	e.Base.StoreOutput(output)
	return output, nil
}

// lookupEmbeddings performs the embedding lookup operation.
// Uses advanced indexing to gather embedding vectors for each token ID.
func (e *GPTEmbedding) lookupEmbeddings(input, embeddings, output tensorTypes.Tensor) error {
	inputShape := input.Shape()

	if len(inputShape) == 1 {
		// Single sequence: [seqLen] -> [seqLen, embedDim]
		seqLen := inputShape[0]

		// For each position in sequence, copy the corresponding embedding
		for i := 0; i < seqLen; i++ {
			tokenID := int(input.At(i))
			if tokenID < 0 || tokenID >= e.vocabSize {
				return fmt.Errorf("GPTEmbedding.lookupEmbeddings: token ID %d out of range [0, %d)", tokenID, e.vocabSize)
			}

			// Copy embedding vector: embeddings[tokenID, :] -> output[i, :]
			// Use tensor operations to copy the row
			embedSlice := embeddings.Slice(nil, 0, tokenID, tokenID+1) // [1, embedDim]
			outputSlice := output.Slice(nil, 0, i, i+1)                // [1, embedDim]
			embedSlice.Copy(outputSlice)
		}
	} else if len(inputShape) == 2 {
		// Batch of sequences: [batchSize, seqLen] -> [batchSize, seqLen, embedDim]
		batchSize := inputShape[0]
		seqLen := inputShape[1]

		// For each batch and position, copy the corresponding embedding
		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.At(b, i))
				if tokenID < 0 || tokenID >= e.vocabSize {
					return fmt.Errorf("GPTEmbedding.lookupEmbeddings: token ID %d out of range [0, %d)", tokenID, e.vocabSize)
				}

				// Copy embedding vector: embeddings[tokenID, :] -> output[b, i, :]
				// Use element-wise copy for reliability with batched 3D tensors
				for d := 0; d < e.embedDim; d++ {
					embedVal := embeddings.At(tokenID, d)
					output.SetAt(embedVal, b, i, d)
				}
			}
		}
	}

	return nil
}

// Backward computes gradients for the embedding table.
// For embedding layers, gradients are accumulated in the embedding table
// based on which tokens were used in the forward pass.
func (e *GPTEmbedding) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if e == nil {
		return nil, fmt.Errorf("GPTEmbedding.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("GPTEmbedding.Backward: empty gradOutput")
	}

	input := e.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GPTEmbedding.Backward: input not stored, must call Forward first")
	}

	// Embedding layer has no input gradient (it's the first layer)
	// Only compute gradients for the embedding table if learning is enabled
	if e.Base.CanLearn() {
		weightParam, ok := e.Base.Parameter(types.ParamWeights)
		if !ok || tensor.IsNil(weightParam.Data) {
			return nil, fmt.Errorf("GPTEmbedding.Backward: embedding table not initialized")
		}

		if weightParam.RequiresGrad {
			if tensor.IsNil(weightParam.Grad) {
				weightParam.Grad = tensor.New(weightParam.Data.DataType(), weightParam.Data.Shape())
			}

			// Accumulate gradients in embedding table
			if err := e.accumulateEmbeddingGradients(input, gradOutput, weightParam.Grad); err != nil {
				return nil, fmt.Errorf("GPTEmbedding.Backward: gradient accumulation failed: %w", err)
			}

			e.Base.SetParam(types.ParamWeights, weightParam)
		}
	}

	// No gradient to propagate backward (first layer)
	return nil, nil
}

// accumulateEmbeddingGradients adds gradients to the embedding table.
// For each token used in the forward pass, adds the corresponding gradient slice.
func (e *GPTEmbedding) accumulateEmbeddingGradients(input, gradOutput, gradEmbeddings tensorTypes.Tensor) error {
	inputShape := input.Shape()

	if len(inputShape) == 1 {
		// Single sequence: [seqLen]
		seqLen := inputShape[0]

		for i := 0; i < seqLen; i++ {
			tokenID := int(input.At(i))

			// Add gradient: gradEmbeddings[tokenID, :] += gradOutput[i, :]
			embedGradSlice := gradEmbeddings.Slice(nil, 0, tokenID, tokenID+1) // [1, embedDim]
			outputGradSlice := gradOutput.Slice(nil, 0, i, i+1)                // [1, embedDim]
			embedGradSlice.Add(embedGradSlice, outputGradSlice)
		}
	} else if len(inputShape) == 2 {
		// Batch of sequences: [batchSize, seqLen]
		batchSize := inputShape[0]
		seqLen := inputShape[1]

		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.At(b, i))

				// Add gradient: gradEmbeddings[tokenID, :] += gradOutput[b, i, :]
				// Use element-wise addition for reliability
				for d := 0; d < e.embedDim; d++ {
					gradVal := gradOutput.At(b, i, d)
					currentVal := gradEmbeddings.At(tokenID, d)
					gradEmbeddings.SetAt(currentVal+gradVal, tokenID, d)
				}
			}
		}
	}

	return nil
}

// OutputShape returns the output shape for given input shape.
// Input: [batchSize, seqLen] or [seqLen]
// Output: [batchSize, seqLen, embedDim] or [seqLen, embedDim]
func (e *GPTEmbedding) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if e == nil {
		return nil, fmt.Errorf("GPTEmbedding.OutputShape: nil layer")
	}

	if len(inputShape) < 1 || len(inputShape) > 2 {
		return nil, fmt.Errorf("GPTEmbedding.OutputShape: input must be 1D or 2D, got %dD", len(inputShape))
	}

	if len(inputShape) == 1 {
		return tensor.NewShape(inputShape[0], e.embedDim), nil
	} else {
		return tensor.NewShape(inputShape[0], inputShape[1], e.embedDim), nil
	}
}

// EmbeddingTable returns the embedding table tensor.
func (e *GPTEmbedding) EmbeddingTable() tensorTypes.Tensor {
	if e == nil {
		return tensor.Empty(tensor.DTFP32)
	}
	weightParam := e.Base.Weights()
	if tensor.IsNil(weightParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return weightParam.Data
}

// SetEmbeddingTable sets the embedding table tensor.
func (e *GPTEmbedding) SetEmbeddingTable(embeddings tensorTypes.Tensor) error {
	if e == nil {
		return fmt.Errorf("GPTEmbedding.SetEmbeddingTable: nil layer")
	}
	if tensor.IsNil(embeddings) {
		return fmt.Errorf("GPTEmbedding.SetEmbeddingTable: empty embeddings tensor")
	}

	// Validate shape
	embedShape := embeddings.Shape()
	if embedShape.Rank() != 2 || embedShape[0] != e.vocabSize || embedShape[1] != e.embedDim {
		return fmt.Errorf("GPTEmbedding.SetEmbeddingTable: embeddings shape %v doesn't match expected [%d, %d]",
			embedShape, e.vocabSize, e.embedDim)
	}

	weightParam, ok := e.Base.Parameter(types.ParamWeights)
	if !ok || tensor.IsNil(weightParam.Data) {
		return fmt.Errorf("GPTEmbedding.SetEmbeddingTable: embedding parameter not initialized")
	}

	weightParam.Data = embeddings
	weightParam.RequiresGrad = e.Base.CanLearn()
	e.Base.SetParam(types.ParamWeights, weightParam)

	return nil
}
