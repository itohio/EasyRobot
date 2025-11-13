package layers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/math/tensor"
)

// TestGPTEmbedding_EmbeddingLearningNote contains information about embedding learning tests.
// The full embedding learning test is moved to the learn package to avoid circular imports.
// See: learn/gpt_embedding_test.go
func TestGPTEmbedding_EmbeddingLearningNote(t *testing.T) {
	t.Log("Note: Full embedding learning test moved to learn/gpt_embedding_test.go")
	t.Log("This avoids circular import issues between nn and learn packages")
}

// TestGPTEmbedding_BasicFunctionality tests basic embedding lookup functionality.
func TestGPTEmbedding_BasicFunctionality(t *testing.T) {
	vocabSize := 10
	embedDim := 4

	embedding, err := NewGPTEmbedding(vocabSize, embedDim, WithCanLearn(false))
	require.NoError(t, err)

	// Test initialization
	inputShape := tensor.NewShape(2, 3) // batchSize=2, seqLen=3
	err = embedding.Init(inputShape)
	require.NoError(t, err)

	// Test output shape
	outputShape, err := embedding.OutputShape(inputShape)
	require.NoError(t, err)
	expectedShape := tensor.NewShape(2, 3, 4) // batchSize, seqLen, embedDim
	assert.Equal(t, expectedShape, outputShape)

	// Test forward pass
	inputData := []float32{0, 1, 2, 3, 4, 5} // 2 sequences of 3 tokens each
	input := tensor.FromFloat32(inputShape, inputData)

	output, err := embedding.Forward(input)
	require.NoError(t, err)
	assert.Equal(t, expectedShape, output.Shape())

	// Verify embedding lookup: output[i,j,:] should equal embedding_table[input[i,j], :]
	embedTable := embedding.EmbeddingTable()
	for b := 0; b < 2; b++ {
		for s := 0; s < 3; s++ {
			tokenID := int(input.At(b, s))
			for d := 0; d < embedDim; d++ {
				expected := embedTable.At(tokenID, d)
				actual := output.At(b, s, d)
				assert.Equal(t, expected, actual,
					"Embedding lookup failed at batch=%d, seq=%d, dim=%d", b, s, d)
			}
		}
	}
}

// TestGPTEmbedding_ErrorCases tests error handling in GPTEmbedding.
func TestGPTEmbedding_ErrorCases(t *testing.T) {
	// Test invalid vocabSize
	_, err := NewGPTEmbedding(0, 64)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "vocabSize must be positive")

	// Test invalid embedDim
	_, err = NewGPTEmbedding(128, 0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "embedDim must be positive")

	// Test valid creation
	embedding, err := NewGPTEmbedding(128, 64)
	require.NoError(t, err)
	require.NotNil(t, embedding)

	// Test invalid input shape for Init
	err = embedding.Init(tensor.NewShape(2, 3, 4)) // 3D input not allowed
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input must be 2D")

	// Test out of bounds token ID
	inputShape := tensor.NewShape(1, 1)
	err = embedding.Init(inputShape)
	require.NoError(t, err)

	input := tensor.FromFloat32(inputShape, []float32{128}) // Token ID 128 >= vocabSize 128
	_, err = embedding.Forward(input)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "out of range")
}
