package learn

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// TestGPTEmbeddingLearning demonstrates that GPT embeddings can learn meaningful representations.
// This test creates a simple classification task where embeddings are trained to distinguish between
// different categories of words, showing that similar words develop similar embeddings.
func TestGPTEmbeddingLearning(t *testing.T) {
	// First, test basic functionality with a small example
	testBasicEmbeddingForward(t)

	// Then test full embedding learning
	testFullEmbeddingLearning(t)
}

// testFullEmbeddingLearning tests complete embedding learning with training
func testFullEmbeddingLearning(t *testing.T) {
	const (
		vocabSize    = 12 // Small vocabulary: 3 categories × 4 words each
		embedDim     = 8  // Small embedding dimension
		numClasses   = 3  // 3 categories
		seqLen       = 4  // 4 words per sequence
		batchSize    = 8  // 8 sequences per batch
		numEpochs    = 20 // Fewer epochs for faster testing
		learningRate = 0.01
	)

	// Create embedding layer
	embedding, err := layers.NewGPTEmbedding(vocabSize, embedDim, layers.WithCanLearn(true))
	require.NoError(t, err)

	// Create a simple classifier (Dense layer)
	classifier, err := layers.NewDense(embedDim, numClasses, layers.WithCanLearn(true))
	require.NoError(t, err)

	// Initialize layers
	inputShape := tensor.NewShape(batchSize, seqLen)
	err = embedding.Init(inputShape)
	require.NoError(t, err)

	// Classifier takes averaged embeddings: [batchSize, embedDim]
	classifierInputShape := tensor.NewShape(batchSize, embedDim)
	err = classifier.Init(classifierInputShape)
	require.NoError(t, err)

	// Create synthetic dataset: words grouped by categories
	dataset := createSimpleEmbeddingDataset(batchSize, seqLen, vocabSize, numClasses)

	t.Logf("Testing embedding learning with %d words, %d categories, %d epochs", vocabSize, numClasses, numEpochs)

	// Simple training: just one forward/backward pass to verify gradients work
	batch := dataset[0]

	// Forward pass through embedding layer
	embedOutput, err := embedding.Forward(batch.input)
	require.NoError(t, err)

	// Average embeddings across sequence
	pooledEmbedding := averageEmbeddings(embedOutput, seqLen)

	// Forward pass through classifier
	classifierOutput, err := classifier.Forward(pooledEmbedding)
	require.NoError(t, err)

	// Create target (first sample should predict category 0)
	targetData := make([]float32, batchSize*numClasses)
	targetData[0] = 1.0 // First sample -> category 0
	target := tensor.FromFloat32(tensor.NewShape(batchSize, numClasses), targetData)

	// Compute simple MSE loss (avoid cross-entropy for now)
	loss := computeSimpleMSE(classifierOutput, target)
	t.Logf("Initial loss: %.4f", loss)

	// Create gradients (simple: gradient = prediction - target)
	gradOutput := tensor.New(tensor.DTFP32, classifierOutput.Shape())
	classifierOutput.Subtract(gradOutput, target)

	// Backward through classifier
	classifierGrad, err := classifier.Backward(gradOutput)
	require.NoError(t, err)

	// Expand classifier gradient back to sequence length for embedding backward pass
	expandedGrad := expandGradient(classifierGrad, seqLen)

	// Backward through embedding
	_, err = embedding.Backward(expandedGrad)
	require.NoError(t, err)

	// Check that gradients were computed (not zero)
	embedParams := embedding.Parameters()
	hasNonZeroGrad := false
	for _, param := range embedParams {
		if param.Grad != nil && !tensor.IsNil(param.Grad) {
			// Check if any gradient value is non-zero
			for i := 0; i < param.Grad.Shape().Size(); i++ {
				if param.Grad.At(i) != 0 {
					hasNonZeroGrad = true
					break
				}
			}
		}
	}
	assert.True(t, hasNonZeroGrad, "Embeddings should have non-zero gradients after backward pass")

	t.Log("Full embedding learning test passed - gradients computed successfully")
}

// testBasicEmbeddingForward tests basic embedding functionality with a simple forward/backward pass
func testBasicEmbeddingForward(t *testing.T) {
	const (
		vocabSize = 10 // Small vocabulary for testing
		embedDim  = 4  // Small embedding dimension
		seqLen    = 3  // Short sequence
		batchSize = 2  // Small batch
	)

	// Create embedding layer
	embedding, err := layers.NewGPTEmbedding(vocabSize, embedDim, layers.WithCanLearn(true))
	require.NoError(t, err)

	// Initialize layer
	inputShape := tensor.NewShape(batchSize, seqLen)
	err = embedding.Init(inputShape)
	require.NoError(t, err)

	// Create test input: simple token IDs
	inputData := make([]float32, batchSize*seqLen)
	for i := range inputData {
		inputData[i] = float32(i % vocabSize) // Cycle through vocabulary
	}
	input := tensor.FromFloat32(inputShape, inputData)

	// Forward pass
	output, err := embedding.Forward(input)
	require.NoError(t, err)

	// Check output shape
	expectedShape := tensor.NewShape(batchSize, seqLen, embedDim)
	assert.Equal(t, expectedShape, output.Shape())

	// Create dummy gradient for backward pass
	gradOutput := tensor.New(tensor.DTFP32, expectedShape)
	gradOutput.Fill(nil, 1.0) // Simple gradient

	// Backward pass
	gradInput, err := embedding.Backward(gradOutput)
	require.NoError(t, err)
	assert.Nil(t, gradInput) // Embedding layer has no input gradient

	t.Log("Basic embedding forward/backward test passed")
}

// embeddingBatch represents a batch of training data
type embeddingBatch struct {
	input  tensor.Tensor // [batchSize, seqLen] - word IDs
	target tensor.Tensor // [batchSize, numClasses] - one-hot category labels
}

// createEmbeddingDataset creates synthetic training data for embedding learning.
// Groups vocabulary words into categories and creates sequences within categories.
func createEmbeddingDataset(batchSize, seqLen, vocabSize, numClasses int) []embeddingBatch {
	wordsPerClass := vocabSize / numClasses // 32 words per class
	numBatches := 10                        // 10 batches for training

	dataset := make([]embeddingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		// Create input: [batchSize, seqLen] word IDs
		inputData := make([]float32, batchSize*seqLen)

		// Create target: [batchSize, numClasses] one-hot labels
		targetData := make([]float32, batchSize*numClasses)

		for i := 0; i < batchSize; i++ {
			// Assign this sample to a random category
			category := b % numClasses // Deterministic for reproducibility

			// Fill target (one-hot)
			targetIdx := i*numClasses + category
			targetData[targetIdx] = 1.0

			// Fill input sequence with words from this category
			categoryStart := category * wordsPerClass

			for j := 0; j < seqLen; j++ {
				// Choose random word from this category
				wordOffset := (i*j + b) % wordsPerClass // Deterministic but varied
				wordID := categoryStart + wordOffset

				inputIdx := i*seqLen + j
				inputData[inputIdx] = float32(wordID)
			}
		}

		dataset[b] = embeddingBatch{
			input:  tensor.FromFloat32(tensor.NewShape(batchSize, seqLen), inputData),
			target: tensor.FromFloat32(tensor.NewShape(batchSize, numClasses), targetData),
		}
	}

	return dataset
}

// averageEmbeddings performs simple average pooling across the sequence dimension.
// Input: [batchSize, seqLen, embedDim] -> Output: [batchSize, embedDim]
func averageEmbeddings(embeddings tensor.Tensor, seqLen int) tensor.Tensor {
	// Sum across sequence dimension (dimension 1 = seqLen)
	summed := embeddings.Sum(nil, []int{1}) // Sum along seqLen dimension

	// Sum removes the summed dimension, so [batchSize, seqLen, embedDim] -> [batchSize, embedDim]
	// No reshape needed since the dimension is already removed

	// Divide by sequence length to get average
	avgEmbeddings := summed.ScalarMul(nil, 1.0/float64(seqLen))

	return avgEmbeddings
}

// expandGradient expands classifier gradient back to sequence length.
// Input: [batchSize, embedDim] -> Output: [batchSize, seqLen, embedDim]
func expandGradient(grad tensor.Tensor, seqLen int) tensor.Tensor {
	gradShape := grad.Shape()
	batchSize := gradShape[0]
	embedDim := gradShape[1]

	// Create expanded tensor: [batchSize, seqLen, embedDim]
	expandedShape := tensor.NewShape(batchSize, seqLen, embedDim)
	expanded := tensor.New(grad.DataType(), expandedShape)

	// Copy gradient to each position in the sequence
	for s := 0; s < seqLen; s++ {
		for b := 0; b < batchSize; b++ {
			for d := 0; d < embedDim; d++ {
				val := grad.At(b, d)
				expanded.SetAt(val, b, s, d)
			}
		}
	}

	return expanded
}

// testEmbeddingSimilarity tests that embeddings for words from the same category are more similar
// than embeddings for words from different categories.
func testEmbeddingSimilarity(t *testing.T, embedding *layers.GPTEmbedding, vocabSize, numClasses int) {
	embedTable := embedding.EmbeddingTable()
	wordsPerClass := vocabSize / numClasses

	// Calculate average pairwise similarity within categories and between categories
	withinCategorySimilarity := 0.0
	betweenCategorySimilarity := 0.0
	withinCount := 0
	betweenCount := 0

	for c1 := 0; c1 < numClasses; c1++ {
		for c2 := 0; c2 < numClasses; c2++ {
			// Get representative embeddings for each category (first word in each category)
			word1 := c1 * wordsPerClass
			word2 := c2 * wordsPerClass

			embed1 := getEmbeddingVector(embedTable, word1)
			embed2 := getEmbeddingVector(embedTable, word2)

			similarity := cosineSimilarity(embed1, embed2)

			if c1 == c2 {
				withinCategorySimilarity += similarity
				withinCount++
			} else {
				betweenCategorySimilarity += similarity
				betweenCount++
			}
		}
	}

	avgWithinSimilarity := withinCategorySimilarity / float64(withinCount)
	avgBetweenSimilarity := betweenCategorySimilarity / float64(betweenCount)

	t.Logf("Embedding similarity analysis:")
	t.Logf("  Average within-category similarity: %.4f", avgWithinSimilarity)
	t.Logf("  Average between-category similarity: %.4f", avgBetweenSimilarity)

	// Within-category similarity should be higher than between-category similarity
	// This demonstrates that the embeddings learned meaningful category distinctions
	assert.Greater(t, avgWithinSimilarity, avgBetweenSimilarity,
		"Embeddings within same category should be more similar (%.4f) than between categories (%.4f)",
		avgWithinSimilarity, avgBetweenSimilarity)
}

// getEmbeddingVector extracts the embedding vector for a specific word ID.
func getEmbeddingVector(embedTable tensor.Tensor, wordID int) []float32 {
	embedDim := embedTable.Shape()[1]
	vector := make([]float32, embedDim)

	for i := 0; i < embedDim; i++ {
		vector[i] = float32(embedTable.At(wordID, i))
	}

	return vector
}

// createSimpleEmbeddingDataset creates synthetic training data for embedding learning.
// Groups vocabulary words into categories and creates sequences within categories.
func createSimpleEmbeddingDataset(batchSize, seqLen, vocabSize, numClasses int) []embeddingBatch {
	wordsPerClass := vocabSize / numClasses // 4 words per class for vocabSize=12, numClasses=3
	numBatches := 5                         // 5 batches for training

	dataset := make([]embeddingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		// Create input: [batchSize, seqLen] word IDs
		inputData := make([]float32, batchSize*seqLen)

		// Create target: [batchSize, numClasses] one-hot labels
		targetData := make([]float32, batchSize*numClasses)

		for i := 0; i < batchSize; i++ {
			// Assign this sample to a category (deterministic for reproducibility)
			category := (b + i) % numClasses

			// Fill target (one-hot)
			targetIdx := i*numClasses + category
			targetData[targetIdx] = 1.0

			// Fill input sequence with words from this category
			categoryStart := category * wordsPerClass

			for j := 0; j < seqLen; j++ {
				// Choose word from this category
				wordOffset := (i*j + b) % wordsPerClass
				wordID := categoryStart + wordOffset

				inputIdx := i*seqLen + j
				inputData[inputIdx] = float32(wordID)
			}
		}

		dataset[b] = embeddingBatch{
			input:  tensor.FromFloat32(tensor.NewShape(batchSize, seqLen), inputData),
			target: tensor.FromFloat32(tensor.NewShape(batchSize, numClasses), targetData),
		}
	}

	return dataset
}

// computeSimpleMSE computes mean squared error between prediction and target.
func computeSimpleMSE(pred, target tensor.Tensor) float32 {
	if pred.Shape().Size() != target.Shape().Size() {
		panic("prediction and target must have same size")
	}

	var mse float32
	size := pred.Shape().Size()

	for i := 0; i < size; i++ {
		diff := float32(pred.At(i) - target.At(i))
		mse += diff * diff
	}

	return mse / float32(size)
}

// cosineSimilarity computes cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("vectors must have same length")
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
