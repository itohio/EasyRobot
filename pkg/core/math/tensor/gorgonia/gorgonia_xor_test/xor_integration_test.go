package gorgonia_xor_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/gorgonia"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// TestXORTrainEagerInferGraph demonstrates the complete workflow:
// 1. Train an XOR network using eager_tensor (immediate execution)
// 2. Extract learned weights
// 3. Build a Gorgonia computation graph with the same architecture
// 4. Load the trained weights into the graph
// 5. Run inference with the compiled graph
// 6. Verify results match expected XOR behavior
func TestXORTrainEagerInferGraph(t *testing.T) {
	t.Log("=== Phase 1: Train XOR network with eager_tensor ===")

	// Train the network
	weights := trainXORWithEager(t)

	t.Log("\n=== Phase 2: Build Gorgonia graph with learned weights ===")

	// Build inference graph
	eg := gorgonia.NewExpressionGraph()
	inferGraph := buildInferenceGraph(t, eg, weights)

	// Compile once
	if err := eg.Compile(); err != nil {
		t.Fatalf("Failed to compile graph: %v", err)
	}

	t.Log("\n=== Phase 3: Run inference on XOR test cases ===")

	// Test all XOR cases
	xorTests := []struct {
		input    []float32
		expected float32
		name     string
	}{
		{[]float32{0, 0}, 0, "0 XOR 0 = 0"},
		{[]float32{0, 1}, 1, "0 XOR 1 = 1"},
		{[]float32{1, 0}, 1, "1 XOR 0 = 1"},
		{[]float32{1, 1}, 0, "1 XOR 1 = 0"},
	}

	for _, tc := range xorTests {
		// Set input
		inferGraph.input.Copy(&simpleTensor{
			data:  tc.input,
			shape: types.Shape{1, 2},
			dtype: types.FP32,
		})

		// Execute graph
		if err := eg.Compute(); err != nil {
			t.Fatalf("Failed to compute for %s: %v", tc.name, err)
		}

		// Get output
		output := inferGraph.output.Data().([]float32)
		prediction := output[0]

		// Check result (with tolerance for floating point)
		diff := math.Abs(float64(prediction - tc.expected))
		if diff > 0.2 { // Allow 20% error (generous for XOR)
			t.Errorf("%s: expected ~%.1f, got %.3f (diff: %.3f)",
				tc.name, tc.expected, prediction, diff)
		} else {
			t.Logf("✓ %s: prediction=%.3f (expected=%.1f)",
				tc.name, prediction, tc.expected)
		}
	}

	t.Log("\n=== Test Complete: Successfully trained with eager, inferred with graph ===")
}

// XORWeights holds the trained network weights
type XORWeights struct {
	W1 []float32 // [2, 4] - input to hidden
	B1 []float32 // [4] - hidden bias
	W2 []float32 // [4, 1] - hidden to output
	B2 []float32 // [1] - output bias
}

// trainXORWithEager trains a simple XOR network using eager execution
func trainXORWithEager(t *testing.T) *XORWeights {
	// XOR training data
	X := [][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	Y := []float32{0, 1, 1, 0}

	// Initialize network: 2 -> 4 -> 1
	W1 := eager_tensor.New(types.FP32, types.NewShape(2, 4))
	B1 := eager_tensor.New(types.FP32, types.NewShape(4))
	W2 := eager_tensor.New(types.FP32, types.NewShape(4, 1))
	B2 := eager_tensor.New(types.FP32, types.NewShape(1))

	// Xavier initialization
	initXavier(W1, 2, 4)
	initXavier(W2, 4, 1)
	B1.Fill(nil, 0.0)
	B2.Fill(nil, 0.0)

	// Training parameters
	learningRate := 0.5
	epochs := 5000

	t.Logf("Training for %d epochs with learning rate %.2f...", epochs, learningRate)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i := 0; i < len(X); i++ {
			// Forward pass
			input := eager_tensor.FromArray(types.NewShape(1, 2), X[i])
			target := Y[i]

			// Layer 1: input * W1 + B1
			hidden := input.MatMul(nil, W1)
			// Add bias element-wise (reshape B1 from [4] to [1,4] for broadcasting)
			for j := 0; j < 4; j++ {
				val := hidden.At(0, j) + B1.At(j)
				hidden.SetAt(val, 0, j)
			}
			hidden = hidden.Sigmoid(nil) // Activation

			// Layer 2: hidden * W2 + B2
			output := hidden.MatMul(nil, W2)
			// Add bias
			val := output.At(0, 0) + B2.At(0)
			output.SetAt(val, 0, 0)
			output = output.Sigmoid(nil) // Activation

			// Compute loss (MSE)
			pred := output.At(0, 0)
			loss := (pred - float64(target)) * (pred - float64(target))
			totalLoss += loss

			// Backward pass (simple gradient descent)
			// Output gradient
			outputGrad := 2.0 * (pred - float64(target)) * pred * (1 - pred) // MSE * sigmoid'

			// Gradients for W2 and B2
			for j := 0; j < 4; j++ {
				hiddenVal := hidden.At(0, j)
				grad := outputGrad * hiddenVal

				// Update W2[j, 0]
				w2Val := W2.At(j, 0)
				W2.SetAt(w2Val-learningRate*grad, j, 0)
			}
			// Update B2
			b2Val := B2.At(0)
			B2.SetAt(b2Val-learningRate*outputGrad, 0)

			// Hidden gradient
			for j := 0; j < 4; j++ {
				hiddenVal := hidden.At(0, j)
				w2Val := W2.At(j, 0)
				hiddenGrad := outputGrad * w2Val * hiddenVal * (1 - hiddenVal)

				// Gradients for W1 and B1
				for k := 0; k < 2; k++ {
					inputVal := input.At(0, k)
					grad := hiddenGrad * inputVal

					// Update W1[k, j]
					w1Val := W1.At(k, j)
					W1.SetAt(w1Val-learningRate*grad, k, j)
				}

				// Update B1[j]
				b1Val := B1.At(j)
				B1.SetAt(b1Val-learningRate*hiddenGrad, j)
			}
		}

		// Log progress
		if (epoch+1)%1000 == 0 || epoch == 0 {
			avgLoss := totalLoss / float64(len(X))
			t.Logf("Epoch %d: avg loss = %.6f", epoch+1, avgLoss)
		}
	}

	// Test trained network
	t.Log("\nTrained network predictions (eager):")
	for i := 0; i < len(X); i++ {
		input := eager_tensor.FromArray(types.NewShape(1, 2), X[i])
		hidden := input.MatMul(nil, W1)
		// Add bias
		for j := 0; j < 4; j++ {
			val := hidden.At(0, j) + B1.At(j)
			hidden.SetAt(val, 0, j)
		}
		hidden = hidden.Sigmoid(nil)
		output := hidden.MatMul(nil, W2)
		val := output.At(0, 0) + B2.At(0)
		output.SetAt(val, 0, 0)
		output = output.Sigmoid(nil)
		pred := output.At(0, 0)
		t.Logf("  Input: %v, Expected: %.0f, Predicted: %.3f", X[i], Y[i], pred)
	}

	// Extract weights
	weights := &XORWeights{
		W1: extractWeights(W1),
		B1: extractWeights(B1),
		W2: extractWeights(W2),
		B2: extractWeights(B2),
	}

	return weights
}

// InferenceGraph holds the graph tensors for inference
type InferenceGraph struct {
	input  *gorgonia.GraphTensor
	output *gorgonia.GraphTensor
}

// buildInferenceGraph constructs a Gorgonia computation graph with trained weights
func buildInferenceGraph(t *testing.T, eg *gorgonia.ExpressionGraph, weights *XORWeights) *InferenceGraph {
	// Create input node (will be filled at inference time)
	input := eg.New(types.FP32, 1, 2).(*gorgonia.GraphTensor)

	// Create weight nodes as constants (loaded from trained weights)
	// Note: Bias needs to be reshaped to [1, N] for proper broadcasting with [1, N] activations
	W1 := eg.NewConstant(weights.W1, 2, 4).(*gorgonia.GraphTensor)
	B1 := eg.NewConstant(weights.B1, 1, 4).(*gorgonia.GraphTensor) // [1, 4] for broadcasting
	W2 := eg.NewConstant(weights.W2, 4, 1).(*gorgonia.GraphTensor)
	B2 := eg.NewConstant(weights.B2, 1, 1).(*gorgonia.GraphTensor) // [1, 1] for broadcasting

	t.Logf("Loaded weights into graph:")
	t.Logf("  W1: %v (shape [2,4])", weights.W1[:8]) // Show first 8 elements
	t.Logf("  B1: %v (shape [1,4] for broadcasting)", weights.B1)
	t.Logf("  W2: %v (shape [4,1])", weights.W2)
	t.Logf("  B2: %v (shape [1,1] for broadcasting)", weights.B2)

	// Build forward pass graph
	// Layer 1: input @ W1 + B1 -> sigmoid
	hidden := input.MatMul(nil, W1).(*gorgonia.GraphTensor)
	hidden = hidden.Add(nil, B1).(*gorgonia.GraphTensor) // [1,4] + [1,4]
	hidden = hidden.Sigmoid(nil).(*gorgonia.GraphTensor)

	// Layer 2: hidden @ W2 + B2 -> sigmoid
	output := hidden.MatMul(nil, W2).(*gorgonia.GraphTensor)
	output = output.Add(nil, B2).(*gorgonia.GraphTensor) // [1,1] + [1,1]
	output = output.Sigmoid(nil).(*gorgonia.GraphTensor)

	t.Logf("Graph built with %d operations", eg.OperationCount())

	return &InferenceGraph{
		input:  input,
		output: output,
	}
}

// Helper functions

func initXavier(t types.Tensor, fanIn, fanOut int) {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	for i := 0; i < t.Size(); i++ {
		// Simple pseudo-random initialization
		val := (float64(i*7919%1000)/1000.0 - 0.5) * 2 * limit
		t.SetAt(val, i)
	}
}

func extractWeights(t types.Tensor) []float32 {
	data := t.Data()
	switch d := data.(type) {
	case []float32:
		result := make([]float32, len(d))
		copy(result, d)
		return result
	case []float64:
		result := make([]float32, len(d))
		for i, v := range d {
			result[i] = float32(v)
		}
		return result
	default:
		panic(fmt.Sprintf("unsupported data type: %T", data))
	}
}

// simpleTensor is a minimal tensor for feeding data to graph
type simpleTensor struct {
	data  any
	shape types.Shape
	dtype types.DataType
}

func (st *simpleTensor) Data() any                                             { return st.data }
func (st *simpleTensor) Shape() types.Shape                                    { return st.shape }
func (st *simpleTensor) DataType() types.DataType                              { return st.dtype }
func (st *simpleTensor) ID() uintptr                                           { return 0 }
func (st *simpleTensor) Rank() int                                             { return len(st.shape) }
func (st *simpleTensor) Size() int                                             { return st.shape.Size() }
func (st *simpleTensor) Empty() bool                                           { return st.Size() == 0 }
func (st *simpleTensor) Strides(dst []int) []int                               { return st.shape.Strides(dst) }
func (st *simpleTensor) IsContiguous() bool                                    { return true }
func (st *simpleTensor) Offset() int                                           { return 0 }
func (st *simpleTensor) DataWithOffset() any                                   { return st.data }
func (st *simpleTensor) At(...int) float64                                     { return 0 }
func (st *simpleTensor) SetAt(float64, ...int)                                 {}
func (st *simpleTensor) Elements(...int) func(func(types.Element) bool)        { return nil }
func (st *simpleTensor) Release()                                              {}
func (st *simpleTensor) Clone() types.Tensor                                   { return st }
func (st *simpleTensor) Copy(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Reshape(types.Tensor, types.Shape) types.Tensor        { return st }
func (st *simpleTensor) Slice(types.Tensor, int, int, int) types.Tensor        { return st }
func (st *simpleTensor) Transpose(types.Tensor, []int) types.Tensor            { return st }
func (st *simpleTensor) Permute(types.Tensor, []int) types.Tensor              { return st }
func (st *simpleTensor) BroadcastTo(types.Tensor, types.Shape) types.Tensor    { return st }
func (st *simpleTensor) Fill(types.Tensor, float64) types.Tensor               { return st }
func (st *simpleTensor) FillFunc(types.Tensor, func() float64) types.Tensor    { return st }
func (st *simpleTensor) Pad(types.Tensor, []int, float64) types.Tensor         { return st }
func (st *simpleTensor) Unpad(types.Tensor, []int) types.Tensor                { return st }
func (st *simpleTensor) Add(types.Tensor, types.Tensor) types.Tensor           { return st }
func (st *simpleTensor) Subtract(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) Multiply(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) Divide(types.Tensor, types.Tensor) types.Tensor        { return st }
func (st *simpleTensor) ScalarMul(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) AddScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) SubScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) MulScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) DivScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) Square(types.Tensor) types.Tensor                      { return st }
func (st *simpleTensor) Sqrt(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Exp(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Log(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Pow(types.Tensor, float64) types.Tensor                { return st }
func (st *simpleTensor) Abs(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Sign(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Cos(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Sin(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Negative(types.Tensor) types.Tensor                    { return st }
func (st *simpleTensor) Equal(types.Tensor, types.Tensor) types.Tensor         { return st }
func (st *simpleTensor) Greater(types.Tensor, types.Tensor) types.Tensor       { return st }
func (st *simpleTensor) Less(types.Tensor, types.Tensor) types.Tensor          { return st }
func (st *simpleTensor) NotEqual(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) GreaterEqual(types.Tensor, types.Tensor) types.Tensor  { return st }
func (st *simpleTensor) LessEqual(types.Tensor, types.Tensor) types.Tensor     { return st }
func (st *simpleTensor) EqualScalar(types.Tensor, float64) types.Tensor        { return st }
func (st *simpleTensor) NotEqualScalar(types.Tensor, float64) types.Tensor     { return st }
func (st *simpleTensor) GreaterScalar(types.Tensor, float64) types.Tensor      { return st }
func (st *simpleTensor) LessScalar(types.Tensor, float64) types.Tensor         { return st }
func (st *simpleTensor) GreaterEqualScalar(types.Tensor, float64) types.Tensor { return st }
func (st *simpleTensor) LessEqualScalar(types.Tensor, float64) types.Tensor    { return st }
func (st *simpleTensor) Where(types.Tensor, types.Tensor, types.Tensor, types.Tensor) types.Tensor {
	return st
}
func (st *simpleTensor) Sum(types.Tensor, []int) types.Tensor           { return st }
func (st *simpleTensor) ReduceSum(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) Mean(types.Tensor, []int) types.Tensor          { return st }
func (st *simpleTensor) ReduceMean(types.Tensor, []int) types.Tensor    { return st }
func (st *simpleTensor) Max(types.Tensor, []int) types.Tensor           { return st }
func (st *simpleTensor) ReduceMax(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) Min(types.Tensor, []int) types.Tensor           { return st }
func (st *simpleTensor) ReduceMin(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) ArgMax(types.Tensor, int) types.Tensor          { return st }
func (st *simpleTensor) ArgMin(types.Tensor, int) types.Tensor          { return st }
func (st *simpleTensor) MatMul(types.Tensor, types.Tensor) types.Tensor { return st }
func (st *simpleTensor) MatMulTransposed(types.Tensor, types.Tensor, bool, bool) types.Tensor {
	return st
}
func (st *simpleTensor) MatVecMulTransposed(types.Tensor, types.Tensor, types.Tensor, float64, float64) types.Tensor {
	return st
}
func (st *simpleTensor) Dot(types.Tensor) float64                                   { return 0 }
func (st *simpleTensor) Tensordot(types.Tensor) float64                             { return 0 }
func (st *simpleTensor) Norm(int) float64                                           { return 0 }
func (st *simpleTensor) L2Normalize(types.Tensor, int) types.Tensor                 { return st }
func (st *simpleTensor) Normalize(types.Tensor, int) types.Tensor                   { return st }
func (st *simpleTensor) AddScaled(types.Tensor, types.Tensor, float64) types.Tensor { return st }
func (st *simpleTensor) ScatterAdd(types.Tensor, types.Tensor, types.Tensor) types.Tensor {
	return st
}
func (st *simpleTensor) BatchNormForward(types.Tensor, types.Tensor, types.Tensor, float64) types.Tensor {
	return st
}
func (st *simpleTensor) BatchNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) LayerNormForward(types.Tensor, types.Tensor, types.Tensor, float64) types.Tensor {
	return st
}
func (st *simpleTensor) LayerNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) RMSNormForward(types.Tensor, types.Tensor, float64) types.Tensor { return st }
func (st *simpleTensor) RMSNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor) {
	return st, st
}
func (st *simpleTensor) InstanceNorm2D(types.Tensor, types.Tensor, types.Tensor, float64) types.Tensor {
	return st
}
func (st *simpleTensor) InstanceNorm2DGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) GroupNormForward(types.Tensor, types.Tensor, types.Tensor, int, float64) types.Tensor {
	return st
}
func (st *simpleTensor) GroupNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, int, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) ReLU(types.Tensor) types.Tensor                           { return st }
func (st *simpleTensor) Sigmoid(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Tanh(types.Tensor) types.Tensor                           { return st }
func (st *simpleTensor) Softmax(int, types.Tensor) types.Tensor                   { return st }
func (st *simpleTensor) ReLU6(types.Tensor) types.Tensor                          { return st }
func (st *simpleTensor) LeakyReLU(types.Tensor, float64) types.Tensor             { return st }
func (st *simpleTensor) ELU(types.Tensor, float64) types.Tensor                   { return st }
func (st *simpleTensor) Softplus(types.Tensor) types.Tensor                       { return st }
func (st *simpleTensor) Swish(types.Tensor) types.Tensor                          { return st }
func (st *simpleTensor) GELU(types.Tensor) types.Tensor                           { return st }
func (st *simpleTensor) ReLUGrad(types.Tensor, types.Tensor) types.Tensor         { return st }
func (st *simpleTensor) SigmoidGrad(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) TanhGrad(types.Tensor, types.Tensor) types.Tensor         { return st }
func (st *simpleTensor) SoftmaxGrad(types.Tensor, types.Tensor, int) types.Tensor { return st }
func (st *simpleTensor) Conv1D(types.Tensor, types.Tensor, types.Tensor, int, int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv2D(types.Tensor, types.Tensor, types.Tensor, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv2DTransposed(types.Tensor, types.Tensor, types.Tensor, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv2DKernelGrad(types.Tensor, types.Tensor, types.Tensor, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv1DKernelGrad(types.Tensor, types.Tensor, types.Tensor, int, int) types.Tensor {
	return st
}
func (st *simpleTensor) Im2Col(types.Tensor, []int, []int, []int) types.Tensor { return st }
func (st *simpleTensor) Col2Im(types.Tensor, []int, []int, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) MaxPool2D(types.Tensor, []int, []int, []int) types.Tensor { return st }
func (st *simpleTensor) MaxPool2DWithIndices(types.Tensor, types.Tensor, []int, []int, []int) (types.Tensor, types.Tensor) {
	return st, st
}
func (st *simpleTensor) MaxPool2DBackward(types.Tensor, types.Tensor, types.Tensor, []int, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) AvgPool2D(types.Tensor, []int, []int, []int) types.Tensor { return st }
func (st *simpleTensor) AvgPool2DBackward(types.Tensor, types.Tensor, []int, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) GlobalAvgPool2D(types.Tensor) types.Tensor              { return st }
func (st *simpleTensor) AdaptiveAvgPool2D(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) DropoutForward(types.Tensor, types.Tensor) types.Tensor { return st }
func (st *simpleTensor) DropoutMask(float64, float64, types.RNG) types.Tensor   { return st }
func (st *simpleTensor) DropoutBackward(types.Tensor, types.Tensor, types.Tensor) types.Tensor {
	return st
}
