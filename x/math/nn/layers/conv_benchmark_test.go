package layers

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Benchmark report structure
type BenchmarkReport struct {
	Layer      string
	Operation  string
	Iterations int
	Duration   float64 // nanoseconds per operation
	Throughput float64 // operations per second
}

var benchmarkResults []BenchmarkReport

func init() {
	benchmarkResults = make([]BenchmarkReport, 0)
}

// Helper to create random input tensor
func createRandomInput(shape tensor.Shape) tensor.Tensor {
	size := tensor.NewShape(shape...).Size()
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(rand.NormFloat64()) * 0.1 // Small random values
	}
	return tensor.FromFloat32(shape, data)
}

// Benchmark Conv2D Forward Pass - SMALL (for debugging)
func BenchmarkConv2D_Forward_Small(b *testing.B) {
	batchSize := 1
	inChannels := 3
	outChannels := 16
	height := 28
	width := 28
	kernelH := 3
	kernelW := 3
	strideH := 1
	strideW := 1
	padH := 1
	padW := 1

	layer, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW, UseBias(true), WithCanLearn(false))
	if err != nil {
		b.Fatalf("Failed to create Conv2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv2D layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "Conv2D",
		Operation:  "Forward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)

	// Debug output - allocations are reported automatically by b.ReportAllocs()
}

// Benchmark Conv2D Backward Pass
func BenchmarkConv2D_Backward(b *testing.B) {
	batchSize := 8
	inChannels := 64
	outChannels := 128
	height := 32
	width := 32
	kernelH := 3
	kernelW := 3
	strideH := 1
	strideW := 1
	padH := 1
	padW := 1

	layer, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW, UseBias(true), WithCanLearn(true))
	if err != nil {
		b.Fatalf("Failed to create Conv2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv2D layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "Conv2D",
		Operation:  "Backward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark Conv1D Forward Pass
func BenchmarkConv1D_Forward(b *testing.B) {
	batchSize := 8
	inChannels := 64
	outChannels := 128
	length := 256
	kernelLen := 3
	stride := 1
	pad := 1

	layer, err := NewConv1D(inChannels, outChannels, kernelLen, stride, pad, UseBias(true), WithCanLearn(false))
	if err != nil {
		b.Fatalf("Failed to create Conv1D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, length)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv1D layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "Conv1D",
		Operation:  "Forward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark Conv1D Backward Pass
func BenchmarkConv1D_Backward(b *testing.B) {
	batchSize := 8
	inChannels := 64
	outChannels := 128
	length := 256
	kernelLen := 3
	stride := 1
	pad := 1

	layer, err := NewConv1D(inChannels, outChannels, kernelLen, stride, pad, UseBias(true), WithCanLearn(true))
	if err != nil {
		b.Fatalf("Failed to create Conv1D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, length)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv1D layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "Conv1D",
		Operation:  "Backward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark MaxPool2D Forward Pass
func BenchmarkMaxPool2D_Forward(b *testing.B) {
	batchSize := 4
	channels := 32
	height := 16
	width := 16
	kernelH := 2
	kernelW := 2
	strideH := 2
	strideW := 2
	padH := 0
	padW := 0

	layer, err := NewMaxPool2D(kernelH, kernelW, strideH, strideW, padH, padW)
	if err != nil {
		b.Fatalf("Failed to create MaxPool2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, channels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init MaxPool2D layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "MaxPool2D",
		Operation:  "Forward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark MaxPool2D Backward Pass
func BenchmarkMaxPool2D_Backward(b *testing.B) {
	batchSize := 4
	channels := 32
	height := 16
	width := 16
	kernelH := 2
	kernelW := 2
	strideH := 2
	strideW := 2
	padH := 0
	padW := 0

	layer, err := NewMaxPool2D(kernelH, kernelW, strideH, strideW, padH, padW)
	if err != nil {
		b.Fatalf("Failed to create MaxPool2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, channels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init MaxPool2D layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "MaxPool2D",
		Operation:  "Backward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark AvgPool2D Forward Pass
func BenchmarkAvgPool2D_Forward(b *testing.B) {
	batchSize := 4
	channels := 32
	height := 16
	width := 16
	kernelH := 2
	kernelW := 2
	strideH := 2
	strideW := 2
	padH := 0
	padW := 0

	layer, err := NewAvgPool2D(kernelH, kernelW, strideH, strideW, padH, padW)
	if err != nil {
		b.Fatalf("Failed to create AvgPool2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, channels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init AvgPool2D layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "AvgPool2D",
		Operation:  "Forward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark AvgPool2D Backward Pass
func BenchmarkAvgPool2D_Backward(b *testing.B) {
	batchSize := 4
	channels := 32
	height := 16
	width := 16
	kernelH := 2
	kernelW := 2
	strideH := 2
	strideW := 2
	padH := 0
	padW := 0

	layer, err := NewAvgPool2D(kernelH, kernelW, strideH, strideW, padH, padW)
	if err != nil {
		b.Fatalf("Failed to create AvgPool2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, channels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init AvgPool2D layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}

	// Record result
	report := BenchmarkReport{
		Layer:      "AvgPool2D",
		Operation:  "Backward",
		Iterations: b.N,
		Duration:   float64(b.Elapsed().Nanoseconds()) / float64(b.N),
		Throughput: float64(b.N) / b.Elapsed().Seconds(),
	}
	benchmarkResults = append(benchmarkResults, report)
}

// Benchmark Dense Forward Pass
func BenchmarkDense_Forward(b *testing.B) {
	batchSize := 32
	inFeatures := 256
	outFeatures := 512

	layer, err := NewDense(inFeatures, outFeatures)
	if err != nil {
		b.Fatalf("Failed to create Dense layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inFeatures)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Dense layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// Benchmark Dense Backward Pass
func BenchmarkDense_Backward(b *testing.B) {
	batchSize := 32
	inFeatures := 256
	outFeatures := 512

	layer, err := NewDense(inFeatures, outFeatures, WithCanLearn(true))
	if err != nil {
		b.Fatalf("Failed to create Dense layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inFeatures)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Dense layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}
}

// Benchmark Softmax Forward Pass
func BenchmarkSoftmax_Forward(b *testing.B) {
	batchSize := 32
	features := 128
	dim := 1 // Apply softmax along feature dimension

	layer := NewSoftmax("softmax", dim)

	inputShape := tensor.NewShape(batchSize, features)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Softmax layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// Benchmark Softmax Backward Pass
func BenchmarkSoftmax_Backward(b *testing.B) {
	batchSize := 32
	features := 128
	dim := 1

	layer := NewSoftmax("softmax", dim)

	inputShape := tensor.NewShape(batchSize, features)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Softmax layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}
}

// Benchmark LSTM Forward Pass
func BenchmarkLSTM_Forward(b *testing.B) {
	batchSize := 16
	inputSize := 128
	hiddenSize := 256

	layer, err := NewLSTM(inputSize, hiddenSize, WithCanLearn(false))
	if err != nil {
		b.Fatalf("Failed to create LSTM layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inputSize)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init LSTM layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// Benchmark Dropout Forward Pass (Training Mode)
func BenchmarkDropout_Forward(b *testing.B) {
	batchSize := 32
	features := 512

	layer := NewDropout("dropout", WithDropoutRate(0.5), WithTrainingMode(true))

	inputShape := tensor.NewShape(batchSize, features)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Dropout layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// Benchmark Dropout Backward Pass
func BenchmarkDropout_Backward(b *testing.B) {
	batchSize := 32
	features := 512

	layer := NewDropout("dropout", WithDropoutRate(0.5), WithTrainingMode(true))

	inputShape := tensor.NewShape(batchSize, features)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Dropout layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}
}

// Benchmark Dense_Matched - Matching benchmark for fair comparison
// Input: [1, 1024] (1024 elements), Output: [1, 32] (32 neurons)
func BenchmarkDense_Matched_Forward(b *testing.B) {
	batchSize := 1
	inFeatures := 1024 // 32×32 = 1024
	outFeatures := 32

	layer, err := NewDense(inFeatures, outFeatures)
	if err != nil {
		b.Fatalf("Failed to create Dense layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inFeatures)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Dense layer: %v", err)
	}

	input := createRandomInput(inputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
	}
}

// Benchmark Dense_Matched Backward
func BenchmarkDense_Matched_Backward(b *testing.B) {
	batchSize := 1
	inFeatures := 1024
	outFeatures := 32

	layer, err := NewDense(inFeatures, outFeatures, WithCanLearn(true))
	if err != nil {
		b.Fatalf("Failed to create Dense layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inFeatures)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Dense layer: %v", err)
	}

	input := createRandomInput(inputShape)
	output, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	gradOutput := createRandomInput(output.Shape())

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}
}

// Benchmark Conv1D_Matched - Matching benchmark for fair comparison
// Input: [1, 1, 1024] (1024 elements), Output: [1, 32] (32 neurons)
// Uses reasonable kernel size (3) with stride=1, pad to maintain size
// Then reduces spatial dimension using Mean along spatial axis (dim 2) to get 32 neurons
// This compares: Conv1D(kernel=3) + Mean reduction vs Dense(1024→32)
func BenchmarkConv1D_Matched_Forward(b *testing.B) {
	benchmarkConv1D_Matched_Forward(b, 3)
}

// Benchmark Conv1D_Matched_Kernel5 - Same as above but with kernel size 5
func BenchmarkConv1D_Matched_Forward_Kernel5(b *testing.B) {
	benchmarkConv1D_Matched_Forward(b, 5)
}

// Benchmark Conv1D_Matched_Kernel7 - Same as above but with kernel size 7
func BenchmarkConv1D_Matched_Forward_Kernel7(b *testing.B) {
	benchmarkConv1D_Matched_Forward(b, 7)
}

func benchmarkConv1D_Matched_Forward(b *testing.B, kernelLen int) {
	batchSize := 1
	inChannels := 1
	length := 1024 // 32×32 = 1024
	outChannels := 32
	stride := 1
	pad := kernelLen / 2 // Maintain size: pad = floor(kernel/2) for same output size

	layer, err := NewConv1D(inChannels, outChannels, kernelLen, stride, pad, UseBias(true), WithCanLearn(false))
	if err != nil {
		b.Fatalf("Failed to create Conv1D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, length)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv1D layer: %v", err)
	}

	input := createRandomInput(inputShape)

	// Output after reduction: [batch, channels] = [1, 32]
	reductionOutputShape := tensor.NewShape(batchSize, outChannels)
	reductionOutput := tensor.New(layer.DataType(), reductionOutputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		convOutput, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
		// Reduce spatial dimension (dim 2) using Mean to get [batch, channels]
		_ = convOutput.Mean(reductionOutput, []int{2})
	}
}

// Benchmark Conv1D_Matched Backward
// Same setup as Forward: kernel=3, then Mean reduction
func BenchmarkConv1D_Matched_Backward(b *testing.B) {
	batchSize := 1
	inChannels := 1
	length := 1024
	outChannels := 32
	kernelLen := 3 // Reasonable kernel size
	stride := 1
	pad := 1

	layer, err := NewConv1D(inChannels, outChannels, kernelLen, stride, pad, UseBias(true), WithCanLearn(true))
	if err != nil {
		b.Fatalf("Failed to create Conv1D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, length)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv1D layer: %v", err)
	}

	input := createRandomInput(inputShape)
	convOutput, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	// For backward, we need gradOutput in conv output shape [1, 32, 1024]
	// In a real scenario, this would come from the reduction layer's backward,
	// but for benchmarking we'll create it directly in the conv output shape
	convOutputShape := convOutput.Shape()
	gradOutput := createRandomInput(convOutputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}
}

// Benchmark Conv2D_Matched - Matching benchmark for fair comparison
// Input: [1, 1, 32, 32] (1024 elements), Output: [1, 32] (32 neurons)
// Uses reasonable kernel size (3×3) with stride=1, pad to maintain size
// Then reduces spatial dimensions using GlobalAvgPool2D to get 32 neurons
// This compares: Conv2D(kernel=3×3) + GlobalAvgPool2D vs Dense(1024→32)
func BenchmarkConv2D_Matched_Forward(b *testing.B) {
	benchmarkConv2D_Matched_Forward(b, 3)
}

// Benchmark Conv2D_Matched_Kernel5 - Same as above but with kernel size 5×5
func BenchmarkConv2D_Matched_Forward_Kernel5(b *testing.B) {
	benchmarkConv2D_Matched_Forward(b, 5)
}

// Benchmark Conv2D_Matched_Kernel7 - Same as above but with kernel size 7×7
func BenchmarkConv2D_Matched_Forward_Kernel7(b *testing.B) {
	benchmarkConv2D_Matched_Forward(b, 7)
}

func benchmarkConv2D_Matched_Forward(b *testing.B, kernelSize int) {
	batchSize := 1
	inChannels := 1
	height := 32
	width := 32
	outChannels := 32
	kernelH := kernelSize
	kernelW := kernelSize
	strideH := 1
	strideW := 1
	padH := kernelSize / 2 // Maintain size: pad = floor(kernel/2) for same output size
	padW := kernelSize / 2

	layer, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW, UseBias(true), WithCanLearn(false))
	if err != nil {
		b.Fatalf("Failed to create Conv2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv2D layer: %v", err)
	}

	input := createRandomInput(inputShape)

	// Pre-allocate output for GlobalAvgPool2D: [batch, channels] = [1, 32]
	poolOutputShape := tensor.NewShape(batchSize, outChannels)
	poolOutput := tensor.New(layer.DataType(), poolOutputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		convOutput, err := layer.Forward(input)
		if err != nil {
			b.Fatalf("Forward pass failed: %v", err)
		}
		// Reduce spatial dimensions using GlobalAvgPool2D to get [batch, channels]
		_ = convOutput.GlobalAvgPool2D(poolOutput)
	}
}

// Benchmark Conv2D_Matched Backward
// Same setup as Forward: kernel=3×3, then GlobalAvgPool2D
func BenchmarkConv2D_Matched_Backward(b *testing.B) {
	batchSize := 1
	inChannels := 1
	height := 32
	width := 32
	outChannels := 32
	kernelH := 3 // Reasonable kernel size
	kernelW := 3
	strideH := 1
	strideW := 1
	padH := 1
	padW := 1

	layer, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW, UseBias(true), WithCanLearn(true))
	if err != nil {
		b.Fatalf("Failed to create Conv2D layer: %v", err)
	}

	inputShape := tensor.NewShape(batchSize, inChannels, height, width)
	if err := layer.Init(inputShape); err != nil {
		b.Fatalf("Failed to init Conv2D layer: %v", err)
	}

	input := createRandomInput(inputShape)
	convOutput, err := layer.Forward(input)
	if err != nil {
		b.Fatalf("Forward pass failed: %v", err)
	}

	// For backward, we need gradOutput in conv output shape [1, 32, 32, 32]
	// In a real scenario, this would come from the GlobalAvgPool2D layer's backward,
	// but for benchmarking we'll create it directly in the conv output shape
	convOutputShape := convOutput.Shape()
	gradOutput := createRandomInput(convOutputShape)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := layer.Backward(gradOutput)
		if err != nil {
			b.Fatalf("Backward pass failed: %v", err)
		}
	}
}

// GenerateBenchmarkReport generates a markdown report from benchmark results
func GenerateBenchmarkReport() string {
	report := "# Convolution Layers Benchmark Report\n\n"
	report += "## Overview\n\n"
	report += "This report contains benchmark results for convolution layer forward and backward passes.\n\n"
	report += "## Results\n\n"
	report += "| Layer | Operation | Iterations | Duration (ns/op) | Throughput (ops/s) |\n"
	report += "|-------|-----------|------------|------------------|--------------------|\n"

	for _, result := range benchmarkResults {
		report += fmt.Sprintf("| %s | %s | %d | %.2f | %.2f |\n",
			result.Layer, result.Operation, result.Iterations, result.Duration, result.Throughput)
	}

	return report
}
