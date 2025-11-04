package nn

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Linear performs a fully connected layer: output = input × weight + bias
// Uses primitive.Gemv_N for single sample or primitive.Gemm_NN for batch
// Handles bias addition via primitive.Axpy
// Input shape: [batch, inFeatures] or [inFeatures]
// Weight shape: [inFeatures, outFeatures]
// Bias shape: [outFeatures] or [1, outFeatures]
// Output shape: [batch, outFeatures] or [outFeatures]
func Linear(t *tensor.Tensor, weight, bias *tensor.Tensor) *tensor.Tensor {
	if t == nil || weight == nil {
		return nil
	}

	tShape := t.Shape()
	weightShape := weight.Shape()

	// Check weight dimensions
	if len(weightShape) != 2 {
		panic(fmt.Sprintf("nn.Linear: weight must be 2D [inFeatures, outFeatures], got %v", weightShape))
	}

	inFeatures := weightShape[0]
	outFeatures := weightShape[1]

	// Handle input shape
	if len(tShape) == 1 {
		return linearSingleSample(t, weight, bias, inFeatures, outFeatures)
	}

	if len(tShape) == 2 {
		batchSize := tShape[0]
		return linearBatch(t, weight, bias, batchSize, inFeatures, outFeatures)
	}

	panic(fmt.Sprintf("nn.Linear: input must be 1D or 2D, got %v", tShape))
}

// linearSingleSample handles single sample case: [inFeatures] → [outFeatures]
func linearSingleSample(t, weight, bias *tensor.Tensor, inFeatures, outFeatures int) *tensor.Tensor {
	result := tensor.New(tensor.DTFP32, tensor.NewShape(outFeatures))

	ldA := outFeatures
	M := inFeatures
	N := outFeatures

	fp32.Gemv_T(
		result.Data(),
		weight.Data(),
		t.Data(),
		ldA, M, N,
		1.0, 0.0,
	)

	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outFeatures {
			fp32.Axpy(result.Data(), bias.Data(), 1, 1, outFeatures, 1.0)
		}
	}

	return result
}

// linearBatch handles batch case: [batch, inFeatures] → [batch, outFeatures]
func linearBatch(t, weight, bias *tensor.Tensor, batchSize, inFeatures, outFeatures int) *tensor.Tensor {
	result := tensor.New(tensor.DTFP32, tensor.NewShape(batchSize, outFeatures))

	fp32.Gemm_NN(
		result.Data(),
		t.Data(),
		weight.Data(),
		outFeatures,
		inFeatures,
		outFeatures,
		batchSize,
		outFeatures,
		inFeatures,
		1.0, 0.0,
	)

	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outFeatures {
			for b := 0; b < batchSize; b++ {
				offset := b * outFeatures
				fp32.Axpy(result.Data()[offset:], bias.Data(), 1, 1, outFeatures, 1.0)
			}
		}
	}

	return result
}

// Relu applies ReLU activation in-place: output = max(0, input)
func Relu(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return t
	}

	return t.ReLU()
}

// Sigmoid applies sigmoid activation: output = 1 / (1 + exp(-input))
func Sigmoid(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}

	result := t.Clone()
	return result.Sigmoid()
}

// Tanh applies hyperbolic tangent activation: output = tanh(input)
func Tanh(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}

	result := t.Clone()
	return result.Tanh()
}

// Softmax applies softmax along specified dimension
func Softmax(t *tensor.Tensor, dim int) *tensor.Tensor {
	if t == nil {
		return nil
	}

	tShape := t.Shape()
	if dim < 0 || dim >= len(tShape) {
		panic(fmt.Sprintf("nn.Softmax: dimension %d out of range for shape %v", dim, tShape))
	}

	result := t.Clone()

	if len(tShape) == 1 {
		return softmax1D(result)
	}

	if len(tShape) == 2 {
		if dim == 0 {
			return softmax2DRows(result)
		} else if dim == 1 {
			return softmax2DCols(result)
		}
	}

	panic(fmt.Sprintf("nn.Softmax: unsupported tensor dimensions %d", len(tShape)))
}

func softmax1D(t *tensor.Tensor) *tensor.Tensor {
	data := t.Data()
	maxVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
		}
	}

	var sum float32
	for i := range data {
		data[i] = math32.Exp(data[i] - maxVal)
		sum += data[i]
	}

	if sum > 0 {
		for i := range data {
			data[i] /= sum
		}
	}

	return t
}

func softmax2DRows(t *tensor.Tensor) *tensor.Tensor {
	tShape := t.Shape()
	M, N := tShape[0], tShape[1]
	data := t.Data()

	for j := 0; j < N; j++ {
		maxVal := data[j]
		for i := 1; i < M; i++ {
			val := data[i*N+j]
			if val > maxVal {
				maxVal = val
			}
		}

		var sum float32
		for i := 0; i < M; i++ {
			val := data[i*N+j] - maxVal
			data[i*N+j] = math32.Exp(val)
			sum += data[i*N+j]
		}

		if sum > 0 {
			for i := 0; i < M; i++ {
				data[i*N+j] /= sum
			}
		}
	}

	return t
}

func softmax2DCols(t *tensor.Tensor) *tensor.Tensor {
	tShape := t.Shape()
	M, N := tShape[0], tShape[1]
	data := t.Data()

	for i := 0; i < M; i++ {
		rowStart := i * N
		maxVal := data[rowStart]
		for j := 1; j < N; j++ {
			val := data[rowStart+j]
			if val > maxVal {
				maxVal = val
			}
		}

		var sum float32
		for j := 0; j < N; j++ {
			val := data[rowStart+j] - maxVal
			data[rowStart+j] = math32.Exp(val)
			sum += data[rowStart+j]
		}

		if sum > 0 {
			for j := 0; j < N; j++ {
				data[rowStart+j] /= sum
			}
		}
	}

	return t
}

// MSE computes Mean Squared Error between tensor and target
func MSE(pred, target *tensor.Tensor) float32 {
	if pred == nil || target == nil {
		return 0
	}

	squaredDiff := pred.Clone()
	squaredDiff = squaredDiff.Sub(target)
	squaredDiff = squaredDiff.Mul(squaredDiff)

	size := pred.Shape().Size()
	sum := squaredDiff.Sum()
	if size > 0 {
		return sum.Data()[0] / float32(size)
	}

	return 0
}

// CrossEntropy computes cross-entropy loss between predictions and targets
func CrossEntropy(pred, target *tensor.Tensor) float32 {
	if pred == nil || target == nil {
		return 0
	}

	var loss float32
	predData := pred.Data()
	targetData := target.Data()
	for i := range predData {
		if targetData[i] != 0 && predData[i] > 0 {
			loss -= targetData[i] * math32.Log(predData[i]+1e-10)
		}
	}

	return loss
}
