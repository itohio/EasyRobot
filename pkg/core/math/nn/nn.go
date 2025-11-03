package nn

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
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
	result := &tensor.Tensor{
		Dim:  []int{outFeatures},
		Data: make([]float32, outFeatures),
	}

	ldA := outFeatures
	M := inFeatures
	N := outFeatures

	primitive.Gemv_T(
		result.Data,
		weight.Data,
		t.Data,
		ldA, M, N,
		1.0, 0.0,
	)

	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outFeatures {
			primitive.Axpy(result.Data, bias.Data, 1, 1, outFeatures, 1.0)
		}
	}

	return result
}

// linearBatch handles batch case: [batch, inFeatures] → [batch, outFeatures]
func linearBatch(t, weight, bias *tensor.Tensor, batchSize, inFeatures, outFeatures int) *tensor.Tensor {
	result := &tensor.Tensor{
		Dim:  []int{batchSize, outFeatures},
		Data: make([]float32, batchSize*outFeatures),
	}

	primitive.Gemm_NN(
		result.Data,
		t.Data,
		weight.Data,
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
				primitive.Axpy(result.Data[offset:], bias.Data, 1, 1, outFeatures, 1.0)
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

	for i := range t.Data {
		if t.Data[i] < 0 {
			t.Data[i] = 0
		}
	}

	return t
}

// Sigmoid applies sigmoid activation: output = 1 / (1 + exp(-input))
func Sigmoid(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}

	result := t.Clone()

	for i := range result.Data {
		x := result.Data[i]
		if x > 10 {
			result.Data[i] = 1.0
		} else if x < -10 {
			result.Data[i] = 0.0
		} else {
			result.Data[i] = 1.0 / (1.0 + math32.Exp(-x))
		}
	}

	return result
}

// Tanh applies hyperbolic tangent activation: output = tanh(input)
func Tanh(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}

	result := t.Clone()

	for i := range result.Data {
		result.Data[i] = math32.Tanh(result.Data[i])
	}

	return result
}

// Softmax applies softmax along specified dimension
func Softmax(t *tensor.Tensor, dim int) *tensor.Tensor {
	if t == nil {
		return nil
	}

	if dim < 0 || dim >= len(t.Dim) {
		panic(fmt.Sprintf("nn.Softmax: dimension %d out of range for shape %v", dim, t.Dim))
	}

	result := t.Clone()

	if len(t.Dim) == 1 {
		return softmax1D(result)
	}

	if len(t.Dim) == 2 {
		if dim == 0 {
			return softmax2DRows(result)
		} else if dim == 1 {
			return softmax2DCols(result)
		}
	}

	panic(fmt.Sprintf("nn.Softmax: unsupported tensor dimensions %d", len(t.Dim)))
}

func softmax1D(t *tensor.Tensor) *tensor.Tensor {
	maxVal := t.Data[0]
	for i := 1; i < len(t.Data); i++ {
		if t.Data[i] > maxVal {
			maxVal = t.Data[i]
		}
	}

	var sum float32
	for i := range t.Data {
		t.Data[i] = math32.Exp(t.Data[i] - maxVal)
		sum += t.Data[i]
	}

	if sum > 0 {
		for i := range t.Data {
			t.Data[i] /= sum
		}
	}

	return t
}

func softmax2DRows(t *tensor.Tensor) *tensor.Tensor {
	M, N := t.Dim[0], t.Dim[1]

	for j := 0; j < N; j++ {
		maxVal := t.Data[j]
		for i := 1; i < M; i++ {
			val := t.Data[i*N+j]
			if val > maxVal {
				maxVal = val
			}
		}

		var sum float32
		for i := 0; i < M; i++ {
			val := t.Data[i*N+j] - maxVal
			t.Data[i*N+j] = math32.Exp(val)
			sum += t.Data[i*N+j]
		}

		if sum > 0 {
			for i := 0; i < M; i++ {
				t.Data[i*N+j] /= sum
			}
		}
	}

	return t
}

func softmax2DCols(t *tensor.Tensor) *tensor.Tensor {
	M, N := t.Dim[0], t.Dim[1]

	for i := 0; i < M; i++ {
		rowStart := i * N
		maxVal := t.Data[rowStart]
		for j := 1; j < N; j++ {
			val := t.Data[rowStart+j]
			if val > maxVal {
				maxVal = val
			}
		}

		var sum float32
		for j := 0; j < N; j++ {
			val := t.Data[rowStart+j] - maxVal
			t.Data[rowStart+j] = math32.Exp(val)
			sum += t.Data[rowStart+j]
		}

		if sum > 0 {
			for j := 0; j < N; j++ {
				t.Data[rowStart+j] /= sum
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
	squaredDiff.Sub(target)
	squaredDiff.Mul(squaredDiff)

	size := pred.Size()
	sum := squaredDiff.Sum()
	if size > 0 {
		return sum.Data[0] / float32(size)
	}

	return 0
}

// CrossEntropy computes cross-entropy loss between predictions and targets
func CrossEntropy(pred, target *tensor.Tensor) float32 {
	if pred == nil || target == nil {
		return 0
	}

	var loss float32
	for i := range pred.Data {
		if target.Data[i] != 0 && pred.Data[i] > 0 {
			loss -= target.Data[i] * math32.Log(pred.Data[i]+1e-10)
		}
	}

	return loss
}
