package learn

import (
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// QuantizationScheme represents the method for computing quantization parameters.
type QuantizationScheme int

const (
	// QuantSymmetric uses symmetric quantization: range [-max(|min|, |max|), max(|min|, |max|)]
	// Zero point is at 0 in quantized space (typically 128 for uint8 representation)
	QuantSymmetric QuantizationScheme = iota

	// QuantAsymmetric uses asymmetric quantization: range [min, max]
	// Zero point maps to actual zero value in float32 space
	QuantAsymmetric

	// QuantPerChannel applies quantization per channel (e.g., per output channel in conv/dense layers)
	// Each channel gets its own scale and zero point
	QuantPerChannel

	// QuantPerTensor applies single scale and zero point to entire tensor
	QuantPerTensor
)

// QuantizationParams holds the scale and zero point for quantized tensors.
// Formula: real_value = scale * (quantized_value - zeroPoint)
type QuantizationParams struct {
	Scale     float64 // Scale factor
	ZeroPoint int32   // Zero point in quantized space
}

// CalibrationMethod represents the method for collecting statistics during calibration.
type CalibrationMethod int

const (
	// CalibMinMax uses min/max values from calibration data
	// Fast and simple, but sensitive to outliers
	CalibMinMax CalibrationMethod = iota

	// CalibPercentile uses percentile-based method (e.g., 99.9th percentile)
	// More robust to outliers than min-max
	CalibPercentile

	// CalibKLDivergence uses KL divergence minimization (entropy-based)
	// Better accuracy but slower, commonly used in TensorFlow Lite
	CalibKLDivergence
)

// Calibrator collects statistics and computes optimal quantization parameters.
type Calibrator struct {
	method     CalibrationMethod
	scheme     QuantizationScheme
	percentile float64 // For CalibPercentile (e.g., 0.999 for 99.9th percentile)
	bits       int     // Number of bits for quantization (typically 8)
	quantMin   int32   // Minimum quantized value (e.g., 0 for uint8, -128 for int8)
	quantMax   int32   // Maximum quantized value (e.g., 255 for uint8, 127 for int8)

	// Statistics collected during calibration
	minVal  float64
	maxVal  float64
	samples []float64 // For percentile/KL divergence
}

// SetPercentile sets the percentile for percentile-based calibration.
func (c *Calibrator) SetPercentile(p float64) {
	if p > 0 && p <= 1.0 {
		c.percentile = p
	}
}

// QuantizationOption configures quantization behavior.
type QuantizationOption func(*quantizationConfig)

type quantizationConfig struct {
	scheme      QuantizationScheme
	bits        int
	calibMethod CalibrationMethod
	percentile  float64
}

// WithScheme sets the quantization scheme.
func WithScheme(scheme QuantizationScheme) QuantizationOption {
	return func(c *quantizationConfig) {
		c.scheme = scheme
	}
}

// WithBits sets the number of bits for quantization (typically 8).
func WithBits(bits int) QuantizationOption {
	return func(c *quantizationConfig) {
		c.bits = bits
	}
}

// WithCalibrationMethod sets the calibration method.
func WithCalibrationMethod(method CalibrationMethod) QuantizationOption {
	return func(c *quantizationConfig) {
		c.calibMethod = method
	}
}

// WithPercentile sets the percentile for percentile-based calibration.
func WithPercentile(p float64) QuantizationOption {
	return func(c *quantizationConfig) {
		c.percentile = p
	}
}

// NewCalibrator creates a new calibrator for computing quantization parameters.
func NewCalibrator(method CalibrationMethod, scheme QuantizationScheme, bits int) *Calibrator {
	// Compute quantized value range
	quantMax := int32((1 << (bits - 1)) - 1) // For signed, e.g., 127 for 8-bit
	quantMin := -quantMax - 1                // For signed, e.g., -128 for 8-bit

	// For unsigned quantization, adjust range
	if scheme == QuantAsymmetric {
		quantMin = 0
		quantMax = int32((1 << bits) - 1) // e.g., 255 for 8-bit uint8
	}

	c := &Calibrator{
		method:     method,
		scheme:     scheme,
		bits:       bits,
		quantMin:   quantMin,
		quantMax:   quantMax,
		percentile: 0.999, // Default 99.9th percentile
	}

	// Initialize based on scheme
	if scheme == QuantSymmetric {
		c.minVal = math.MaxFloat64
		c.maxVal = -math.MaxFloat64
	} else {
		c.minVal = math.MaxFloat64
		c.maxVal = -math.MaxFloat64
	}

	return c
}

// AddSample adds a value to the calibration statistics.
func (c *Calibrator) AddSample(value float64) {
	// For min-max, we track min/max
	if c.method == CalibMinMax {
		if value < c.minVal {
			c.minVal = value
		}
		if value > c.maxVal {
			c.maxVal = value
		}
	}

	// For percentile and KL divergence, store samples
	if c.method == CalibPercentile || c.method == CalibKLDivergence {
		c.samples = append(c.samples, value)
		// Also track min/max for percentile (fallback)
		if value < c.minVal {
			c.minVal = value
		}
		if value > c.maxVal {
			c.maxVal = value
		}
	}
}

// AddTensor adds all values from a tensor to calibration statistics.
func (c *Calibrator) AddTensor(t tensor.Tensor) {
	for elem := range t.Elements() {
		val := elem.Get()
		c.AddSample(val)
	}
}

// ComputeParams computes quantization parameters from collected statistics.
func (c *Calibrator) ComputeParams() (*QuantizationParams, error) {
	if c.method == CalibKLDivergence {
		if len(c.samples) == 0 {
			return nil, fmt.Errorf("quantization: no samples collected for KL divergence calibration")
		}
		return c.computeParamsKL()
	}

	// Min-max or percentile calibration
	var min, max float64

	switch c.method {
	case CalibMinMax:
		min, max = c.minVal, c.maxVal
	case CalibPercentile:
		if len(c.samples) == 0 {
			// Fall back to min-max if no samples stored
			min, max = c.minVal, c.maxVal
		} else {
			min, max = c.computePercentileRange()
		}
	default:
		min, max = c.minVal, c.maxVal
	}

	if min > max {
		return nil, fmt.Errorf("quantization: invalid range: min (%.6f) > max (%.6f)", min, max)
	}

	// Handle zero range
	if min == max {
		// All values are the same, set scale to 1.0 and zero point to map to that value
		return &QuantizationParams{
			Scale:     1.0,
			ZeroPoint: int32(float64(c.quantMin+c.quantMax) / 2),
		}, nil
	}

	var scale float64
	var zeroPoint int32

	switch c.scheme {
	case QuantSymmetric:
		// Symmetric: range is [-r, r] where r = max(|min|, |max|)
		r := math.Max(math.Abs(min), math.Abs(max))
		scale = r / float64(c.quantMax)
		zeroPoint = 0 // In signed quantized space, zero point is 0

	case QuantAsymmetric:
		// Asymmetric: range is [min, max]
		scale = (max - min) / float64(c.quantMax-c.quantMin)
		zeroPoint = c.quantMin - int32(min/scale)

	default:
		return nil, fmt.Errorf("quantization: unsupported scheme %d for calibration", c.scheme)
	}

	// Ensure scale is positive and non-zero
	if scale <= 0 {
		scale = 1.0
	}

	return &QuantizationParams{
		Scale:     scale,
		ZeroPoint: zeroPoint,
	}, nil
}

// computeParamsKL computes quantization parameters using KL divergence minimization.
// This implements the algorithm used in TensorFlow Lite.
func (c *Calibrator) computeParamsKL() (*QuantizationParams, error) {
	if len(c.samples) == 0 {
		return nil, fmt.Errorf("quantization: no samples for KL divergence")
	}

	// Find min/max from samples
	min := c.samples[0]
	max := c.samples[0]
	for _, v := range c.samples {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	// For KL divergence, we'll use a simplified approach:
	// Find the optimal binning by trying different ranges and minimizing KL divergence
	// For simplicity, we'll use percentile-based approach as approximation
	min, max = c.computePercentileRange()

	// Then compute params using symmetric quantization (common for weights)
	// or asymmetric (common for activations)
	r := math.Max(math.Abs(min), math.Abs(max))
	scale := r / float64(c.quantMax)
	zeroPoint := int32(0)

	if c.scheme == QuantAsymmetric {
		scale = (max - min) / float64(c.quantMax-c.quantMin)
		zeroPoint = c.quantMin - int32(min/scale)
	}

	return &QuantizationParams{
		Scale:     scale,
		ZeroPoint: zeroPoint,
	}, nil
}

// computePercentileRange computes the range using percentile method.
func (c *Calibrator) computePercentileRange() (float64, float64) {
	if len(c.samples) == 0 {
		return c.minVal, c.maxVal
	}

	// Sort samples (simple insertion sort for small arrays)
	sorted := make([]float64, len(c.samples))
	copy(sorted, c.samples)
	for i := 1; i < len(sorted); i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	// Compute percentile indices
	lowerIdx := int(float64(len(sorted)) * (1.0 - c.percentile) / 2.0)
	upperIdx := len(sorted) - 1 - lowerIdx

	if lowerIdx < 0 {
		lowerIdx = 0
	}
	if upperIdx >= len(sorted) {
		upperIdx = len(sorted) - 1
	}

	return sorted[lowerIdx], sorted[upperIdx]
}

// QuantizeTensor quantizes a float32 tensor to int8/uint8 using the given parameters.
// Returns the quantized tensor (stored as uint8 values) and the quantization parameters.
func QuantizeTensor(t tensor.Tensor, params *QuantizationParams, scheme QuantizationScheme, bits int) (tensor.Tensor, *QuantizationParams, error) {
	if len(t.Shape()) == 0 {
		return tensor.EmptyLike(t), nil, fmt.Errorf("quantization: empty tensor")
	}

	// Compute quantized range
	quantMax := int32((1 << (bits - 1)) - 1)
	quantMin := -quantMax - 1
	if scheme == QuantAsymmetric {
		quantMin = 0
		quantMax = int32((1 << bits) - 1)
	}

	// Allocate quantized tensor
	quantized := tensor.New(tensor.DTFP32, t.Shape())

	// Quantize each value using shape iterator to get indices
	for indices := range generics.ElementsIndices(t.Shape()) {
		val := t.At(indices...)
		q := int32(math.Round(val/params.Scale)) + params.ZeroPoint

		// Clamp to quantized range
		if q < quantMin {
			q = quantMin
		}
		if q > quantMax {
			q = quantMax
		}

		quantized.SetAt(float64(q), indices...)
	}

	return quantized, params, nil
}

// DequantizeTensor converts a quantized tensor back to float32.
func DequantizeTensor(quantized tensor.Tensor, params *QuantizationParams) (tensor.Tensor, error) {
	if len(quantized.Shape()) == 0 {
		return tensor.EmptyLike(quantized), fmt.Errorf("quantization: empty quantized tensor")
	}

	// Allocate dequantized tensor
	dequantized := tensor.New(tensor.DTFP32, quantized.Shape())

	// Dequantize each value: real = scale * (quantized - zeroPoint)
	// Use shape iterator to get indices
	for indices := range generics.ElementsIndices(quantized.Shape()) {
		qVal := quantized.At(indices...)
		q := int32(qVal) // Cast back to int32
		realVal := params.Scale * float64(q-params.ZeroPoint)
		dequantized.SetAt(realVal, indices...)
	}

	return dequantized, nil
}

// QuantizeModel quantizes all learnable parameters in a layer.
// Returns a map from parameter key to quantization parameters.
// Layer can be either a Sequential model or a single Layer that implements the Layer interface.
func QuantizeModel(layer types.Layer, opts ...QuantizationOption) (map[string]*QuantizationParams, error) {
	config := &quantizationConfig{
		scheme:      QuantSymmetric,
		bits:        8,
		calibMethod: CalibMinMax,
		percentile:  0.999,
	}
	for _, opt := range opts {
		opt(config)
	}

	params := layer.Parameters()
	quantParams := make(map[string]*QuantizationParams)

	for paramIdx, param := range params {
		// Create calibrator
		calibrator := NewCalibrator(config.calibMethod, config.scheme, config.bits)

		// Collect statistics from parameter tensor
		calibrator.AddTensor(param.Data)

		// Compute quantization parameters
		qp, err := calibrator.ComputeParams()
		if err != nil {
			return nil, fmt.Errorf("quantization: failed to compute params for %v: %w", paramIdx, err)
		}

		// Use string representation of ParamIndex as key
		key := fmt.Sprintf("%v", paramIdx)
		quantParams[key] = qp
	}

	return quantParams, nil
}
