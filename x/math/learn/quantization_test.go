package learn_test

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/learn"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

func TestQuantizationParams(t *testing.T) {
	// Test symmetric quantization
	calibrator := learn.NewCalibrator(learn.CalibMinMax, learn.QuantSymmetric, 8)

	// Add samples with range [-10, 10]
	for i := -10; i <= 10; i++ {
		calibrator.AddSample(float64(i))
	}

	params, err := calibrator.ComputeParams()
	if err != nil {
		t.Fatalf("ComputeParams failed: %v", err)
	}

	// For symmetric quantization, scale should map [-10, 10] to [-127, 127]
	expectedScale := float64(10.0 / 127.0)
	if math.Abs(params.Scale-expectedScale) > 0.001 {
		t.Errorf("Expected scale ~%.6f, got %.6f", expectedScale, params.Scale)
	}

	if params.ZeroPoint != 0 {
		t.Errorf("Expected zero point 0 for symmetric, got %d", params.ZeroPoint)
	}
}

func TestQuantizationAsymmetric(t *testing.T) {
	calibrator := learn.NewCalibrator(learn.CalibMinMax, learn.QuantAsymmetric, 8)

	// Add samples with range [0, 255]
	for i := 0; i <= 255; i++ {
		calibrator.AddSample(float64(i))
	}

	params, err := calibrator.ComputeParams()
	if err != nil {
		t.Fatalf("ComputeParams failed: %v", err)
	}

	// For asymmetric quantization, scale should map [0, 255] to [0, 255]
	expectedScale := float64(1.0)
	if math.Abs(params.Scale-expectedScale) > 0.1 {
		t.Errorf("Expected scale ~%.6f, got %.6f", expectedScale, params.Scale)
	}
}

func TestQuantizeDequantize(t *testing.T) {
	// Create a simple tensor
	input := tensor.FromFloat32(tensor.NewShape(4), []float32{-10.0, -5.0, 0.0, 10.0})

	// Compute quantization parameters
	calibrator := learn.NewCalibrator(learn.CalibMinMax, learn.QuantSymmetric, 8)
	calibrator.AddTensor(input)
	params, err := calibrator.ComputeParams()
	if err != nil {
		t.Fatalf("ComputeParams failed: %v", err)
	}

	// Quantize
	quantized, qp, err := learn.QuantizeTensor(input, params, learn.QuantSymmetric, 8)
	if err != nil {
		t.Fatalf("QuantizeTensor failed: %v", err)
	}

	if qp.Scale != params.Scale {
		t.Errorf("Scale mismatch: expected %.6f, got %.6f", params.Scale, qp.Scale)
	}

	// Dequantize
	dequantized, err := learn.DequantizeTensor(quantized, params)
	if err != nil {
		t.Fatalf("DequantizeTensor failed: %v", err)
	}

	// Check reconstruction error (should be small)
	for indices := range generics.ElementsIndices(input.Shape()) {
		val := input.At(indices...)
		reconstructed := dequantized.At(indices...)
		error := math.Abs(val - reconstructed)
		// Error should be less than one quantization step
		maxError := params.Scale / 2.0
		if error > float64(maxError) {
			t.Errorf("Large reconstruction error at %v: expected ~%.6f, got %.6f (error: %.6f, max: %.6f)",
				indices, val, reconstructed, error, maxError)
		}
	}
}

func TestQuantizationPercentile(t *testing.T) {
	calibrator := learn.NewCalibrator(learn.CalibPercentile, learn.QuantSymmetric, 8)
	calibrator.SetPercentile(0.95) // 95th percentile

	// Add mostly small values, but a few outliers
	for i := 0; i < 100; i++ {
		calibrator.AddSample(float64(i) * 0.1) // Range [0, 9.9]
	}
	// Add outliers
	calibrator.AddSample(100.0)
	calibrator.AddSample(-100.0)

	params, err := calibrator.ComputeParams()
	if err != nil {
		t.Fatalf("ComputeParams failed: %v", err)
	}

	// Scale should be based on percentile range, not the outliers
	// The 95th percentile should ignore the extreme outliers
	if params.Scale > 1.0 {
		t.Errorf("Expected scale < 1.0 to ignore outliers, got %.6f", params.Scale)
	}
}

func TestQuantizationZeroRange(t *testing.T) {
	calibrator := learn.NewCalibrator(learn.CalibMinMax, learn.QuantSymmetric, 8)

	// All values are the same
	for i := 0; i < 10; i++ {
		calibrator.AddSample(5.0)
	}

	params, err := calibrator.ComputeParams()
	if err != nil {
		t.Fatalf("ComputeParams failed: %v", err)
	}

	// Should handle zero range gracefully
	if params.Scale <= 0 {
		t.Errorf("Expected positive scale, got %.6f", params.Scale)
	}
}

func TestQuantizeModel(t *testing.T) {
	// This is a placeholder - we'd need a real model to test
	// For now, test that the function signature works
	t.Skip("Requires actual model implementation to test fully")
}
