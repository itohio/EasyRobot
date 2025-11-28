package st

import (
	"math"
	"testing"
)

// P10.1: Create convert_test.go with test structure
// P10.2: Test same-type fast path (int8→int8, int16→int16, etc.)
func TestConvertSameType(t *testing.T) {
	tests := []struct {
		name string
		src  []int8
		dst  []int8
		n    int
		want []int8
	}{
		{
			name: "int8_to_int8",
			src:  []int8{1, 2, 3, 4, 5},
			dst:  make([]int8, 5),
			n:    5,
			want: []int8{1, 2, 3, 4, 5},
		},
		{
			name: "int8_to_int8_partial",
			src:  []int8{1, 2, 3, 4, 5},
			dst:  make([]int8, 3),
			n:    3,
			want: []int8{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Convert(tt.dst, tt.src, tt.n)
			for i := 0; i < tt.n; i++ {
				if tt.dst[i] != tt.want[i] {
					t.Errorf("dst[%d] = %d, want %d", i, tt.dst[i], tt.want[i])
				}
			}
		})
	}

	// Test int16→int16
	src16 := []int16{100, 200, 300}
	dst16 := make([]int16, 3)
	Convert(dst16, src16, 3)
	for i := range src16 {
		if dst16[i] != src16[i] {
			t.Errorf("int16 copy failed: dst[%d] = %d, want %d", i, dst16[i], src16[i])
		}
	}

	// Test int32→int32
	src32 := []int32{1000, 2000, 3000}
	dst32 := make([]int32, 3)
	Convert(dst32, src32, 3)
	for i := range src32 {
		if dst32[i] != src32[i] {
			t.Errorf("int32 copy failed: dst[%d] = %d, want %d", i, dst32[i], src32[i])
		}
	}

	// Test float32→float32
	srcF32 := []float32{0.5, -0.5, 0.0}
	dstF32 := make([]float32, 3)
	Convert(dstF32, srcF32, 3)
	for i := range srcF32 {
		if dstF32[i] != srcF32[i] {
			t.Errorf("float32 copy failed: dst[%d] = %f, want %f", i, dstF32[i], srcF32[i])
		}
	}

	// Test complex64→complex64
	srcC64 := []complex64{complex(0.5, 0), complex(-0.5, 0), complex(0, 0)}
	dstC64 := make([]complex64, 3)
	Convert(dstC64, srcC64, 3)
	for i := range srcC64 {
		if dstC64[i] != srcC64[i] {
			t.Errorf("complex64 copy failed: dst[%d] = %v, want %v", i, dstC64[i], srcC64[i])
		}
	}
}

// P10.3: Test float32 → int16 conversion (scaling + dithering)
func TestConvertFloat32ToInt16(t *testing.T) {
	// Test boundary values
	src := []float32{-1.0, 0.0, 1.0, 0.5, -0.5}
	dst := make([]int16, len(src))
	Convert(dst, src, len(src))

	// Check that -1.0 maps to approximately -32768
	if dst[0] < -32768 || dst[0] > -32700 {
		t.Errorf("float32 -1.0 -> int16: got %d, want approximately -32768", dst[0])
	}

	// Check that 0.0 maps to approximately 0
	if dst[1] != 0 {
		t.Errorf("float32 0.0 -> int16: got %d, want 0", dst[1])
	}

	// Check that 1.0 maps to approximately 32767
	if dst[2] < 32700 || dst[2] > 32767 {
		t.Errorf("float32 1.0 -> int16: got %d, want approximately 32767", dst[2])
	}

	// Check that 0.5 maps to approximately 16383
	if dst[3] < 16300 || dst[3] > 16400 {
		t.Errorf("float32 0.5 -> int16: got %d, want approximately 16383", dst[3])
	}

	// Check that -0.5 maps to approximately -16383
	if dst[4] > -16300 || dst[4] < -16400 {
		t.Errorf("float32 -0.5 -> int16: got %d, want approximately -16383", dst[4])
	}
}

// P10.4: Test int16 → float32 conversion (scaling only)
func TestConvertInt16ToFloat32(t *testing.T) {
	// Test boundary values
	src := []int16{-32768, 0, 32767, 16383, -16383}
	dst := make([]float32, len(src))
	Convert(dst, src, len(src))

	// Check that -32768 maps to approximately -1.0
	if dst[0] < -1.01 || dst[0] > -0.99 {
		t.Errorf("int16 -32768 -> float32: got %f, want approximately -1.0", dst[0])
	}

	// Check that 0 maps to 0.0
	if dst[1] != 0.0 {
		t.Errorf("int16 0 -> float32: got %f, want 0.0", dst[1])
	}

	// Check that 32767 maps to approximately 1.0
	if dst[2] < 0.99 || dst[2] > 1.01 {
		t.Errorf("int16 32767 -> float32: got %f, want approximately 1.0", dst[2])
	}

	// Check that 16383 maps to approximately 0.5
	if dst[3] < 0.49 || dst[3] > 0.51 {
		t.Errorf("int16 16383 -> float32: got %f, want approximately 0.5", dst[3])
	}

	// Check that -16383 maps to approximately -0.5
	if dst[4] > -0.49 || dst[4] < -0.51 {
		t.Errorf("int16 -16383 -> float32: got %f, want approximately -0.5", dst[4])
	}
}

// P10.5: Test float32 → int8 conversion (scaling + dithering)
func TestConvertFloat32ToInt8(t *testing.T) {
	// Test boundary values
	src := []float32{-1.0, 0.0, 1.0, 0.5, -0.5}
	dst := make([]int8, len(src))
	Convert(dst, src, len(src))

	// Check that -1.0 maps to approximately -128
	if dst[0] < -128 || dst[0] > -120 {
		t.Errorf("float32 -1.0 -> int8: got %d, want approximately -128", dst[0])
	}

	// Check that 0.0 maps to approximately 0
	if dst[1] != 0 {
		t.Errorf("float32 0.0 -> int8: got %d, want 0", dst[1])
	}

	// Check that 1.0 maps to approximately 127
	if dst[2] < 120 || dst[2] > 127 {
		t.Errorf("float32 1.0 -> int8: got %d, want approximately 127", dst[2])
	}

	// Check that values are within expected range (dithering may cause variation)
	for i := range dst {
		if dst[i] < -128 || dst[i] > 127 {
			t.Errorf("int8 value out of range: dst[%d] = %d", i, dst[i])
		}
	}
}

// P10.6: Test int8 → float32 conversion (scaling only)
func TestConvertInt8ToFloat32(t *testing.T) {
	// Test boundary values
	src := []int8{-128, 0, 127, 63, -63}
	dst := make([]float32, len(src))
	Convert(dst, src, len(src))

	// Check that -128 maps to approximately -1.0
	if dst[0] < -1.01 || dst[0] > -0.99 {
		t.Errorf("int8 -128 -> float32: got %f, want approximately -1.0", dst[0])
	}

	// Check that 0 maps to 0.0
	if dst[1] != 0.0 {
		t.Errorf("int8 0 -> float32: got %f, want 0.0", dst[1])
	}

	// Check that 127 maps to approximately 1.0
	if dst[2] < 0.99 || dst[2] > 1.01 {
		t.Errorf("int8 127 -> float32: got %f, want approximately 1.0", dst[2])
	}

	// Check that 63 maps to approximately 0.5
	if dst[3] < 0.49 || dst[3] > 0.51 {
		t.Errorf("int8 63 -> float32: got %f, want approximately 0.5", dst[3])
	}

	// Check that -63 maps to approximately -0.5
	if dst[4] > -0.49 || dst[4] < -0.51 {
		t.Errorf("int8 -63 -> float32: got %f, want approximately -0.5", dst[4])
	}
}

// P10.7: Test int32 → int16 conversion (scaling + dithering)
func TestConvertInt32ToInt16(t *testing.T) {
	// Test boundary values
	src := []int32{-2147483648, 0, 2147483647, 1073741823, -1073741823}
	dst := make([]int16, len(src))
	Convert(dst, src, len(src))

	// Check that values are within int16 range
	for i := range dst {
		if dst[i] < -32768 || dst[i] > 32767 {
			t.Errorf("int16 value out of range: dst[%d] = %d", i, dst[i])
		}
	}

	// Check that 0 maps to 0
	if dst[1] != 0 {
		t.Errorf("int32 0 -> int16: got %d, want 0", dst[1])
	}
}

// P10.8: Test int16 → int32 conversion (scaling only)
func TestConvertInt16ToInt32(t *testing.T) {
	// Test boundary values
	src := []int16{-32768, 0, 32767, 16383, -16383}
	dst := make([]int32, len(src))
	Convert(dst, src, len(src))

	// Check that -32768 maps to approximately -2147483648 (scaled up by 2^16)
	expected := int32(-32768) * 65536
	if dst[0] != expected {
		t.Errorf("int16 -32768 -> int32: got %d, want %d", dst[0], expected)
	}

	// Check that 0 maps to 0
	if dst[1] != 0 {
		t.Errorf("int16 0 -> int32: got %d, want 0", dst[1])
	}

	// Check that 32767 maps to approximately 2147450880 (scaled up by 2^16)
	expected = int32(32767) * 65536
	if dst[2] != expected {
		t.Errorf("int16 32767 -> int32: got %d, want %d", dst[2], expected)
	}
}

// P10.9: Test edge cases: empty slices, boundary values
func TestConvertEdgeCases(t *testing.T) {
	// Test empty slices
	var emptySrc []float32
	var emptyDst []int16
	Convert(emptyDst, emptySrc, 0)

	// Test with n=0
	src := []float32{1.0, 2.0, 3.0}
	dst := make([]int16, 3)
	Convert(dst, src, 0)
	// Should not panic and dst should remain zero

	// Test boundary values for all types
	tests := []struct {
		name string
		src  []float32
		dst  []int8
	}{
		{
			name: "boundary_negative_one",
			src:  []float32{-1.0},
			dst:  make([]int8, 1),
		},
		{
			name: "boundary_one",
			src:  []float32{1.0},
			dst:  make([]int8, 1),
		},
		{
			name: "boundary_zero",
			src:  []float32{0.0},
			dst:  make([]int8, 1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Convert(tt.dst, tt.src, 1)
			if tt.dst[0] < -128 || tt.dst[0] > 127 {
				t.Errorf("value out of range: %d", tt.dst[0])
			}
		})
	}

	// Test int8 boundary values
	int8Src := []int8{-128, 127, 0}
	int8Dst := make([]float32, 3)
	Convert(int8Dst, int8Src, 3)
	if int8Dst[0] < -1.01 || int8Dst[0] > -0.99 {
		t.Errorf("int8 -128 -> float32: got %f, want approximately -1.0", int8Dst[0])
	}
	if int8Dst[1] < 0.99 || int8Dst[1] > 1.01 {
		t.Errorf("int8 127 -> float32: got %f, want approximately 1.0", int8Dst[1])
	}
	if int8Dst[2] != 0.0 {
		t.Errorf("int8 0 -> float32: got %f, want 0.0", int8Dst[2])
	}

	// Test int16 boundary values
	int16Src := []int16{-32768, 32767, 0}
	int16Dst := make([]float32, 3)
	Convert(int16Dst, int16Src, 3)
	if int16Dst[0] < -1.01 || int16Dst[0] > -0.99 {
		t.Errorf("int16 -32768 -> float32: got %f, want approximately -1.0", int16Dst[0])
	}
	if int16Dst[1] < 0.99 || int16Dst[1] > 1.01 {
		t.Errorf("int16 32767 -> float32: got %f, want approximately 1.0", int16Dst[1])
	}
	if int16Dst[2] != 0.0 {
		t.Errorf("int16 0 -> float32: got %f, want 0.0", int16Dst[2])
	}
}

// P10.10: Test dithering quality (verify TPDF distribution)
func TestConvertDitheringQuality(t *testing.T) {
	// Generate many samples to verify TPDF distribution
	const numSamples = 10000
	src := make([]float32, numSamples)
	for i := range src {
		src[i] = 0.0 // Convert zero with dithering
	}
	dst := make([]int8, numSamples)
	Convert(dst, src, numSamples)

	// Count values - with TPDF dithering, we should see distribution around 0
	// Most values should be -1, 0, or 1 (since we're converting 0.0)
	counts := make(map[int8]int)
	for _, v := range dst {
		counts[v]++
	}

	// With TPDF dithering, values should be distributed around 0
	// We expect most values to be -1, 0, or 1 when converting 0.0
	zeroCount := counts[0]
	if zeroCount < numSamples/10 {
		t.Errorf("TPDF dithering: expected significant number of zeros, got %d out of %d", zeroCount, numSamples)
	}

	// Verify that values are within reasonable range (dithering should keep them close to 0)
	maxAbs := int8(0)
	for _, v := range dst {
		if absInt8(v) > maxAbs {
			maxAbs = absInt8(v)
		}
	}
	// With TPDF dithering, most values should be within a few LSBs
	if maxAbs > 10 {
		t.Errorf("TPDF dithering: max absolute value too large: %d", maxAbs)
	}
}

func absInt8(x int8) int8 {
	if x < 0 {
		return -x
	}
	return x
}

// P10.11: Test round-trip conversions (A→B→A)
func TestConvertRoundTrip(t *testing.T) {
	// Test float32 → int16 → float32
	original := []float32{-1.0, -0.5, 0.0, 0.5, 1.0}
	intermediate := make([]int16, len(original))
	recovered := make([]float32, len(original))

	Convert(intermediate, original, len(original))
	Convert(recovered, intermediate, len(intermediate))

	// Round-trip should preserve values within tolerance
	tolerance := float32(0.01)
	for i := range original {
		diff := recovered[i] - original[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Errorf("round-trip float32→int16→float32[%d]: original=%f, recovered=%f, diff=%f", i, original[i], recovered[i], diff)
		}
	}

	// Test float32 → int8 → float32
	intermediate8 := make([]int8, len(original))
	recovered8 := make([]float32, len(original))

	Convert(intermediate8, original, len(original))
	Convert(recovered8, intermediate8, len(intermediate8))

	// Round-trip with int8 has lower precision, so larger tolerance
	tolerance8 := float32(0.02)
	for i := range original {
		diff := recovered8[i] - original[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance8 {
			t.Errorf("round-trip float32→int8→float32[%d]: original=%f, recovered=%f, diff=%f", i, original[i], recovered8[i], diff)
		}
	}
}

// P10.12: Test strided conversions
func TestConvertStrided(t *testing.T) {
	// Create a 2x3 matrix (6 elements total)
	shape := []int{2, 3}
	src := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	dst := make([]int16, 6)

	// Contiguous strides (row-major)
	dstStrides := []int{3, 1}
	srcStrides := []int{3, 1}

	ConvertStrided(dst, src, shape, dstStrides, srcStrides)

	// Verify conversion - values should be clamped to int16 range
	for i := range src {
		// All values > 1.0 will clamp to 32767
		if src[i] >= 1.0 {
			if dst[i] != 32767 {
				t.Errorf("strided conversion[%d]: src=%f, got %d, want 32767 (clamped)", i, src[i], dst[i])
			}
		} else {
			expected := int16(src[i] * 32767.0)
			// Allow some tolerance for dithering/clamping
			diff := dst[i] - expected
			if diff < 0 {
				diff = -diff
			}
			if diff > 100 {
				t.Errorf("strided conversion[%d]: got %d, want approximately %d", i, dst[i], expected)
			}
		}
	}

	// Test with non-contiguous strides (transposed)
	shape2 := []int{3, 2}
	src2 := []float32{0.5, -0.5, 0.3, -0.3, 0.1, -0.1}
	dst2 := make([]int16, 6)
	dstStrides2 := []int{1, 3} // Column-major
	srcStrides2 := []int{1, 3}

	ConvertStrided(dst2, src2, shape2, dstStrides2, srcStrides2)

	// Verify values are converted and in range
	for i := range dst2 {
		if dst2[i] < -32768 || dst2[i] > 32767 {
			t.Errorf("strided conversion value out of range: dst2[%d] = %d", i, dst2[i])
		}
	}
}

// P10.13: Test complex64 ↔ float32 conversions
func TestConvertComplex64Float32(t *testing.T) {
	// Test complex64 → float32 (extract real component)
	srcC64 := []complex64{complex(0.5, 0.3), complex(-0.5, -0.3), complex(0.0, 1.0)}
	dstF32 := make([]float32, len(srcC64))
	Convert(dstF32, srcC64, len(srcC64))

	// Should extract real component
	if dstF32[0] != 0.5 {
		t.Errorf("complex64 -> float32[0]: got %f, want 0.5", dstF32[0])
	}
	if dstF32[1] != -0.5 {
		t.Errorf("complex64 -> float32[1]: got %f, want -0.5", dstF32[1])
	}
	if dstF32[2] != 0.0 {
		t.Errorf("complex64 -> float32[2]: got %f, want 0.0", dstF32[2])
	}

	// Test float32 → complex64
	srcF32 := []float32{0.5, -0.5, 0.0}
	dstC64 := make([]complex64, len(srcF32))
	Convert(dstC64, srcF32, len(srcF32))

	// Should create complex64 with real component and zero imaginary
	for i := range srcF32 {
		if real(dstC64[i]) != srcF32[i] {
			t.Errorf("float32 -> complex64[%d] real: got %f, want %f", i, real(dstC64[i]), srcF32[i])
		}
		if imag(dstC64[i]) != 0.0 {
			t.Errorf("float32 -> complex64[%d] imag: got %f, want 0.0", i, imag(dstC64[i]))
		}
	}
}

// P10.14: Test complex64 ↔ int8 conversions (with dithering)
func TestConvertComplex64Int8(t *testing.T) {
	// Test complex64 → int8 (extract real, scale, dither)
	srcC64 := []complex64{complex(1.0, 0.5), complex(-1.0, -0.5), complex(0.0, 0.0)}
	dstInt8 := make([]int8, len(srcC64))
	Convert(dstInt8, srcC64, len(srcC64))

	// Check that values are in int8 range
	for i := range dstInt8 {
		if dstInt8[i] < -128 || dstInt8[i] > 127 {
			t.Errorf("complex64 -> int8[%d]: value out of range: %d", i, dstInt8[i])
		}
	}

	// Check that 1.0 maps to approximately 127
	if dstInt8[0] < 120 || dstInt8[0] > 127 {
		t.Errorf("complex64(1.0, 0.5) -> int8: got %d, want approximately 127", dstInt8[0])
	}

	// Check that -1.0 maps to approximately -128
	if dstInt8[1] < -128 || dstInt8[1] > -120 {
		t.Errorf("complex64(-1.0, -0.5) -> int8: got %d, want approximately -128", dstInt8[1])
	}

	// Test int8 → complex64
	srcInt8 := []int8{-128, 0, 127}
	dstC64 := make([]complex64, len(srcInt8))
	Convert(dstC64, srcInt8, len(srcInt8))

	// Check that -128 maps to approximately -1.0
	if real(dstC64[0]) < -1.01 || real(dstC64[0]) > -0.99 {
		t.Errorf("int8 -128 -> complex64 real: got %f, want approximately -1.0", real(dstC64[0]))
	}
	if imag(dstC64[0]) != 0.0 {
		t.Errorf("int8 -128 -> complex64 imag: got %f, want 0.0", imag(dstC64[0]))
	}

	// Check that 0 maps to 0.0
	if real(dstC64[1]) != 0.0 {
		t.Errorf("int8 0 -> complex64 real: got %f, want 0.0", real(dstC64[1]))
	}

	// Check that 127 maps to approximately 1.0
	if real(dstC64[2]) < 0.99 || real(dstC64[2]) > 1.01 {
		t.Errorf("int8 127 -> complex64 real: got %f, want approximately 1.0", real(dstC64[2]))
	}
}

// P10.15: Test complex64 ↔ int16 conversions (with dithering)
func TestConvertComplex64Int16(t *testing.T) {
	// Test complex64 → int16 (extract real, scale, dither)
	srcC64 := []complex64{complex(1.0, 0.5), complex(-1.0, -0.5), complex(0.0, 0.0)}
	dstInt16 := make([]int16, len(srcC64))
	Convert(dstInt16, srcC64, len(srcC64))

	// Check that values are in int16 range
	for i := range dstInt16 {
		if dstInt16[i] < -32768 || dstInt16[i] > 32767 {
			t.Errorf("complex64 -> int16[%d]: value out of range: %d", i, dstInt16[i])
		}
	}

	// Check that 1.0 maps to approximately 32767
	if dstInt16[0] < 32700 || dstInt16[0] > 32767 {
		t.Errorf("complex64(1.0, 0.5) -> int16: got %d, want approximately 32767", dstInt16[0])
	}

	// Check that -1.0 maps to approximately -32768
	if dstInt16[1] < -32768 || dstInt16[1] > -32700 {
		t.Errorf("complex64(-1.0, -0.5) -> int16: got %d, want approximately -32768", dstInt16[1])
	}

	// Test int16 → complex64
	srcInt16 := []int16{-32768, 0, 32767}
	dstC64 := make([]complex64, len(srcInt16))
	Convert(dstC64, srcInt16, len(srcInt16))

	// Check that -32768 maps to approximately -1.0
	if real(dstC64[0]) < -1.01 || real(dstC64[0]) > -0.99 {
		t.Errorf("int16 -32768 -> complex64 real: got %f, want approximately -1.0", real(dstC64[0]))
	}

	// Check that 0 maps to 0.0
	if real(dstC64[1]) != 0.0 {
		t.Errorf("int16 0 -> complex64 real: got %f, want 0.0", real(dstC64[1]))
	}

	// Check that 32767 maps to approximately 1.0
	if real(dstC64[2]) < 0.99 || real(dstC64[2]) > 1.01 {
		t.Errorf("int16 32767 -> complex64 real: got %f, want approximately 1.0", real(dstC64[2]))
	}
}

// P10.16: Test complex64 ↔ int32 conversions (scaling only)
func TestConvertComplex64Int32(t *testing.T) {
	// Test complex64 → int32 (extract real, scale, no dithering)
	// Use smaller values to avoid float32 precision issues with MaxInt32
	srcC64 := []complex64{complex(0.5, 0.3), complex(-0.5, -0.3), complex(0.0, 0.0)}
	dstInt32 := make([]int32, len(srcC64))
	Convert(dstInt32, srcC64, len(srcC64))

	// Check that values are in int32 range
	for i := range dstInt32 {
		if dstInt32[i] < math.MinInt32 || dstInt32[i] > math.MaxInt32 {
			t.Errorf("complex64 -> int32[%d]: value out of range: %d", i, dstInt32[i])
		}
	}

	// Check that 0.5 maps to approximately MaxInt32/2
	expected := int32(float32(0.5) * float32(2147483647))
	diff := dstInt32[0] - expected
	if diff < 0 {
		diff = -diff
	}
	if diff > 1000000 {
		t.Errorf("complex64(0.5, 0.3) -> int32: got %d, want approximately %d", dstInt32[0], expected)
	}

	// Check that -0.5 maps to approximately -MaxInt32/2
	expected = int32(float32(-0.5) * float32(2147483647))
	diff = dstInt32[1] - expected
	if diff < 0 {
		diff = -diff
	}
	if diff > 1000000 {
		t.Errorf("complex64(-0.5, -0.3) -> int32: got %d, want approximately %d", dstInt32[1], expected)
	}

	// Test int32 → complex64
	srcInt32 := []int32{1073741823, 0, -1073741823} // Use MaxInt32/2 to avoid precision issues
	dstC64 := make([]complex64, len(srcInt32))
	Convert(dstC64, srcInt32, len(srcInt32))

	// Check that MaxInt32/2 maps to approximately 0.5
	if real(dstC64[0]) < 0.49 || real(dstC64[0]) > 0.51 {
		t.Errorf("int32 1073741823 -> complex64 real: got %f, want approximately 0.5", real(dstC64[0]))
	}

	// Check that 0 maps to 0.0
	if real(dstC64[1]) != 0.0 {
		t.Errorf("int32 0 -> complex64 real: got %f, want 0.0", real(dstC64[1]))
	}

	// Check that -MaxInt32/2 maps to approximately -0.5
	if real(dstC64[2]) > -0.49 || real(dstC64[2]) < -0.51 {
		t.Errorf("int32 -1073741823 -> complex64 real: got %f, want approximately -0.5", real(dstC64[2]))
	}
}

