package primitive

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCopyWithStrides(t *testing.T) {
	t.Run("float32 same type contiguous", func(t *testing.T) {
		src := []float32{1, 2, 3, 4}
		dst := make([]float32, 4)
		shape := []int{4}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{1, 2, 3, 4}, dst)
	})

	t.Run("float32 same type with strides", func(t *testing.T) {
		// Source: [1, 0, 2, 0, 3, 0] with stride 2
		// Dest: [0, 0, 0, 0, 0, 0] with stride 2
		src := []float32{1, 0, 2, 0, 3, 0}
		dst := make([]float32, 6)
		shape := []int{3}
		srcStrides := []int{2}
		dstStrides := []int{2}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{1, 0, 2, 0, 3, 0}, dst)
	})

	t.Run("float32 2D contiguous", func(t *testing.T) {
		src := []float32{1, 2, 3, 4, 5, 6}
		dst := make([]float32, 6)
		shape := []int{2, 3}
		srcStrides := []int{3, 1}
		dstStrides := []int{3, 1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, dst)
	})

	t.Run("float32 2D with different strides", func(t *testing.T) {
		// Source: 2x3 matrix with padding: [1,2,3,0,4,5,6,0]
		// Dest: contiguous 2x3 matrix
		src := []float32{1, 2, 3, 0, 4, 5, 6, 0}
		dst := make([]float32, 6)
		shape := []int{2, 3}
		srcStrides := []int{4, 1} // Source has stride 4 in first dimension
		dstStrides := []int{3, 1} // Dest is contiguous

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, dst)
	})

	t.Run("float64 same type", func(t *testing.T) {
		src := []float64{1.5, 2.5, 3.5}
		dst := make([]float64, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float64{1.5, 2.5, 3.5}, dst)
	})

	t.Run("int16 same type", func(t *testing.T) {
		src := []int16{10, 20, 30}
		dst := make([]int16, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []int16{10, 20, 30}, dst)
	})

	t.Run("int8 same type", func(t *testing.T) {
		src := []int8{1, 2, 3}
		dst := make([]int8, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []int8{1, 2, 3}, dst)
	})

	t.Run("float32 to float64 conversion", func(t *testing.T) {
		src := []float32{1.5, 2.5, 3.5}
		dst := make([]float64, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.InDeltaSlice(t, []float64{1.5, 2.5, 3.5}, dst, 1e-5)
	})

	t.Run("float64 to float32 conversion", func(t *testing.T) {
		src := []float64{1.5, 2.5, 3.5}
		dst := make([]float32, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.InDeltaSlice(t, []float32{1.5, 2.5, 3.5}, dst, 1e-5)
	})

	t.Run("int16 to float32 conversion", func(t *testing.T) {
		src := []int16{10, 20, 30}
		dst := make([]float32, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{10, 20, 30}, dst)
	})

	t.Run("float32 to int16 conversion", func(t *testing.T) {
		src := []float32{10.7, 20.3, 30.9}
		dst := make([]int16, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []int16{10, 20, 30}, dst)
	})

	t.Run("int8 to float32 conversion", func(t *testing.T) {
		src := []int8{10, 20, 30}
		dst := make([]float32, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{10, 20, 30}, dst)
	})

	t.Run("float32 to int8 conversion", func(t *testing.T) {
		src := []float32{10.7, 20.3, 30.9}
		dst := make([]int8, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []int8{10, 20, 30}, dst)
	})

	t.Run("int16 to int8 conversion", func(t *testing.T) {
		src := []int16{10, 20, 30}
		dst := make([]int8, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []int8{10, 20, 30}, dst)
	})

	t.Run("conversion with strides", func(t *testing.T) {
		// Source: [1, 0, 2, 0, 3, 0] with stride 2 (float32)
		// Dest: contiguous (float64)
		src := []float32{1, 0, 2, 0, 3, 0}
		dst := make([]float64, 3)
		shape := []int{3}
		srcStrides := []int{2}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.InDeltaSlice(t, []float64{1, 2, 3}, dst, 1e-5)
	})

	t.Run("3D shape same type", func(t *testing.T) {
		// 2x2x2 tensor
		src := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		dst := make([]float32, 8)
		shape := []int{2, 2, 2}
		srcStrides := []int{4, 2, 1}
		dstStrides := []int{4, 2, 1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		assert.Equal(t, []float32{1, 2, 3, 4, 5, 6, 7, 8}, dst)
	})

	t.Run("3D shape with conversion", func(t *testing.T) {
		src := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		dst := make([]float64, 8)
		shape := []int{2, 2, 2}
		srcStrides := []int{4, 2, 1}
		dstStrides := []int{4, 2, 1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		expected := []float64{1, 2, 3, 4, 5, 6, 7, 8}
		assert.InDeltaSlice(t, expected, dst, 1e-5)
	})

	t.Run("nil src", func(t *testing.T) {
		dst := make([]float32, 3)
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(nil, dst, shape, srcStrides, dstStrides)

		// Should not panic, dst unchanged
		assert.Equal(t, []float32{0, 0, 0}, dst)
	})

	t.Run("nil dst", func(t *testing.T) {
		src := []float32{1, 2, 3}
		shape := []int{3}
		srcStrides := []int{1}
		dstStrides := []int{1}

		// Should not panic
		CopyWithStrides(src, nil, shape, srcStrides, dstStrides)
	})

	t.Run("empty shape", func(t *testing.T) {
		src := []float32{1, 2, 3}
		dst := make([]float32, 3)
		shape := []int{}
		srcStrides := []int{}
		dstStrides := []int{}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		// Should not panic, dst unchanged
		assert.Equal(t, []float32{0, 0, 0}, dst)
	})

	t.Run("zero size shape", func(t *testing.T) {
		src := []float32{1, 2, 3}
		dst := make([]float32, 3)
		shape := []int{0}
		srcStrides := []int{1}
		dstStrides := []int{1}

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		// Should not panic, dst unchanged
		assert.Equal(t, []float32{0, 0, 0}, dst)
	})

	t.Run("missing strides uses defaults", func(t *testing.T) {
		src := []float32{1, 2, 3, 4, 5, 6}
		dst := make([]float32, 6)
		shape := []int{2, 3}
		srcStrides := []int{} // Missing strides
		dstStrides := []int{} // Missing strides

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		// Should use default row-major strides [3, 1]
		assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, dst)
	})

	t.Run("transpose-like copy with strides", func(t *testing.T) {
		// Copy 2x3 matrix with different strides to transpose effect
		// Source: row-major [1,2,3,4,5,6] -> [[1,2,3],[4,5,6]]
		// Dest: column-major layout
		src := []float32{1, 2, 3, 4, 5, 6}
		dst := make([]float32, 6)
		shape := []int{2, 3}
		srcStrides := []int{3, 1} // Row-major
		dstStrides := []int{1, 2} // Column-major

		CopyWithStrides(src, dst, shape, srcStrides, dstStrides)

		// Expected: dst[0]=src[0,0]=1, dst[1]=src[0,1]=2, dst[2]=src[0,2]=3,
		//          dst[3]=src[1,0]=4, dst[4]=src[1,1]=5, dst[5]=src[1,2]=6
		// But with column-major strides [1,2]:
		// dst[0] = src[0*3+0] = 1
		// dst[0+1*2] = dst[2] = src[0*3+1] = 2
		// dst[0+2*2] = dst[4] = src[0*3+2] = 3
		// dst[1] = src[1*3+0] = 4
		// dst[1+1*2] = dst[3] = src[1*3+1] = 5
		// dst[1+2*2] = dst[5] = src[1*3+2] = 6
		// So dst = [1, 4, 2, 5, 3, 6]
		assert.Equal(t, []float32{1, 4, 2, 5, 3, 6}, dst)
	})
}

func TestCopyWithConversion(t *testing.T) {
	t.Run("float32 to float32 same type", func(t *testing.T) {
		src := []float32{1.0, 2.0, 3.0}
		dst := make([]float32, 3)

		result := CopyWithConversion(dst, src)

		assert.Equal(t, []float32{1.0, 2.0, 3.0}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("float64 to float32 conversion", func(t *testing.T) {
		src := []float64{1.5, 2.5, 3.5}
		dst := make([]float32, 3)

		result := CopyWithConversion(dst, src)

		assert.InDeltaSlice(t, []float32{1.5, 2.5, 3.5}, dst, 1e-5)
		assert.Equal(t, dst, result)
	})

	t.Run("float32 to float64 conversion", func(t *testing.T) {
		src := []float32{1.5, 2.5, 3.5}
		dst := make([]float64, 3)

		result := CopyWithConversion(dst, src)

		assert.InDeltaSlice(t, []float64{1.5, 2.5, 3.5}, dst, 1e-5)
		assert.Equal(t, dst, result)
	})

	t.Run("int16 to float32 conversion", func(t *testing.T) {
		src := []int16{10, 20, 30}
		dst := make([]float32, 3)

		result := CopyWithConversion(dst, src)

		assert.Equal(t, []float32{10, 20, 30}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("float32 to int16 conversion", func(t *testing.T) {
		src := []float32{10.7, 20.3, 30.9}
		dst := make([]int16, 3)

		result := CopyWithConversion(dst, src)

		assert.Equal(t, []int16{10, 20, 30}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("int8 to float32 conversion", func(t *testing.T) {
		src := []int8{10, 20, 30}
		dst := make([]float32, 3)

		result := CopyWithConversion(dst, src)

		assert.Equal(t, []float32{10, 20, 30}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("float32 to int8 conversion", func(t *testing.T) {
		src := []float32{10.7, 20.3, 30.9}
		dst := make([]int8, 3)

		result := CopyWithConversion(dst, src)

		assert.Equal(t, []int8{10, 20, 30}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("int16 to int8 conversion with clamping", func(t *testing.T) {
		src := []int16{100, 200, 300, 20000, -20000}
		dst := make([]int8, 5)

		result := CopyWithConversion(dst, src)

		// Values should be clamped: 100, 127 (clamped), 127 (clamped), 127 (clamped), -128 (clamped)
		assert.Equal(t, []int8{100, 127, 127, 127, -128}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("float32 to int8 conversion with large value clamping", func(t *testing.T) {
		src := []float32{123456.0}
		dst := make([]int8, 1)

		result := CopyWithConversion(dst, src)

		// 123456 > 127, so should be clamped to 127 (max int8)
		assert.Equal(t, []int8{127}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("float32 to int8 conversion with negative clamping", func(t *testing.T) {
		src := []float32{-50000.0}
		dst := make([]int8, 1)

		result := CopyWithConversion(dst, src)

		// -50000 < -128, so should be clamped to -128 (min int8)
		assert.Equal(t, []int8{-128}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("int8 to int16 conversion", func(t *testing.T) {
		src := []int8{10, 20, 30}
		dst := make([]int16, 3)

		result := CopyWithConversion(dst, src)

		assert.Equal(t, []int16{10, 20, 30}, dst)
		assert.Equal(t, dst, result)
	})

	t.Run("nil src returns nil", func(t *testing.T) {
		dst := make([]float32, 3)

		result := CopyWithConversion(dst, nil)

		assert.Nil(t, result)
	})

	t.Run("nil dst returns nil", func(t *testing.T) {
		src := []float32{1, 2, 3}

		result := CopyWithConversion(nil, src)

		assert.Nil(t, result)
	})

	t.Run("dst shorter than src", func(t *testing.T) {
		src := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
		dst := make([]float64, 3)

		result := CopyWithConversion(dst, src)

		// Should copy only 3 elements (min of lengths)
		assert.InDeltaSlice(t, []float64{1.0, 2.0, 3.0}, dst, 1e-5)
		assert.Equal(t, dst, result)
	})

	t.Run("src shorter than dst", func(t *testing.T) {
		src := []float32{1.0, 2.0}
		dst := make([]float64, 5)

		result := CopyWithConversion(dst, src)

		// Should copy only 2 elements (min of lengths)
		expected := []float64{1.0, 2.0, 0, 0, 0}
		assert.InDeltaSlice(t, expected, dst, 1e-5)
		assert.Equal(t, dst, result)
	})

	t.Run("wrong dst type returns nil", func(t *testing.T) {
		src := []float32{1.0, 2.0, 3.0}
		dst := []string{"a", "b", "c"} // Wrong type (not a numeric type)

		result := CopyWithConversion(dst, src)

		assert.Nil(t, result)
	})

	t.Run("unknown src type returns nil", func(t *testing.T) {
		src := []string{"a", "b", "c"}
		dst := make([]float32, 3)

		result := CopyWithConversion(dst, src)

		assert.Nil(t, result)
	})
}

// TestCopyWithConversionMatrix tests all supported type combinations
// This ensures every conversion path is exercised: 7 types x 7 types = 49 combinations
func TestCopyWithConversionMatrix(t *testing.T) {
	// Define all supported types
	types := []struct {
		name string
		make func([]int) any
		cast func(any) []int
		zero func(int) any
	}{
		{"float64", func(v []int) any {
			r := make([]float64, len(v))
			for i, x := range v {
				r[i] = float64(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]float64)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]float64, n) }},
		{"int64", func(v []int) any {
			r := make([]int64, len(v))
			for i, x := range v {
				r[i] = int64(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int64)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int64, n) }},
		{"float32", func(v []int) any {
			r := make([]float32, len(v))
			for i, x := range v {
				r[i] = float32(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]float32)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]float32, n) }},
		{"int", func(v []int) any {
			r := make([]int, len(v))
			copy(r, v)
			return r
		}, func(a any) []int {
			return a.([]int)
		}, func(n int) any { return make([]int, n) }},
		{"int32", func(v []int) any {
			r := make([]int32, len(v))
			for i, x := range v {
				r[i] = int32(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int32)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int32, n) }},
		{"int16", func(v []int) any {
			r := make([]int16, len(v))
			for i, x := range v {
				r[i] = int16(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int16)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int16, n) }},
		{"int8", func(v []int) any {
			r := make([]int8, len(v))
			for i, x := range v {
				r[i] = int8(x)
			}
			return r
		}, func(a any) []int {
			aa := a.([]int8)
			r := make([]int, len(aa))
			for i, x := range aa {
				r[i] = int(x)
			}
			return r
		}, func(n int) any { return make([]int8, n) }},
	}

	// Test values: small values that fit in all types
	testValues := []int{1, 2, 3, 10, 20, 30, 100, -50, -100}

	for _, srcType := range types {
		for _, dstType := range types {
			t.Run(srcType.name+"_to_"+dstType.name, func(t *testing.T) {
				src := srcType.make(testValues)
				dst := dstType.zero(len(testValues))

				result := CopyWithConversion(dst, src)

				// Should not return nil for valid conversions
				assert.NotNil(t, result, "conversion from %s to %s should not return nil", srcType.name, dstType.name)
				assert.Equal(t, dst, result, "result should be dst")

				// For numeric conversions, verify approximate equality
				// Extract values and compare (accounting for clamping)
				srcValues := srcType.cast(src)
				dstValues := dstType.cast(result)

				assert.Equal(t, len(srcValues), len(dstValues), "lengths should match")

				// For same-type conversions, values should match exactly
				if srcType.name == dstType.name {
					assert.Equal(t, srcValues, dstValues, "same-type conversion should preserve values")
				} else {
					// For cross-type conversions, check approximate equality
					// Down-conversions to int8/int16 may clamp values
					for i := range srcValues {
						srcVal := srcValues[i]
						dstVal := dstValues[i]

						// Check if clamping occurred (for down-conversions to smaller int types)
						if dstType.name == "int8" {
							if srcVal > 127 {
								assert.Equal(t, 127, dstVal, "value %d should be clamped to 127", srcVal)
							} else if srcVal < -128 {
								assert.Equal(t, -128, dstVal, "value %d should be clamped to -128", srcVal)
							} else {
								assert.Equal(t, srcVal, dstVal, "value should match")
							}
						} else if dstType.name == "int16" {
							if srcVal > 32767 {
								assert.Equal(t, 32767, dstVal, "value %d should be clamped to 32767", srcVal)
							} else if srcVal < -32768 {
								assert.Equal(t, -32768, dstVal, "value %d should be clamped to -32768", srcVal)
							} else {
								assert.Equal(t, srcVal, dstVal, "value should match")
							}
						} else if dstType.name == "int32" {
							// int32 can hold values up to ±2^31-1, but we only test small values
							assert.Equal(t, srcVal, dstVal, "value should match")
						} else {
							// For float types and larger int types, values should match (within precision)
							assert.Equal(t, srcVal, dstVal, "value should match")
						}
					}
				}
			})
		}
	}
}

// TestCopyWithConversionClamping tests clamping behavior for all down-conversions
func TestCopyWithConversionClamping(t *testing.T) {
	tests := []struct {
		name     string
		src      any
		dst      any
		expected any
	}{
		// float64 -> int8 clamping
		{
			name:     "float64_to_int8_positive_clamp",
			src:      []float64{1000.0, 200.0, 50.0},
			dst:      make([]int8, 3),
			expected: []int8{127, 127, 50},
		},
		{
			name:     "float64_to_int8_negative_clamp",
			src:      []float64{-1000.0, -200.0, -50.0},
			dst:      make([]int8, 3),
			expected: []int8{-128, -128, -50},
		},
		// float32 -> int8 clamping
		{
			name:     "float32_to_int8_clamp",
			src:      []float32{500.0, -500.0, 100.0},
			dst:      make([]int8, 3),
			expected: []int8{127, -128, 100},
		},
		// int64 -> int8 clamping
		{
			name:     "int64_to_int8_clamp",
			src:      []int64{1000, 200, -1000},
			dst:      make([]int8, 3),
			expected: []int8{127, 127, -128},
		},
		// int32 -> int8 clamping
		{
			name:     "int32_to_int8_clamp",
			src:      []int32{1000, 200, -1000},
			dst:      make([]int8, 3),
			expected: []int8{127, 127, -128},
		},
		// int16 -> int8 clamping
		{
			name:     "int16_to_int8_clamp",
			src:      []int16{1000, 200, -1000},
			dst:      make([]int8, 3),
			expected: []int8{127, 127, -128},
		},
		// int -> int8 clamping
		{
			name:     "int_to_int8_clamp",
			src:      []int{1000, 200, -1000},
			dst:      make([]int8, 3),
			expected: []int8{127, 127, -128},
		},
		// float64 -> int16 clamping
		{
			name:     "float64_to_int16_clamp",
			src:      []float64{50000.0, 20000.0, -50000.0},
			dst:      make([]int16, 3),
			expected: []int16{32767, 20000, -32768},
		},
		// float32 -> int16 clamping
		{
			name:     "float32_to_int16_clamp",
			src:      []float32{50000.0, 20000.0, -50000.0},
			dst:      make([]int16, 3),
			expected: []int16{32767, 20000, -32768},
		},
		// int64 -> int16 clamping
		{
			name:     "int64_to_int16_clamp",
			src:      []int64{50000, 20000, -50000},
			dst:      make([]int16, 3),
			expected: []int16{32767, 20000, -32768},
		},
		// int32 -> int16 clamping
		{
			name:     "int32_to_int16_clamp",
			src:      []int32{50000, 20000, -50000},
			dst:      make([]int16, 3),
			expected: []int16{32767, 20000, -32768},
		},
		// int -> int16 clamping
		{
			name:     "int_to_int16_clamp",
			src:      []int{50000, 20000, -50000},
			dst:      make([]int16, 3),
			expected: []int16{32767, 20000, -32768},
		},
		// float64 -> int32 clamping
		{
			name:     "float64_to_int32_clamp",
			src:      []float64{3e9, 2e9, -3e9},
			dst:      make([]int32, 3),
			expected: []int32{2147483647, 2000000000, -2147483648},
		},
		// float32 -> int32 clamping
		{
			name:     "float32_to_int32_clamp",
			src:      []float32{3e9, 2000000000.0, -3e9},
			dst:      make([]int32, 3),
			expected: []int32{2147483647, 2000000000, -2147483648},
		},
		// int64 -> int32 clamping
		{
			name:     "int64_to_int32_clamp",
			src:      []int64{3000000000, 2000000000, -3000000000},
			dst:      make([]int32, 3),
			expected: []int32{2147483647, 2000000000, -2147483648},
		},
		// int -> int32 clamping (on 64-bit platforms where int is int64)
		{
			name:     "int_to_int32_clamp",
			src:      []int{math.MaxInt, 2000000000, math.MinInt},
			dst:      make([]int32, 3),
			expected: []int32{2147483647, 2000000000, -2147483648},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CopyWithConversion(tt.dst, tt.src)
			assert.NotNil(t, result)
			assert.Equal(t, tt.dst, result)

			// Compare expected values
			switch expected := tt.expected.(type) {
			case []int8:
				actual := result.([]int8)
				assert.Equal(t, expected, actual)
			case []int16:
				actual := result.([]int16)
				assert.Equal(t, expected, actual)
			case []int32:
				actual := result.([]int32)
				assert.Equal(t, expected, actual)
			}
		})
	}
}

// TestConvertValueMatrix tests all supported type combinations for ConvertValue
// This ensures every conversion path is exercised: 7 types x 7 types = 49 combinations
func TestConvertValueMatrix(t *testing.T) {
	// Define all supported types with their test values
	typeTestCases := []struct {
		name      string
		testValue int // Base test value
		makeValue func(int) any
		verify    func(t *testing.T, srcVal int, dstVal any, dstType string)
	}{
		{"float64", 42, func(v int) any { return float64(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.Equal(t, float64(srcVal), dstVal.(float64))
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				expected := int64(srcVal)
				assert.Equal(t, expected, dstVal.(int64))
			case "int":
				expected := int(srcVal)
				assert.Equal(t, expected, dstVal.(int))
			case "int32":
				expected := int32(srcVal)
				if int64(srcVal) > 2147483647 {
					assert.Equal(t, int32(2147483647), dstVal.(int32))
				} else if int64(srcVal) < -2147483648 {
					assert.Equal(t, int32(-2147483648), dstVal.(int32))
				} else {
					assert.Equal(t, expected, dstVal.(int32))
				}
			case "int16":
				expected := int16(srcVal)
				if int64(srcVal) > 32767 {
					assert.Equal(t, int16(32767), dstVal.(int16))
				} else if int64(srcVal) < -32768 {
					assert.Equal(t, int16(-32768), dstVal.(int16))
				} else {
					assert.Equal(t, expected, dstVal.(int16))
				}
			case "int8":
				expected := int8(srcVal)
				if int64(srcVal) > 127 {
					assert.Equal(t, int8(127), dstVal.(int8))
				} else if int64(srcVal) < -128 {
					assert.Equal(t, int8(-128), dstVal.(int8))
				} else {
					assert.Equal(t, expected, dstVal.(int8))
				}
			}
		}},
		{"int64", 42, func(v int) any { return int64(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.Equal(t, float64(srcVal), dstVal.(float64))
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				assert.Equal(t, int64(srcVal), dstVal.(int64))
			case "int":
				assert.Equal(t, int(srcVal), dstVal.(int))
			case "int32":
				expected := int32(srcVal)
				if int64(srcVal) > 2147483647 {
					assert.Equal(t, int32(2147483647), dstVal.(int32))
				} else if int64(srcVal) < -2147483648 {
					assert.Equal(t, int32(-2147483648), dstVal.(int32))
				} else {
					assert.Equal(t, expected, dstVal.(int32))
				}
			case "int16":
				expected := int16(srcVal)
				if int64(srcVal) > 32767 {
					assert.Equal(t, int16(32767), dstVal.(int16))
				} else if int64(srcVal) < -32768 {
					assert.Equal(t, int16(-32768), dstVal.(int16))
				} else {
					assert.Equal(t, expected, dstVal.(int16))
				}
			case "int8":
				expected := int8(srcVal)
				if int64(srcVal) > 127 {
					assert.Equal(t, int8(127), dstVal.(int8))
				} else if int64(srcVal) < -128 {
					assert.Equal(t, int8(-128), dstVal.(int8))
				} else {
					assert.Equal(t, expected, dstVal.(int8))
				}
			}
		}},
		{"float32", 42, func(v int) any { return float32(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.InDelta(t, float64(srcVal), dstVal.(float64), 0.01)
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				assert.Equal(t, int64(srcVal), dstVal.(int64))
			case "int":
				assert.Equal(t, int(srcVal), dstVal.(int))
			case "int32":
				expected := int32(srcVal)
				if int64(srcVal) > 2147483647 {
					assert.Equal(t, int32(2147483647), dstVal.(int32))
				} else if int64(srcVal) < -2147483648 {
					assert.Equal(t, int32(-2147483648), dstVal.(int32))
				} else {
					assert.Equal(t, expected, dstVal.(int32))
				}
			case "int16":
				expected := int16(srcVal)
				if int64(srcVal) > 32767 {
					assert.Equal(t, int16(32767), dstVal.(int16))
				} else if int64(srcVal) < -32768 {
					assert.Equal(t, int16(-32768), dstVal.(int16))
				} else {
					assert.Equal(t, expected, dstVal.(int16))
				}
			case "int8":
				expected := int8(srcVal)
				if int64(srcVal) > 127 {
					assert.Equal(t, int8(127), dstVal.(int8))
				} else if int64(srcVal) < -128 {
					assert.Equal(t, int8(-128), dstVal.(int8))
				} else {
					assert.Equal(t, expected, dstVal.(int8))
				}
			}
		}},
		{"int", 42, func(v int) any { return int(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.Equal(t, float64(srcVal), dstVal.(float64))
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				assert.Equal(t, int64(srcVal), dstVal.(int64))
			case "int":
				assert.Equal(t, int(srcVal), dstVal.(int))
			case "int32":
				expected := int32(srcVal)
				if int64(srcVal) > 2147483647 {
					assert.Equal(t, int32(2147483647), dstVal.(int32))
				} else if int64(srcVal) < -2147483648 {
					assert.Equal(t, int32(-2147483648), dstVal.(int32))
				} else {
					assert.Equal(t, expected, dstVal.(int32))
				}
			case "int16":
				expected := int16(srcVal)
				if int64(srcVal) > 32767 {
					assert.Equal(t, int16(32767), dstVal.(int16))
				} else if int64(srcVal) < -32768 {
					assert.Equal(t, int16(-32768), dstVal.(int16))
				} else {
					assert.Equal(t, expected, dstVal.(int16))
				}
			case "int8":
				expected := int8(srcVal)
				if int64(srcVal) > 127 {
					assert.Equal(t, int8(127), dstVal.(int8))
				} else if int64(srcVal) < -128 {
					assert.Equal(t, int8(-128), dstVal.(int8))
				} else {
					assert.Equal(t, expected, dstVal.(int8))
				}
			}
		}},
		{"int32", 42, func(v int) any { return int32(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.Equal(t, float64(srcVal), dstVal.(float64))
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				assert.Equal(t, int64(srcVal), dstVal.(int64))
			case "int":
				assert.Equal(t, int(srcVal), dstVal.(int))
			case "int32":
				assert.Equal(t, int32(srcVal), dstVal.(int32))
			case "int16":
				expected := int16(srcVal)
				if int64(srcVal) > 32767 {
					assert.Equal(t, int16(32767), dstVal.(int16))
				} else if int64(srcVal) < -32768 {
					assert.Equal(t, int16(-32768), dstVal.(int16))
				} else {
					assert.Equal(t, expected, dstVal.(int16))
				}
			case "int8":
				expected := int8(srcVal)
				if int64(srcVal) > 127 {
					assert.Equal(t, int8(127), dstVal.(int8))
				} else if int64(srcVal) < -128 {
					assert.Equal(t, int8(-128), dstVal.(int8))
				} else {
					assert.Equal(t, expected, dstVal.(int8))
				}
			}
		}},
		{"int16", 42, func(v int) any { return int16(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.Equal(t, float64(srcVal), dstVal.(float64))
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				assert.Equal(t, int64(srcVal), dstVal.(int64))
			case "int":
				assert.Equal(t, int(srcVal), dstVal.(int))
			case "int32":
				assert.Equal(t, int32(srcVal), dstVal.(int32))
			case "int16":
				assert.Equal(t, int16(srcVal), dstVal.(int16))
			case "int8":
				expected := int8(srcVal)
				if int64(srcVal) > 127 {
					assert.Equal(t, int8(127), dstVal.(int8))
				} else if int64(srcVal) < -128 {
					assert.Equal(t, int8(-128), dstVal.(int8))
				} else {
					assert.Equal(t, expected, dstVal.(int8))
				}
			}
		}},
		{"int8", 42, func(v int) any { return int8(v) }, func(t *testing.T, srcVal int, dstVal any, dstType string) {
			switch dstType {
			case "float64":
				assert.Equal(t, float64(srcVal), dstVal.(float64))
			case "float32":
				assert.InDelta(t, float64(srcVal), float64(dstVal.(float32)), 0.01)
			case "int64":
				assert.Equal(t, int64(srcVal), dstVal.(int64))
			case "int":
				assert.Equal(t, int(srcVal), dstVal.(int))
			case "int32":
				assert.Equal(t, int32(srcVal), dstVal.(int32))
			case "int16":
				assert.Equal(t, int16(srcVal), dstVal.(int16))
			case "int8":
				assert.Equal(t, int8(srcVal), dstVal.(int8))
			}
		}},
	}

	// Test values: small values that fit in all types
	testValues := []int{1, 2, 10, 42, 100, -50, -100}

	for _, srcType := range typeTestCases {
		for _, dstType := range typeTestCases {
			for _, testVal := range testValues {
				testName := srcType.name + "_to_" + dstType.name + "_value_" + fmt.Sprintf("%d", testVal)
				t.Run(testName, func(t *testing.T) {
					srcValue := srcType.makeValue(testVal)

					// Convert using ConvertValue
					var result any
					switch srcValue := srcValue.(type) {
					case float64:
						switch dstType.name {
						case "float64":
							result = ConvertValue[float64, float64](srcValue)
						case "float32":
							result = ConvertValue[float64, float32](srcValue)
						case "int64":
							result = ConvertValue[float64, int64](srcValue)
						case "int":
							result = ConvertValue[float64, int](srcValue)
						case "int32":
							result = ConvertValue[float64, int32](srcValue)
						case "int16":
							result = ConvertValue[float64, int16](srcValue)
						case "int8":
							result = ConvertValue[float64, int8](srcValue)
						}
					case float32:
						switch dstType.name {
						case "float64":
							result = ConvertValue[float32, float64](srcValue)
						case "float32":
							result = ConvertValue[float32, float32](srcValue)
						case "int64":
							result = ConvertValue[float32, int64](srcValue)
						case "int":
							result = ConvertValue[float32, int](srcValue)
						case "int32":
							result = ConvertValue[float32, int32](srcValue)
						case "int16":
							result = ConvertValue[float32, int16](srcValue)
						case "int8":
							result = ConvertValue[float32, int8](srcValue)
						}
					case int64:
						switch dstType.name {
						case "float64":
							result = ConvertValue[int64, float64](srcValue)
						case "float32":
							result = ConvertValue[int64, float32](srcValue)
						case "int64":
							result = ConvertValue[int64, int64](srcValue)
						case "int":
							result = ConvertValue[int64, int](srcValue)
						case "int32":
							result = ConvertValue[int64, int32](srcValue)
						case "int16":
							result = ConvertValue[int64, int16](srcValue)
						case "int8":
							result = ConvertValue[int64, int8](srcValue)
						}
					case int:
						switch dstType.name {
						case "float64":
							result = ConvertValue[int, float64](srcValue)
						case "float32":
							result = ConvertValue[int, float32](srcValue)
						case "int64":
							result = ConvertValue[int, int64](srcValue)
						case "int":
							result = ConvertValue[int, int](srcValue)
						case "int32":
							result = ConvertValue[int, int32](srcValue)
						case "int16":
							result = ConvertValue[int, int16](srcValue)
						case "int8":
							result = ConvertValue[int, int8](srcValue)
						}
					case int32:
						switch dstType.name {
						case "float64":
							result = ConvertValue[int32, float64](srcValue)
						case "float32":
							result = ConvertValue[int32, float32](srcValue)
						case "int64":
							result = ConvertValue[int32, int64](srcValue)
						case "int":
							result = ConvertValue[int32, int](srcValue)
						case "int32":
							result = ConvertValue[int32, int32](srcValue)
						case "int16":
							result = ConvertValue[int32, int16](srcValue)
						case "int8":
							result = ConvertValue[int32, int8](srcValue)
						}
					case int16:
						switch dstType.name {
						case "float64":
							result = ConvertValue[int16, float64](srcValue)
						case "float32":
							result = ConvertValue[int16, float32](srcValue)
						case "int64":
							result = ConvertValue[int16, int64](srcValue)
						case "int":
							result = ConvertValue[int16, int](srcValue)
						case "int32":
							result = ConvertValue[int16, int32](srcValue)
						case "int16":
							result = ConvertValue[int16, int16](srcValue)
						case "int8":
							result = ConvertValue[int16, int8](srcValue)
						}
					case int8:
						switch dstType.name {
						case "float64":
							result = ConvertValue[int8, float64](srcValue)
						case "float32":
							result = ConvertValue[int8, float32](srcValue)
						case "int64":
							result = ConvertValue[int8, int64](srcValue)
						case "int":
							result = ConvertValue[int8, int](srcValue)
						case "int32":
							result = ConvertValue[int8, int32](srcValue)
						case "int16":
							result = ConvertValue[int8, int16](srcValue)
						case "int8":
							result = ConvertValue[int8, int8](srcValue)
						}
					}

					// Verify the result
					assert.NotNil(t, result, "ConvertValue should not return zero value")
					dstType.verify(t, testVal, result, dstType.name)
				})
			}
		}
	}
}

// TestConvertValueClamping tests clamping behavior for ConvertValue
func TestConvertValueClamping(t *testing.T) {
	tests := []struct {
		name     string
		src      any
		expected any
	}{
		// float64 -> int8 clamping
		{
			name:     "float64_to_int8_positive",
			src:      float64(1000),
			expected: int8(127),
		},
		{
			name:     "float64_to_int8_negative",
			src:      float64(-1000),
			expected: int8(-128),
		},
		// float32 -> int8 clamping
		{
			name:     "float32_to_int8_positive",
			src:      float32(500),
			expected: int8(127),
		},
		{
			name:     "float32_to_int8_negative",
			src:      float32(-500),
			expected: int8(-128),
		},
		// int64 -> int8 clamping
		{
			name:     "int64_to_int8_positive",
			src:      int64(1000),
			expected: int8(127),
		},
		{
			name:     "int64_to_int8_negative",
			src:      int64(-1000),
			expected: int8(-128),
		},
		// int32 -> int8 clamping
		{
			name:     "int32_to_int8_positive",
			src:      int32(1000),
			expected: int8(127),
		},
		{
			name:     "int32_to_int8_negative",
			src:      int32(-1000),
			expected: int8(-128),
		},
		// int16 -> int8 clamping
		{
			name:     "int16_to_int8_positive",
			src:      int16(1000),
			expected: int8(127),
		},
		{
			name:     "int16_to_int8_negative",
			src:      int16(-1000),
			expected: int8(-128),
		},
		// int -> int8 clamping
		{
			name:     "int_to_int8_positive",
			src:      int(1000),
			expected: int8(127),
		},
		{
			name:     "int_to_int8_negative",
			src:      int(-1000),
			expected: int8(-128),
		},
		// float64 -> int16 clamping
		{
			name:     "float64_to_int16_positive",
			src:      float64(50000),
			expected: int16(32767),
		},
		{
			name:     "float64_to_int16_negative",
			src:      float64(-50000),
			expected: int16(-32768),
		},
		// float32 -> int16 clamping
		{
			name:     "float32_to_int16_positive",
			src:      float32(50000),
			expected: int16(32767),
		},
		{
			name:     "float32_to_int16_negative",
			src:      float32(-50000),
			expected: int16(-32768),
		},
		// int64 -> int16 clamping
		{
			name:     "int64_to_int16_positive",
			src:      int64(50000),
			expected: int16(32767),
		},
		{
			name:     "int64_to_int16_negative",
			src:      int64(-50000),
			expected: int16(-32768),
		},
		// int32 -> int16 clamping
		{
			name:     "int32_to_int16_positive",
			src:      int32(50000),
			expected: int16(32767),
		},
		{
			name:     "int32_to_int16_negative",
			src:      int32(-50000),
			expected: int16(-32768),
		},
		// int -> int16 clamping
		{
			name:     "int_to_int16_positive",
			src:      int(50000),
			expected: int16(32767),
		},
		{
			name:     "int_to_int16_negative",
			src:      int(-50000),
			expected: int16(-32768),
		},
		// float64 -> int32 clamping
		{
			name:     "float64_to_int32_positive",
			src:      float64(3e9),
			expected: int32(2147483647),
		},
		{
			name:     "float64_to_int32_negative",
			src:      float64(-3e9),
			expected: int32(-2147483648),
		},
		// float32 -> int32 clamping
		{
			name:     "float32_to_int32_positive",
			src:      float32(3e9),
			expected: int32(2147483647),
		},
		{
			name:     "float32_to_int32_negative",
			src:      float32(-3e9),
			expected: int32(-2147483648),
		},
		// int64 -> int32 clamping
		{
			name:     "int64_to_int32_positive",
			src:      int64(3000000000),
			expected: int32(2147483647),
		},
		{
			name:     "int64_to_int32_negative",
			src:      int64(-3000000000),
			expected: int32(-2147483648),
		},
		// int -> int32 clamping (on 64-bit platforms)
		{
			name:     "int_to_int32_positive",
			src:      int(math.MaxInt),
			expected: int32(2147483647),
		},
		{
			name:     "int_to_int32_negative",
			src:      int(math.MinInt),
			expected: int32(-2147483648),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result any
			switch src := tt.src.(type) {
			case float64:
				switch expected := tt.expected.(type) {
				case int8:
					result = ConvertValue[float64, int8](src)
					assert.Equal(t, expected, result)
				case int16:
					result = ConvertValue[float64, int16](src)
					assert.Equal(t, expected, result)
				case int32:
					result = ConvertValue[float64, int32](src)
					assert.Equal(t, expected, result)
				}
			case float32:
				switch expected := tt.expected.(type) {
				case int8:
					result = ConvertValue[float32, int8](src)
					assert.Equal(t, expected, result)
				case int16:
					result = ConvertValue[float32, int16](src)
					assert.Equal(t, expected, result)
				case int32:
					result = ConvertValue[float32, int32](src)
					assert.Equal(t, expected, result)
				}
			case int64:
				switch expected := tt.expected.(type) {
				case int8:
					result = ConvertValue[int64, int8](src)
					assert.Equal(t, expected, result)
				case int16:
					result = ConvertValue[int64, int16](src)
					assert.Equal(t, expected, result)
				case int32:
					result = ConvertValue[int64, int32](src)
					assert.Equal(t, expected, result)
				}
			case int32:
				switch expected := tt.expected.(type) {
				case int8:
					result = ConvertValue[int32, int8](src)
					assert.Equal(t, expected, result)
				case int16:
					result = ConvertValue[int32, int16](src)
					assert.Equal(t, expected, result)
				}
			case int16:
				switch expected := tt.expected.(type) {
				case int8:
					result = ConvertValue[int16, int8](src)
					assert.Equal(t, expected, result)
				}
			case int:
				switch expected := tt.expected.(type) {
				case int8:
					result = ConvertValue[int, int8](src)
					assert.Equal(t, expected, result)
				case int16:
					result = ConvertValue[int, int16](src)
					assert.Equal(t, expected, result)
				case int32:
					result = ConvertValue[int, int32](src)
					assert.Equal(t, expected, result)
				}
			}
			assert.NotNil(t, result)
		})
	}
}
