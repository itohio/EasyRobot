package primitive

import (
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
		dst := []int{1, 2, 3} // Wrong type (not []int8, []int16, []float32, []float64)

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
