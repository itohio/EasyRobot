package primitive

import (
	"testing"
)

func BenchmarkCopyWithConversion(b *testing.B) {
	b.Run("float32_to_float32_same_type", func(b *testing.B) {
		src := make([]float32, 1000)
		dst := make([]float32, 1000)
		for i := range src {
			src[i] = float32(i)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst, src)
		}
	})

	b.Run("float64_to_float32", func(b *testing.B) {
		src := make([]float64, 1000)
		dst := make([]float32, 1000)
		for i := range src {
			src[i] = float64(i)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst, src)
		}
	})

	b.Run("float32_to_float64", func(b *testing.B) {
		src := make([]float32, 1000)
		dst := make([]float64, 1000)
		for i := range src {
			src[i] = float32(i)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst, src)
		}
	})

	b.Run("int16_to_int8_with_clamping", func(b *testing.B) {
		src := make([]int16, 1000)
		dst := make([]int8, 1000)
		for i := range src {
			src[i] = int16(i * 100) // Many will need clamping
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst, src)
		}
	})

	b.Run("float32_to_int8_with_clamping", func(b *testing.B) {
		src := make([]float32, 1000)
		dst := make([]int8, 1000)
		for i := range src {
			src[i] = float32(i * 100) // Many will need clamping
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst, src)
		}
	})

	b.Run("int8_to_int16", func(b *testing.B) {
		src := make([]int8, 1000)
		dst := make([]int16, 1000)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst, src)
		}
	})
}

func BenchmarkCopyWithStrides(b *testing.B) {
	b.Run("float32_same_type_contiguous", func(b *testing.B) {
		src := make([]float32, 1000)
		dst := make([]float32, 1000)
		for i := range src {
			src[i] = float32(i)
		}
		shape := []int{1000}
		srcStrides := []int{1}
		dstStrides := []int{1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("float32_same_type_strided", func(b *testing.B) {
		src := make([]float32, 2000)
		dst := make([]float32, 2000)
		for i := range src {
			src[i] = float32(i)
		}
		shape := []int{10, 100}
		srcStrides := []int{100, 1}
		dstStrides := []int{200, 2} // Non-contiguous destination
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("float64_to_float32_with_strides", func(b *testing.B) {
		src := make([]float64, 1000)
		dst := make([]float32, 1000)
		for i := range src {
			src[i] = float64(i)
		}
		shape := []int{1000}
		srcStrides := []int{1}
		dstStrides := []int{1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("float32_to_float64_with_strides", func(b *testing.B) {
		src := make([]float32, 1000)
		dst := make([]float64, 1000)
		for i := range src {
			src[i] = float32(i)
		}
		shape := []int{1000}
		srcStrides := []int{1}
		dstStrides := []int{1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("int16_to_int8_with_clamping_strided", func(b *testing.B) {
		src := make([]int16, 1000)
		dst := make([]int8, 1000)
		for i := range src {
			src[i] = int16(i * 100) // Many will need clamping
		}
		shape := []int{1000}
		srcStrides := []int{1}
		dstStrides := []int{1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("float32_to_int8_with_clamping_strided", func(b *testing.B) {
		src := make([]float32, 1000)
		dst := make([]int8, 1000)
		for i := range src {
			src[i] = float32(i * 100) // Many will need clamping
		}
		shape := []int{1000}
		srcStrides := []int{1}
		dstStrides := []int{1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("int8_to_int16_with_strides", func(b *testing.B) {
		src := make([]int8, 1000)
		dst := make([]int16, 1000)
		for i := range src {
			src[i] = int8(i % 128)
		}
		shape := []int{1000}
		srcStrides := []int{1}
		dstStrides := []int{1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})

	b.Run("2d_float32_to_float64", func(b *testing.B) {
		src := make([]float32, 10000)
		dst := make([]float64, 10000)
		for i := range src {
			src[i] = float32(i)
		}
		shape := []int{100, 100}
		srcStrides := []int{100, 1}
		dstStrides := []int{100, 1}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithStrides(src, dst, shape, srcStrides, dstStrides)
		}
	})
}
