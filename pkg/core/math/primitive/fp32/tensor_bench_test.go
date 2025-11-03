package fp32

import "testing"

func BenchmarkElemAdd(b *testing.B) {
	shape := []int{128, 256}
	contiguousStrides := ComputeStrides(shape)
	size := shape[0] * shape[1]
	dst := make([]float32, size)
	a := make([]float32, size)
	bvec := make([]float32, size)

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ElemAdd(dst, a, bvec, shape, contiguousStrides, contiguousStrides, contiguousStrides)
		}
	})

	rowStride := shape[1] * 2
	stridedShape := []int{shape[0], shape[1]}
	strides := []int{rowStride, 1}
	stridedDst := make([]float32, shape[0]*rowStride)
	stridedA := make([]float32, shape[0]*rowStride)
	stridedB := make([]float32, shape[0]*rowStride)

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ElemAdd(stridedDst, stridedA, stridedB, stridedShape, strides, strides, strides)
		}
	})
}

func BenchmarkReduceSum(b *testing.B) {
	shape := []int{256, 128}
	contiguousStrides := ComputeStrides(shape)
	size := shape[0] * shape[1]
	src := make([]float32, size)
	dstShape := []int{shape[0]}
	dstStrides := ComputeStrides(dstShape)
	dst := make([]float32, dstShape[0])
	axes := []int{1}

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ReduceSum(dst, dstShape, dstStrides, src, shape, contiguousStrides, axes)
		}
	})

	rowStride := shape[1] * 2
	strides := []int{rowStride, 1}
	stridedSrc := make([]float32, shape[0]*rowStride)

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ReduceSum(dst, dstShape, dstStrides, stridedSrc, shape, strides, axes)
		}
	})
}
