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

func BenchmarkHadamardProduct(b *testing.B) {
	num := 5000
	a := make([]float32, num*2)
	bvec := make([]float32, num*2)
	dst := make([]float32, num)

	for i := range a {
		a[i] = float32(i)
		bvec[i] = float32(i * 2)
	}

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			HadamardProduct(dst, a, bvec, num, 1, 1, 1)
		}
	})

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			HadamardProduct(dst, a, bvec, num, 1, 2, 2)
		}
	})
}

func BenchmarkConvolve1D(b *testing.B) {
	N := 1000
	M := 10
	stride := 1
	vec := make([]float32, N)
	kernel := make([]float32, M)
	dstSize := (N - M) / stride
	if (N-M)%stride != 0 {
		dstSize++
	}
	dst := make([]float32, dstSize)

	for i := range vec {
		vec[i] = float32(i)
	}
	for i := range kernel {
		kernel[i] = float32(i)
	}

	b.Run("forward", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convolve1D(dst, vec, kernel, N, M, stride, false)
		}
	})

	b.Run("transposed", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convolve1D(dst, vec, kernel, N, M, stride, true)
		}
	})
}

func BenchmarkNormalizeVec(b *testing.B) {
	num := 500
	src := make([]float32, num*2)
	dst := make([]float32, num)

	for i := range src {
		src[i] = float32(i + 1)
	}

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			NormalizeVec(dst, src, num, 1, 1)
		}
	})

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			NormalizeVec(dst, src, num, 1, 2)
		}
	})
}

func BenchmarkSumArrScalar(b *testing.B) {
	num := 5000
	src := make([]float32, num*2)
	dst := make([]float32, num)
	c := float32(10.5)

	for i := range src {
		src[i] = float32(i)
	}

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			SumArrScalar(dst, src, c, num, 1, 1)
		}
	})

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			SumArrScalar(dst, src, c, num, 1, 2)
		}
	})
}

func BenchmarkDiffArrScalar(b *testing.B) {
	num := 5000
	src := make([]float32, num*2)
	dst := make([]float32, num)
	c := float32(10.5)

	for i := range src {
		src[i] = float32(i)
	}

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			DiffArrScalar(dst, src, c, num, 1, 1)
		}
	})

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			DiffArrScalar(dst, src, c, num, 1, 2)
		}
	})
}

func BenchmarkElemScale(b *testing.B) {
	shape := []int{128, 256}
	contiguousStrides := ComputeStrides(shape)
	size := shape[0] * shape[1]
	dst := make([]float32, size)
	src := make([]float32, size)
	scalar := float32(2.5)

	for i := range src {
		src[i] = float32(i)
	}

	b.Run("contiguous", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ElemScale(dst, src, scalar, shape, contiguousStrides, contiguousStrides)
		}
	})

	rowStride := shape[1] * 2
	stridedShape := []int{shape[0], shape[1]}
	strides := []int{rowStride, 1}
	stridedDst := make([]float32, shape[0]*rowStride)
	stridedSrc := make([]float32, shape[0]*rowStride)

	b.Run("strided", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ElemScale(stridedDst, stridedSrc, scalar, stridedShape, strides, strides)
		}
	})
}
