package generics

import (
	"testing"
)

var (
	scalarBenchSrc = make([]float32, 10000)
	scalarBenchDst = make([]float32, 10000)
	scalarValue    = float32(5.0)
)

func init() {
	for i := range scalarBenchSrc {
		scalarBenchSrc[i] = float32(i)
	}
}

// BenchmarkElemFill_Generic benchmarks generic fill (contiguous)
func BenchmarkElemFill_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemFill(scalarBenchDst, scalarValue, 10000)
	}
}

// BenchmarkElemFill_NonGeneric benchmarks non-generic fill
func BenchmarkElemFill_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fillNonGeneric(scalarBenchDst, scalarValue, 10000)
	}
}

// BenchmarkElemFill_DirectLoop benchmarks direct loop fill
func BenchmarkElemFill_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			scalarBenchDst[j] = scalarValue
		}
	}
}

// BenchmarkElemEqualScalar_Generic benchmarks generic equal scalar (contiguous)
func BenchmarkElemEqualScalar_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemEqualScalar(scalarBenchDst, scalarBenchSrc, scalarValue, 10000)
	}
}

// BenchmarkElemEqualScalar_NonGeneric benchmarks non-generic equal scalar
func BenchmarkElemEqualScalar_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		equalScalarNonGeneric(scalarBenchDst, scalarBenchSrc, scalarValue, 10000)
	}
}

// BenchmarkElemEqualScalar_DirectLoop benchmarks direct loop
func BenchmarkElemEqualScalar_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			if scalarBenchSrc[j] == scalarValue {
				scalarBenchDst[j] = 1
			} else {
				scalarBenchDst[j] = 0
			}
		}
	}
}

// BenchmarkElemGreaterScalar_Generic benchmarks generic greater scalar (contiguous)
func BenchmarkElemGreaterScalar_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemGreaterScalar(scalarBenchDst, scalarBenchSrc, scalarValue, 10000)
	}
}

// BenchmarkElemGreaterScalar_NonGeneric benchmarks non-generic greater scalar
func BenchmarkElemGreaterScalar_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		greaterScalarNonGeneric(scalarBenchDst, scalarBenchSrc, scalarValue, 10000)
	}
}

// BenchmarkElemGreaterScalar_DirectLoop benchmarks direct loop
func BenchmarkElemGreaterScalar_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			if scalarBenchSrc[j] > scalarValue {
				scalarBenchDst[j] = 1
			} else {
				scalarBenchDst[j] = 0
			}
		}
	}
}

// Non-generic helper functions
func fillNonGeneric(dst []float32, value float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i] = value
	}
}

func equalScalarNonGeneric(dst, src []float32, scalar float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] == scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

func greaterScalarNonGeneric(dst, src []float32, scalar float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] > scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}
