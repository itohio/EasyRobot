package helpers

func FromComplexToT[T NumericReal](dst []T, src []complex64) {
	n := min(len(src), len(dst))
	for i := 0; i < n; i++ {
		dst[i] = T(real(src[i]))
	}
}

func FromTToComplex[T NumericReal](dst []complex64, src []T) {
	n := min(len(src), len(dst))
	for i := 0; i < n; i++ {
		dst[i] = complex(float32(src[i]), 0)
	}
}

func FromUToT[T NumericReal, U NumericReal](dst []T, src []U) {
	n := min(len(src), len(dst))
	for i := 0; i < n; i++ {
		dst[i] = T(src[i])
	}
}
