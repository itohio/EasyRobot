//go:build !use_mt

package iter

import (
	st "github.com/itohio/EasyRobot/x/iter/st"
	"github.com/itohio/EasyRobot/x/types"
)

// Re-export vector apply functions from single-threaded implementation
func ElemVecApplyUnaryStrided[T types.Number](dst, src []T, n int, strideDst, strideSrc int, op func(T) T) {
	st.ElemVecApplyUnaryStrided(dst, src, n, strideDst, strideSrc, op)
}

func ElemVecApplyBinaryStrided[T types.Number](dst, a, b []T, n int, strideDst, strideA, strideB int, op func(T, T) T) {
	st.ElemVecApplyBinaryStrided(dst, a, b, n, strideDst, strideA, strideB, op)
}

func ElemVecApplyTernaryStrided[T types.Number](dst, condition, a, b []T, n int, strideDst, strideCond, strideA, strideB int, op func(T, T, T) T) {
	st.ElemVecApplyTernaryStrided(dst, condition, a, b, n, strideDst, strideCond, strideA, strideB, op)
}

func ElemVecApplyUnaryScalarStrided[T types.Number](dst, src []T, scalar T, n int, strideDst, strideSrc int, op func(T, T) T) {
	st.ElemVecApplyUnaryScalarStrided(dst, src, scalar, n, strideDst, strideSrc, op)
}

func ElemVecApplyBinaryScalarStrided[T types.Number](dst, a []T, scalar T, n int, strideDst, strideA int, op func(T, T) T) {
	st.ElemVecApplyBinaryScalarStrided(dst, a, scalar, n, strideDst, strideA, op)
}

func ElemVecApplyTernaryScalarStrided[T types.Number](dst, condition, a []T, scalar T, n int, strideDst, strideCond, strideA int, op func(T, T, T) T) {
	st.ElemVecApplyTernaryScalarStrided(dst, condition, a, scalar, n, strideDst, strideCond, strideA, op)
}
