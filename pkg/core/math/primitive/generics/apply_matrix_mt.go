//go:build use_mt

package generics

import (
	mt "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/mt"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

// Re-export matrix apply functions from multi-threaded implementation
func ElemMatApplyUnaryStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T) {
	mt.ElemMatApplyUnaryStrided(dst, src, rows, cols, ldDst, ldSrc, op)
}

func ElemMatApplyBinaryStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int, op func(T, T) T) {
	mt.ElemMatApplyBinaryStrided(dst, a, b, rows, cols, ldDst, ldA, ldB, op)
}

func ElemMatApplyTernaryStrided[T Numeric](dst, condition, a, b []T, rows, cols int, ldDst, ldCond, ldA, ldB int, op func(T, T, T) T) {
	mt.ElemMatApplyTernaryStrided(dst, condition, a, b, rows, cols, ldDst, ldCond, ldA, ldB, op)
}

func ElemMatApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, rows, cols int, ldDst, ldSrc int, op func(T, T) T) {
	mt.ElemMatApplyUnaryScalarStrided(dst, src, scalar, rows, cols, ldDst, ldSrc, op)
}

func ElemMatApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, rows, cols int, ldDst, ldA int, op func(T, T) T) {
	mt.ElemMatApplyBinaryScalarStrided(dst, a, scalar, rows, cols, ldDst, ldA, op)
}

func ElemMatApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, rows, cols int, ldDst, ldCond, ldA int, op func(T, T, T) T) {
	mt.ElemMatApplyTernaryScalarStrided(dst, condition, a, scalar, rows, cols, ldDst, ldCond, ldA, op)
}

