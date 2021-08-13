package mat

import (
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

func BenchmarkCloneMatrix2(b *testing.B) {
	v := New(2, 2, 1, 2, 3, 4)
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneMatrix3(b *testing.B) {
	v := New(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneMatrix4(b *testing.B) {
	v := New(4, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneMatrix2x2(b *testing.B) {
	v := Matrix2x2{}
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneMatrix3x3(b *testing.B) {
	v := Matrix3x3{}
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneMatrix4x4(b *testing.B) {
	v := Matrix4x4{}
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkMatrixMatrix4x4(b *testing.B) {
	v := Matrix4x4{}
	for i := 0; i < b.N; i++ {
		_ = v.Matrix()
	}
}

func BenchmarkMul4(b *testing.B) {
	va := New(4, 4)
	vb := New(4, 4)
	dst := New(4, 4)
	va.Eye()
	vb.Eye()
	for i := 0; i < b.N; i++ {
		_ = dst.Mul(va, vb)
	}
}

func BenchmarkMul4x4(b *testing.B) {
	va := Matrix4x4{}
	vb := Matrix4x4{}
	dst := Matrix4x4{}
	va.Eye()
	vb.Eye()
	for i := 0; i < b.N; i++ {
		_ = dst.Mul(va, vb)
	}
}

func BenchmarkMulV(b *testing.B) {
	va := New(4, 4)
	vb := vec.New(4)
	dst := vec.New(4)
	va.Eye()
	for i := 0; i < b.N; i++ {
		_ = va.MulVec(vb, dst)
	}
}

func BenchmarkMulV4x4(b *testing.B) {
	va := Matrix4x4{}
	vb := vec.Vector4D{}
	dst := vec.Vector4D{}
	va.Eye()
	for i := 0; i < b.N; i++ {
		_ = va.MulVec(vb, dst.Vector())
	}
}
