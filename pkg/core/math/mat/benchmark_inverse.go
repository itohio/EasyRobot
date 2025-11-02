package mat

import "testing"

func BenchmarkMatrix_Inverse_3x3(b *testing.B) {
	m := New(3, 3)
	m.Eye()
	dst := New(3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Inverse(dst)
	}
}

func BenchmarkMatrix_Inverse_4x4(b *testing.B) {
	m := New(4, 4)
	m.Eye()
	dst := New(4, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Inverse(dst)
	}
}

func BenchmarkMatrix2x2_Inverse(b *testing.B) {
	m := Matrix2x2{}
	m.Eye()
	var dst Matrix2x2
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Inverse(&dst)
	}
}

func BenchmarkMatrix3x3_Inverse(b *testing.B) {
	m := Matrix3x3{}
	m.Eye()
	var dst Matrix3x3
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Inverse(&dst)
	}
}

func BenchmarkMatrix4x4_Inverse(b *testing.B) {
	m := Matrix4x4{}
	m.Eye()
	var dst Matrix4x4
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Inverse(&dst)
	}
}

func BenchmarkMatrix_PseudoInverse_3x2(b *testing.B) {
	m := New(3, 2, 1, 0, 0, 1, 1, 1)
	dst := New(2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.PseudoInverse(dst)
	}
}

func BenchmarkMatrix_PseudoInverse_6x3(b *testing.B) {
	// Common Jacobian size for 6 DOF robot
	m := New(6, 3,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	)
	dst := New(3, 6)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.PseudoInverse(dst)
	}
}

func BenchmarkMatrix_DampedLeastSquares_6x3(b *testing.B) {
	// Common Jacobian size for 6 DOF robot
	m := New(6, 3,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	)
	dst := New(3, 6)
	lambda := float32(0.1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.DampedLeastSquares(lambda, dst)
	}
}

func BenchmarkCalculateJacobianColumn_Revolute(b *testing.B) {
	jointPos := [3]float32{0, 0, 0}
	jointAxis := [3]float32{0, 0, 1}
	eePos := [3]float32{1, 0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalculateJacobianColumn(jointPos, jointAxis, eePos, true)
	}
}

func BenchmarkCalculateJacobianColumn_Prismatic(b *testing.B) {
	jointPos := [3]float32{0, 0, 0}
	jointAxis := [3]float32{1, 0, 0}
	eePos := [3]float32{1, 2, 3}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalculateJacobianColumn(jointPos, jointAxis, eePos, false)
	}
}
