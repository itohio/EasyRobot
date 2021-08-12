package vec

import "testing"

func BenchmarkCloneVector2(b *testing.B) {
	v := NewFrom(1, 2)
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneVector3(b *testing.B) {
	v := NewFrom(1, 2, 3)
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneVector4(b *testing.B) {
	v := NewFrom(1, 2, 3, 4)
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneVector2D(b *testing.B) {
	v := Vector2D{}
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneVector3D(b *testing.B) {
	v := Vector3D{}
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkCloneVector4D(b *testing.B) {
	v := Vector3D{}
	for i := 0; i < b.N; i++ {
		_ = v.Clone()
	}
}

func BenchmarkDot4(b *testing.B) {
	va := NewFrom(1, 2, 3, 4)
	vb := NewFrom(1, 2, 3, 4)
	for i := 0; i < b.N; i++ {
		_ = va.Dot(vb)
	}
}

func BenchmarkDot4D(b *testing.B) {
	va := Vector4D{1, 2, 3, 4}
	vb := Vector4D{1, 2, 3, 4}
	for i := 0; i < b.N; i++ {
		_ = va.Dot(vb)
	}
}

func BenchmarkCross(b *testing.B) {
	va := NewFrom(1, 2, 3)
	vb := NewFrom(1, 2, 3)
	for i := 0; i < b.N; i++ {
		_ = va.Cross(vb)
	}
}

func BenchmarkCross3D(b *testing.B) {
	va := Vector3D{1, 2, 3}
	vb := Vector3D{1, 2, 3}
	for i := 0; i < b.N; i++ {
		_ = va.Cross(vb)
	}
}
