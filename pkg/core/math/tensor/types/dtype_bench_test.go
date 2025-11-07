package types

import (
	"testing"
)

var (
	benchFloat32Slice = make([]float32, 10000)
	benchFloat64Slice = make([]float64, 10000)
	benchIntSlice     = make([]int, 10000)
	benchInt32Slice   = make([]int32, 10000)
	benchInt64Slice   = make([]int64, 10000)
	benchInt16Slice   = make([]int16, 10000)
	benchInt8Slice    = make([]int8, 10000)
)

func init() {
	for i := range benchFloat32Slice {
		benchFloat32Slice[i] = float32(i)
	}
	for i := range benchFloat64Slice {
		benchFloat64Slice[i] = float64(i)
	}
	for i := range benchIntSlice {
		benchIntSlice[i] = i
	}
	for i := range benchInt32Slice {
		benchInt32Slice[i] = int32(i)
	}
	for i := range benchInt64Slice {
		benchInt64Slice[i] = int64(i)
	}
	for i := range benchInt16Slice {
		benchInt16Slice[i] = int16(i)
	}
	for i := range benchInt8Slice {
		benchInt8Slice[i] = int8(i)
	}
}

// BenchmarkTypeFromData benchmarks TypeFromData with various types
func BenchmarkTypeFromData_Float32Slice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchFloat32Slice)
	}
}

func BenchmarkTypeFromData_Float64Slice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchFloat64Slice)
	}
}

func BenchmarkTypeFromData_IntSlice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchIntSlice)
	}
}

func BenchmarkTypeFromData_Int32Slice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchInt32Slice)
	}
}

func BenchmarkTypeFromData_Int64Slice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchInt64Slice)
	}
}

func BenchmarkTypeFromData_Int16Slice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchInt16Slice)
	}
}

func BenchmarkTypeFromData_Int8Slice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(benchInt8Slice)
	}
}

func BenchmarkTypeFromData_Float32Scalar(b *testing.B) {
	val := float32(42.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(val)
	}
}

func BenchmarkTypeFromData_IntScalar(b *testing.B) {
	val := int(42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TypeFromData(val)
	}
}

// BenchmarkMakeTensorData benchmarks MakeTensorData with various data types
func BenchmarkMakeTensorData_FP32(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(FP32, size)
	}
}

func BenchmarkMakeTensorData_FP64(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(FP64, size)
	}
}

func BenchmarkMakeTensorData_INT(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(INT, size)
	}
}

func BenchmarkMakeTensorData_INT32(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(INT32, size)
	}
}

func BenchmarkMakeTensorData_INT64(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(INT64, size)
	}
}

func BenchmarkMakeTensorData_INT16(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(INT16, size)
	}
}

func BenchmarkMakeTensorData_INT8(b *testing.B) {
	size := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(INT8, size)
	}
}

func BenchmarkMakeTensorData_Small(b *testing.B) {
	size := 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(FP32, size)
	}
}

func BenchmarkMakeTensorData_Large(b *testing.B) {
	size := 1000000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MakeTensorData(FP32, size)
	}
}

// BenchmarkCloneTensorData benchmarks CloneTensorData
func BenchmarkCloneTensorData_Float32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchFloat32Slice)
	}
}

func BenchmarkCloneTensorData_Float64(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchFloat64Slice)
	}
}

func BenchmarkCloneTensorData_Int(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchIntSlice)
	}
}

func BenchmarkCloneTensorData_Int32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchInt32Slice)
	}
}

func BenchmarkCloneTensorData_Int64(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchInt64Slice)
	}
}

func BenchmarkCloneTensorData_Int16(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchInt16Slice)
	}
}

func BenchmarkCloneTensorData_Int8(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorData(benchInt8Slice)
	}
}

// BenchmarkCloneTensorDataTo benchmarks CloneTensorDataTo with various conversions
func BenchmarkCloneTensorDataTo_Float32ToFloat32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(FP32, benchFloat32Slice)
	}
}

func BenchmarkCloneTensorDataTo_Float64ToFloat32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(FP32, benchFloat64Slice)
	}
}

func BenchmarkCloneTensorDataTo_Float32ToFloat64(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(FP64, benchFloat32Slice)
	}
}

func BenchmarkCloneTensorDataTo_Int32ToFloat32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(FP32, benchInt32Slice)
	}
}

func BenchmarkCloneTensorDataTo_Int64ToFloat32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(FP32, benchInt64Slice)
	}
}

func BenchmarkCloneTensorDataTo_Float32ToInt32(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(INT32, benchFloat32Slice)
	}
}

func BenchmarkCloneTensorDataTo_Int16ToInt8(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(INT8, benchInt16Slice)
	}
}

func BenchmarkCloneTensorDataTo_Int8ToInt16(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CloneTensorDataTo(INT16, benchInt8Slice)
	}
}

