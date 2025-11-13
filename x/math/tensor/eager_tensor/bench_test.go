package eager_tensor

import (
	"math/rand"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Benchmark tensor sizes
var (
	benchSizes = []struct {
		name  string
		size  int
		shape types.Shape
	}{
		{"1K", 1000, types.NewShape(1000)},
		{"10K", 10000, types.NewShape(10000)},
		{"100K", 100000, types.NewShape(100000)},
		{"1M", 1000000, types.NewShape(1000000)},
		{"2D_100x100", 10000, types.NewShape(100, 100)},
		{"2D_1000x100", 100000, types.NewShape(1000, 100)},
		{"3D_50x50x50", 125000, types.NewShape(50, 50, 50)},
	}
)

// makeBenchTensor creates a tensor filled with test data
func makeBenchTensor(shape types.Shape) Tensor {
	size := shape.Size()
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i%100)/10.0 + 0.1 // Range: 0.1 to 10.0
	}
	return FromFloat32(shape, data)
}

// makeBenchTensorOther creates another tensor with different test data
func makeBenchTensorOther(shape types.Shape) Tensor {
	size := shape.Size()
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i%50)/5.0 + 0.2 // Range: 0.2 to 10.2
	}
	return FromFloat32(shape, data)
}

// Benchmark Add operations
func BenchmarkTensorAdd_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Add(nil, t2)
			}
		})
	}
}

func BenchmarkTensorAdd_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Add(dst, t2)
			}
		})
	}
}

// Benchmark Subtract operations
func BenchmarkTensorSubtract_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Subtract(nil, t2)
			}
		})
	}
}

func BenchmarkTensorSubtract_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Subtract(dst, t2)
			}
		})
	}
}

// Benchmark Multiply operations
func BenchmarkTensorMultiply_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Multiply(nil, t2)
			}
		})
	}
}

func BenchmarkTensorMultiply_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Multiply(dst, t2)
			}
		})
	}
}

// Benchmark Divide operations
func BenchmarkTensorDivide_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			// Ensure no division by zero
			data2 := t2.Data().([]float32)
			for i := range data2 {
				if data2[i] < 0.1 {
					data2[i] = 0.1
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Divide(nil, t2)
			}
		})
	}
}

func BenchmarkTensorDivide_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			// Ensure no division by zero
			data2 := t2.Data().([]float32)
			for i := range data2 {
				if data2[i] < 0.1 {
					data2[i] = 0.1
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Divide(dst, t2)
			}
		})
	}
}

// Benchmark ScalarMul operations
func BenchmarkTensorScalarMul_InPlace(b *testing.B) {
	scalar := 2.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ScalarMul(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorScalarMul_Destination(b *testing.B) {
	scalar := 2.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ScalarMul(dst, scalar)
			}
		})
	}
}

// Benchmark AddScalar operations
func BenchmarkTensorAddScalar_InPlace(b *testing.B) {
	scalar := 1.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.AddScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorAddScalar_Destination(b *testing.B) {
	scalar := 1.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.AddScalar(dst, scalar)
			}
		})
	}
}

// Benchmark SubScalar operations
func BenchmarkTensorSubScalar_InPlace(b *testing.B) {
	scalar := 1.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.SubScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorSubScalar_Destination(b *testing.B) {
	scalar := 1.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.SubScalar(dst, scalar)
			}
		})
	}
}

// Benchmark MulScalar operations
func BenchmarkTensorMulScalar_InPlace(b *testing.B) {
	scalar := 2.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.MulScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorMulScalar_Destination(b *testing.B) {
	scalar := 2.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.MulScalar(dst, scalar)
			}
		})
	}
}

// Benchmark DivScalar operations
func BenchmarkTensorDivScalar_InPlace(b *testing.B) {
	scalar := 2.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.DivScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorDivScalar_Destination(b *testing.B) {
	scalar := 2.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.DivScalar(dst, scalar)
			}
		})
	}
}

// Benchmark Square operations
func BenchmarkTensorSquare_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Square(nil)
			}
		})
	}
}

func BenchmarkTensorSquare_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Square(dst)
			}
		})
	}
}

// Benchmark Sqrt operations
func BenchmarkTensorSqrt_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Ensure all values are positive
			for i := range t.Data().([]float32) {
				if t.Data().([]float32)[i] < 0 {
					t.Data().([]float32)[i] = -t.Data().([]float32)[i]
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sqrt(nil)
			}
		})
	}
}

func BenchmarkTensorSqrt_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Ensure all values are positive
			for i := range t.Data().([]float32) {
				if t.Data().([]float32)[i] < 0 {
					t.Data().([]float32)[i] = -t.Data().([]float32)[i]
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sqrt(dst)
			}
		})
	}
}

// Benchmark Exp operations
func BenchmarkTensorExp_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Clamp values to avoid overflow
			for i := range t.Data().([]float32) {
				if t.Data().([]float32)[i] > 10 {
					t.Data().([]float32)[i] = 10
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Exp(nil)
			}
		})
	}
}

func BenchmarkTensorExp_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Clamp values to avoid overflow
			for i := range t.Data().([]float32) {
				if t.Data().([]float32)[i] > 10 {
					t.Data().([]float32)[i] = 10
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Exp(dst)
			}
		})
	}
}

// Benchmark Log operations
func BenchmarkTensorLog_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Ensure all values are positive
			for i := range t.Data().([]float32) {
				if t.Data().([]float32)[i] <= 0 {
					t.Data().([]float32)[i] = 0.1
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Log(nil)
			}
		})
	}
}

func BenchmarkTensorLog_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Ensure all values are positive
			for i := range t.Data().([]float32) {
				if t.Data().([]float32)[i] <= 0 {
					t.Data().([]float32)[i] = 0.1
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Log(dst)
			}
		})
	}
}

// Benchmark Pow operations
func BenchmarkTensorPow_InPlace(b *testing.B) {
	power := 2.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Pow(nil, power)
			}
		})
	}
}

func BenchmarkTensorPow_Destination(b *testing.B) {
	power := 2.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Pow(dst, power)
			}
		})
	}
}

// Benchmark Abs operations
func BenchmarkTensorAbs_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Abs(nil)
			}
		})
	}
}

func BenchmarkTensorAbs_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Abs(dst)
			}
		})
	}
}

// Benchmark Sign operations
func BenchmarkTensorSign_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive, negative, and zero values
			data := t.Data().([]float32)
			for i := range data {
				switch i % 3 {
				case 0:
					data[i] = -data[i]
				case 1:
					data[i] = 0
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sign(nil)
			}
		})
	}
}

func BenchmarkTensorSign_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive, negative, and zero values
			data := t.Data().([]float32)
			for i := range data {
				switch i % 3 {
				case 0:
					data[i] = -data[i]
				case 1:
					data[i] = 0
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sign(dst)
			}
		})
	}
}

// Benchmark Cos operations
func BenchmarkTensorCos_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Cos(nil)
			}
		})
	}
}

func BenchmarkTensorCos_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Cos(dst)
			}
		})
	}
}

// Benchmark Sin operations
func BenchmarkTensorSin_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sin(nil)
			}
		})
	}
}

func BenchmarkTensorSin_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sin(dst)
			}
		})
	}
}

// Benchmark Negative operations
func BenchmarkTensorNegative_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Negative(nil)
			}
		})
	}
}

func BenchmarkTensorNegative_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Negative(dst)
			}
		})
	}
}

// Benchmark ReLU operations
func BenchmarkTensorReLU_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ReLU(nil)
			}
		})
	}
}

func BenchmarkTensorReLU_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ReLU(dst)
			}
		})
	}
}

// Benchmark Sigmoid operations
func BenchmarkTensorSigmoid_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sigmoid(nil)
			}
		})
	}
}

func BenchmarkTensorSigmoid_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sigmoid(dst)
			}
		})
	}
}

// Benchmark Tanh operations
func BenchmarkTensorTanh_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Tanh(nil)
			}
		})
	}
}

func BenchmarkTensorTanh_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Tanh(dst)
			}
		})
	}
}

// Benchmark Fill operations
func BenchmarkTensorFill_InPlace(b *testing.B) {
	value := 3.14
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Fill(nil, value)
			}
		})
	}
}

func BenchmarkTensorFill_Destination(b *testing.B) {
	value := 3.14
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Fill(dst, value)
			}
		})
	}
}

// Benchmark Comparison Operations
func BenchmarkTensorEqual_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Equal(nil, t2)
			}
		})
	}
}

func BenchmarkTensorEqual_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Equal(dst, t2)
			}
		})
	}
}

func BenchmarkTensorGreater_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Greater(nil, t2)
			}
		})
	}
}

func BenchmarkTensorGreater_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Greater(dst, t2)
			}
		})
	}
}

func BenchmarkTensorLess_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Less(nil, t2)
			}
		})
	}
}

func BenchmarkTensorLess_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Less(dst, t2)
			}
		})
	}
}

func BenchmarkTensorGreaterScalar_InPlace(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.GreaterScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorGreaterScalar_Destination(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.GreaterScalar(dst, scalar)
			}
		})
	}
}

func BenchmarkTensorEqualScalar_InPlace(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.EqualScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorEqualScalar_Destination(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.EqualScalar(dst, scalar)
			}
		})
	}
}

func BenchmarkTensorNotEqual_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.NotEqual(nil, t2)
			}
		})
	}
}

func BenchmarkTensorNotEqual_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.NotEqual(dst, t2)
			}
		})
	}
}

func BenchmarkTensorGreaterEqual_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.GreaterEqual(nil, t2)
			}
		})
	}
}

func BenchmarkTensorGreaterEqual_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.GreaterEqual(dst, t2)
			}
		})
	}
}

func BenchmarkTensorLessEqual_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.LessEqual(nil, t2)
			}
		})
	}
}

func BenchmarkTensorLessEqual_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.LessEqual(dst, t2)
			}
		})
	}
}

func BenchmarkTensorNotEqualScalar_InPlace(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.NotEqualScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorNotEqualScalar_Destination(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.NotEqualScalar(dst, scalar)
			}
		})
	}
}

func BenchmarkTensorLessScalar_InPlace(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.LessScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorLessScalar_Destination(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.LessScalar(dst, scalar)
			}
		})
	}
}

func BenchmarkTensorGreaterEqualScalar_InPlace(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.GreaterEqualScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorGreaterEqualScalar_Destination(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.GreaterEqualScalar(dst, scalar)
			}
		})
	}
}

func BenchmarkTensorLessEqualScalar_InPlace(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.LessEqualScalar(nil, scalar)
			}
		})
	}
}

func BenchmarkTensorLessEqualScalar_Destination(b *testing.B) {
	scalar := 5.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.LessEqualScalar(dst, scalar)
			}
		})
	}
}

// Benchmark Conditional Operations
func BenchmarkTensorWhere_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			condition := makeBenchTensor(size.shape)
			// Make condition have both true and false values
			data := condition.Data().([]float32)
			for i := range data {
				data[i] = float32(i % 2) // 0 or 1
			}
			a := makeBenchTensor(size.shape)
			bTensor := makeBenchTensorOther(size.shape)
			// Use condition as receiver (tensor used for shape matching)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				condition.Where(nil, condition, a, bTensor)
			}
		})
	}
}

func BenchmarkTensorWhere_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			condition := makeBenchTensor(size.shape)
			// Make condition have both true and false values
			data := condition.Data().([]float32)
			for i := range data {
				data[i] = float32(i % 2) // 0 or 1
			}
			a := makeBenchTensor(size.shape)
			bTensor := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			// Use condition as receiver (tensor used for shape matching)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				condition.Where(dst, condition, a, bTensor)
			}
		})
	}
}

// Benchmark Reduction Operations
var reductionBenchSizes = []struct {
	name  string
	shape types.Shape
	dims  []int
}{
	{"1D_1K", types.NewShape(1000), []int{0}},
	{"1D_10K", types.NewShape(10000), []int{0}},
	{"2D_100x100_dim0", types.NewShape(100, 100), []int{0}},
	{"2D_100x100_dim1", types.NewShape(100, 100), []int{1}},
	{"2D_100x100_all", types.NewShape(100, 100), nil},
	{"3D_50x50x50_dim0", types.NewShape(50, 50, 50), []int{0}},
	{"3D_50x50x50_dim1", types.NewShape(50, 50, 50), []int{1}},
	{"3D_50x50x50_dim2", types.NewShape(50, 50, 50), []int{2}},
}

func BenchmarkTensorSum_InPlace(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sum(nil, size.dims)
			}
		})
	}
}

func BenchmarkTensorSum_Destination(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.Sum(nil, size.dims)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Sum(dst, size.dims)
			}
		})
	}
}

func BenchmarkTensorMean_InPlace(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Mean(nil, size.dims)
			}
		})
	}
}

func BenchmarkTensorMean_Destination(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.Mean(nil, size.dims)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Mean(dst, size.dims)
			}
		})
	}
}

func BenchmarkTensorMax_InPlace(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Max(nil, size.dims)
			}
		})
	}
}

func BenchmarkTensorMax_Destination(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.Max(nil, size.dims)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Max(dst, size.dims)
			}
		})
	}
}

func BenchmarkTensorMin_InPlace(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Min(nil, size.dims)
			}
		})
	}
}

func BenchmarkTensorMin_Destination(b *testing.B) {
	for _, size := range reductionBenchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.Min(nil, size.dims)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Min(dst, size.dims)
			}
		})
	}
}

func BenchmarkTensorArgMax_InPlace(b *testing.B) {
	argmaxSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"2D_100x100_dim0", types.NewShape(100, 100), 0},
		{"2D_100x100_dim1", types.NewShape(100, 100), 1},
		{"3D_50x50x50_dim0", types.NewShape(50, 50, 50), 0},
		{"3D_50x50x50_dim1", types.NewShape(50, 50, 50), 1},
		{"3D_50x50x50_dim2", types.NewShape(50, 50, 50), 2},
	}
	for _, size := range argmaxSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ArgMax(nil, size.dim)
			}
		})
	}
}

func BenchmarkTensorArgMax_Destination(b *testing.B) {
	argmaxSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"2D_100x100_dim0", types.NewShape(100, 100), 0},
		{"2D_100x100_dim1", types.NewShape(100, 100), 1},
		{"3D_50x50x50_dim0", types.NewShape(50, 50, 50), 0},
		{"3D_50x50x50_dim1", types.NewShape(50, 50, 50), 1},
		{"3D_50x50x50_dim2", types.NewShape(50, 50, 50), 2},
	}
	for _, size := range argmaxSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.ArgMax(nil, size.dim)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ArgMax(dst, size.dim)
			}
		})
	}
}

func BenchmarkTensorArgMin_InPlace(b *testing.B) {
	argminSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"2D_100x100_dim0", types.NewShape(100, 100), 0},
		{"2D_100x100_dim1", types.NewShape(100, 100), 1},
		{"3D_50x50x50_dim0", types.NewShape(50, 50, 50), 0},
		{"3D_50x50x50_dim1", types.NewShape(50, 50, 50), 1},
		{"3D_50x50x50_dim2", types.NewShape(50, 50, 50), 2},
	}
	for _, size := range argminSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ArgMin(nil, size.dim)
			}
		})
	}
}

func BenchmarkTensorArgMin_Destination(b *testing.B) {
	argminSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"2D_100x100_dim0", types.NewShape(100, 100), 0},
		{"2D_100x100_dim1", types.NewShape(100, 100), 1},
		{"3D_50x50x50_dim0", types.NewShape(50, 50, 50), 0},
		{"3D_50x50x50_dim1", types.NewShape(50, 50, 50), 1},
		{"3D_50x50x50_dim2", types.NewShape(50, 50, 50), 2},
	}
	for _, size := range argminSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.ArgMin(nil, size.dim)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ArgMin(dst, size.dim)
			}
		})
	}
}

// Benchmark Broadcasting
func BenchmarkTensorBroadcastTo_InPlace(b *testing.B) {
	broadcastCases := []struct {
		name     string
		srcShape types.Shape
		dstShape types.Shape
	}{
		{"1x100_to_10x100", types.NewShape(1, 100), types.NewShape(10, 100)},
		{"1x1x50_to_10x10x50", types.NewShape(1, 1, 50), types.NewShape(10, 10, 50)},
		{"1x50x100_to_10x50x100", types.NewShape(1, 50, 100), types.NewShape(10, 50, 100)},
		{"1x1_to_100x100", types.NewShape(1, 1), types.NewShape(100, 100)},
	}
	for _, bc := range broadcastCases {
		b.Run(bc.name, func(b *testing.B) {
			t := makeBenchTensor(bc.srcShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.BroadcastTo(nil, bc.dstShape)
			}
		})
	}
}

func BenchmarkTensorBroadcastTo_Destination(b *testing.B) {
	broadcastCases := []struct {
		name     string
		srcShape types.Shape
		dstShape types.Shape
	}{
		{"1x100_to_10x100", types.NewShape(1, 100), types.NewShape(10, 100)},
		{"1x1x50_to_10x10x50", types.NewShape(1, 1, 50), types.NewShape(10, 10, 50)},
		{"1x50x100_to_10x50x100", types.NewShape(1, 50, 100), types.NewShape(10, 50, 100)},
		{"1x1_to_100x100", types.NewShape(1, 1), types.NewShape(100, 100)},
	}
	for _, bc := range broadcastCases {
		b.Run(bc.name, func(b *testing.B) {
			t := makeBenchTensor(bc.srcShape)
			dst := New(types.FP32, bc.dstShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.BroadcastTo(dst, bc.dstShape)
			}
		})
	}
}

// Benchmark Linear Algebra Operations
var matMulSizes = []struct {
	name   string
	aShape types.Shape
	bShape types.Shape
}{
	{"2D_100x50_50x100", types.NewShape(100, 50), types.NewShape(50, 100)},
	{"2D_256x128_128x256", types.NewShape(256, 128), types.NewShape(128, 256)},
	{"2D_512x256_256x512", types.NewShape(512, 256), types.NewShape(256, 512)},
	{"batched_32x64x128_32x128x64", types.NewShape(32, 64, 128), types.NewShape(32, 128, 64)},
}

func BenchmarkTensorMatMul_InPlace(b *testing.B) {
	for _, size := range matMulSizes {
		b.Run(size.name, func(b *testing.B) {
			a := makeBenchTensor(size.aShape)
			bTensor := makeBenchTensor(size.bShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				a.MatMul(nil, bTensor)
			}
		})
	}
}

func BenchmarkTensorMatMul_Destination(b *testing.B) {
	for _, size := range matMulSizes {
		b.Run(size.name, func(b *testing.B) {
			a := makeBenchTensor(size.aShape)
			bTensor := makeBenchTensor(size.bShape)
			result := a.MatMul(nil, bTensor)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				a.MatMul(dst, bTensor)
			}
		})
	}
}

func BenchmarkTensorTranspose_InPlace(b *testing.B) {
	transposeSizes := []struct {
		name  string
		shape types.Shape
		dims  []int
	}{
		{"2D_100x50", types.NewShape(100, 50), []int{1, 0}},
		{"2D_256x128", types.NewShape(256, 128), []int{1, 0}},
		{"3D_32x64x128", types.NewShape(32, 64, 128), []int{0, 2, 1}},
	}
	for _, size := range transposeSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Transpose(nil, size.dims)
			}
		})
	}
}

func BenchmarkTensorTranspose_Destination(b *testing.B) {
	transposeSizes := []struct {
		name  string
		shape types.Shape
		dims  []int
	}{
		{"2D_100x50", types.NewShape(100, 50), []int{1, 0}},
		{"2D_256x128", types.NewShape(256, 128), []int{1, 0}},
		{"3D_32x64x128", types.NewShape(32, 64, 128), []int{0, 2, 1}},
	}
	for _, size := range transposeSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.Transpose(nil, size.dims)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Transpose(dst, size.dims)
			}
		})
	}
}

func BenchmarkTensorPermute_InPlace(b *testing.B) {
	permuteSizes := []struct {
		name  string
		shape types.Shape
		dims  []int
	}{
		{"2D_100x50", types.NewShape(100, 50), []int{1, 0}},
		{"3D_32x64x128_021", types.NewShape(32, 64, 128), []int{0, 2, 1}},
		{"3D_32x64x128_102", types.NewShape(32, 64, 128), []int{1, 0, 2}},
		{"3D_32x64x128_210", types.NewShape(32, 64, 128), []int{2, 1, 0}},
	}
	for _, size := range permuteSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Permute(nil, size.dims)
			}
		})
	}
}

func BenchmarkTensorPermute_Destination(b *testing.B) {
	permuteSizes := []struct {
		name  string
		shape types.Shape
		dims  []int
	}{
		{"2D_100x50", types.NewShape(100, 50), []int{1, 0}},
		{"3D_32x64x128_021", types.NewShape(32, 64, 128), []int{0, 2, 1}},
		{"3D_32x64x128_102", types.NewShape(32, 64, 128), []int{1, 0, 2}},
		{"3D_32x64x128_210", types.NewShape(32, 64, 128), []int{2, 1, 0}},
	}
	for _, size := range permuteSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			result := t.Permute(nil, size.dims)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Permute(dst, size.dims)
			}
		})
	}
}

func BenchmarkTensorMatMulTransposed_InPlace(b *testing.B) {
	matMulTransSizes := []struct {
		name   string
		aShape types.Shape
		bShape types.Shape
		transA bool
		transB bool
	}{
		{"2D_100x50_50x100_NN", types.NewShape(100, 50), types.NewShape(50, 100), false, false},
		{"2D_100x50_100x50_NT", types.NewShape(100, 50), types.NewShape(100, 50), false, true},
		{"2D_50x100_50x100_TN", types.NewShape(50, 100), types.NewShape(50, 100), true, false},
		{"2D_50x100_100x50_TT", types.NewShape(50, 100), types.NewShape(100, 50), true, true},
	}
	for _, size := range matMulTransSizes {
		b.Run(size.name, func(b *testing.B) {
			a := makeBenchTensor(size.aShape)
			bTensor := makeBenchTensor(size.bShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				a.MatMulTransposed(nil, bTensor, size.transA, size.transB)
			}
		})
	}
}

func BenchmarkTensorMatMulTransposed_Destination(b *testing.B) {
	matMulTransSizes := []struct {
		name   string
		aShape types.Shape
		bShape types.Shape
		transA bool
		transB bool
	}{
		{"2D_100x50_50x100_NN", types.NewShape(100, 50), types.NewShape(50, 100), false, false},
		{"2D_100x50_100x50_NT", types.NewShape(100, 50), types.NewShape(100, 50), false, true},
		{"2D_50x100_50x100_TN", types.NewShape(50, 100), types.NewShape(50, 100), true, false},
		{"2D_50x100_100x50_TT", types.NewShape(50, 100), types.NewShape(100, 50), true, true},
	}
	for _, size := range matMulTransSizes {
		b.Run(size.name, func(b *testing.B) {
			a := makeBenchTensor(size.aShape)
			bTensor := makeBenchTensor(size.bShape)
			result := a.MatMulTransposed(nil, bTensor, size.transA, size.transB)
			dst := New(types.FP32, result.Shape())
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				a.MatMulTransposed(dst, bTensor, size.transA, size.transB)
			}
		})
	}
}

func BenchmarkTensorMatVecMulTransposed_InPlace(b *testing.B) {
	matVecSizes := []struct {
		name        string
		matrixShape types.Shape
		vectorShape types.Shape
		alpha       float64
		beta        float64
	}{
		{"100x50_100", types.NewShape(100, 50), types.NewShape(100), 1.0, 0.0},
		{"256x128_256", types.NewShape(256, 128), types.NewShape(256), 1.0, 0.0},
		{"512x256_512", types.NewShape(512, 256), types.NewShape(512), 2.0, 0.5},
	}
	for _, size := range matVecSizes {
		b.Run(size.name, func(b *testing.B) {
			matrix := makeBenchTensor(size.matrixShape)
			vector := makeBenchTensor(size.vectorShape)
			// Create output vector (receiver used as output)
			output := New(types.FP32, types.NewShape(size.matrixShape[1]))
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				output.MatVecMulTransposed(nil, matrix, vector, size.alpha, size.beta)
			}
		})
	}
}

func BenchmarkTensorMatVecMulTransposed_Destination(b *testing.B) {
	matVecSizes := []struct {
		name        string
		matrixShape types.Shape
		vectorShape types.Shape
		alpha       float64
		beta        float64
	}{
		{"100x50_100", types.NewShape(100, 50), types.NewShape(100), 1.0, 0.0},
		{"256x128_256", types.NewShape(256, 128), types.NewShape(256), 1.0, 0.0},
		{"512x256_512", types.NewShape(512, 256), types.NewShape(512), 2.0, 0.5},
	}
	for _, size := range matVecSizes {
		b.Run(size.name, func(b *testing.B) {
			matrix := makeBenchTensor(size.matrixShape)
			vector := makeBenchTensor(size.vectorShape)
			result := New(types.FP32, types.NewShape(size.matrixShape[1]))
			result.MatVecMulTransposed(nil, matrix, vector, size.alpha, size.beta)
			dst := New(types.FP32, types.NewShape(size.matrixShape[1]))
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				result.MatVecMulTransposed(dst, matrix, vector, size.alpha, size.beta)
			}
		})
	}
}

func BenchmarkTensorDot(b *testing.B) {
	dotSizes := []struct {
		name  string
		shape types.Shape
	}{
		{"1D_1000", types.NewShape(1000)},
		{"1D_10000", types.NewShape(10000)},
		{"2D_100x100", types.NewShape(100, 100)},
		{"2D_256x256", types.NewShape(256, 256)},
	}
	for _, size := range dotSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.Dot(t2)
			}
		})
	}
}

// Benchmark Convolution Operations
var conv1DSizes = []struct {
	name        string
	inputShape  types.Shape // [batch, inChannels, length] or [inChannels, length]
	kernelShape types.Shape // [outChannels, inChannels, kernelLen]
	stride      int
	padding     int
}{
	{"batch32_in64_out128_k3_s1_p1", types.NewShape(32, 64, 128), types.NewShape(128, 64, 3), 1, 1},
	{"batch16_in32_out64_k5_s2_p2", types.NewShape(16, 32, 256), types.NewShape(64, 32, 5), 2, 2},
	{"batch8_in16_out32_k7_s1_p3", types.NewShape(8, 16, 512), types.NewShape(32, 16, 7), 1, 3},
	{"no_batch_in64_out128_k3_s1_p1", types.NewShape(64, 128), types.NewShape(128, 64, 3), 1, 1},
}

func BenchmarkTensorConv1D_Destination(b *testing.B) {
	for _, size := range conv1DSizes {
		b.Run(size.name, func(b *testing.B) {
			input := makeBenchTensor(size.inputShape)
			kernel := makeBenchTensor(size.kernelShape)
			// Create bias: [outChannels]
			biasShape := types.NewShape(size.kernelShape[0])
			bias := makeBenchTensor(biasShape)
			// Calculate output shape
			inputShape := input.Shape()
			var inLength int
			if len(inputShape) == 3 {
				inLength = inputShape[2]
			} else {
				inLength = inputShape[1]
			}
			outLength := (inLength+2*size.padding-size.kernelShape[2])/size.stride + 1
			var outputShape types.Shape
			if len(inputShape) == 3 {
				outputShape = types.NewShape(inputShape[0], size.kernelShape[0], outLength)
			} else {
				outputShape = types.NewShape(size.kernelShape[0], outLength)
			}
			dst := New(types.FP32, outputShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				input.Conv1D(dst, kernel, bias, size.stride, size.padding)
			}
		})
	}
}

var conv2DSizes = []struct {
	name        string
	inputShape  types.Shape // [batch, inChannels, height, width]
	kernelShape types.Shape // [outChannels, inChannels, kernelH, kernelW]
	stride      []int
	padding     []int
}{
	{"batch32_in64_out128_k3x3_s1x1_p1x1", types.NewShape(32, 64, 64, 64), types.NewShape(128, 64, 3, 3), []int{1, 1}, []int{1, 1}},
	{"batch16_in32_out64_k5x5_s2x2_p2x2", types.NewShape(16, 32, 128, 128), types.NewShape(64, 32, 5, 5), []int{2, 2}, []int{2, 2}},
	{"batch8_in16_out32_k7x7_s1x1_p3x3", types.NewShape(8, 16, 256, 256), types.NewShape(32, 16, 7, 7), []int{1, 1}, []int{3, 3}},
	{"batch1_in3_out64_k3x3_s1x1_p1x1", types.NewShape(1, 3, 224, 224), types.NewShape(64, 3, 3, 3), []int{1, 1}, []int{1, 1}},
}

func BenchmarkTensorConv2D_Destination(b *testing.B) {
	for _, size := range conv2DSizes {
		b.Run(size.name, func(b *testing.B) {
			input := makeBenchTensor(size.inputShape)
			kernel := makeBenchTensor(size.kernelShape)
			// Create bias: [outChannels]
			biasShape := types.NewShape(size.kernelShape[0])
			bias := makeBenchTensor(biasShape)
			// Calculate output shape
			inputShape := input.Shape()
			inHeight := inputShape[2]
			inWidth := inputShape[3]
			outHeight := (inHeight+2*size.padding[0]-size.kernelShape[2])/size.stride[0] + 1
			outWidth := (inWidth+2*size.padding[1]-size.kernelShape[3])/size.stride[1] + 1
			outputShape := types.NewShape(inputShape[0], size.kernelShape[0], outHeight, outWidth)
			dst := New(types.FP32, outputShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				input.Conv2D(dst, kernel, bias, size.stride, size.padding)
			}
		})
	}
}

func BenchmarkTensorConv2DTransposed_Destination(b *testing.B) {
	for _, size := range conv2DSizes {
		b.Run(size.name, func(b *testing.B) {
			input := makeBenchTensor(size.inputShape)
			// For transposed convolution, kernel shape is [inChannels, outChannels, kernelH, kernelW]
			transposedKernelShape := types.NewShape(size.kernelShape[1], size.kernelShape[0], size.kernelShape[2], size.kernelShape[3])
			kernel := makeBenchTensor(transposedKernelShape)
			// Create bias: [outChannels] - note: outChannels is now the second dimension
			biasShape := types.NewShape(transposedKernelShape[1])
			bias := makeBenchTensor(biasShape)
			// Calculate output shape for transposed convolution
			inputShape := input.Shape()
			inHeight := inputShape[2]
			inWidth := inputShape[3]
			outHeight := (inHeight-1)*size.stride[0] + size.kernelShape[2] - 2*size.padding[0]
			outWidth := (inWidth-1)*size.stride[1] + size.kernelShape[3] - 2*size.padding[1]
			outputShape := types.NewShape(inputShape[0], transposedKernelShape[1], outHeight, outWidth)
			dst := New(types.FP32, outputShape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				input.Conv2DTransposed(dst, kernel, bias, size.stride, size.padding)
			}
		})
	}
}

func BenchmarkTensorNorm(b *testing.B) {
	normSizes := []struct {
		name  string
		shape types.Shape
		ord   int
	}{
		{"1D_1000_L1", types.NewShape(1000), 0},
		{"1D_1000_L2", types.NewShape(1000), 1},
		{"1D_10000_L2", types.NewShape(10000), 1},
		{"2D_100x100_Frobenius", types.NewShape(100, 100), 2},
	}
	for _, size := range normSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Norm(size.ord)
			}
		})
	}
}

func BenchmarkTensorL2Normalize_InPlace(b *testing.B) {
	normalizeSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"1D_1000", types.NewShape(1000), 0},
		{"2D_100x50_dim0", types.NewShape(100, 50), 0},
		{"2D_100x50_dim1", types.NewShape(100, 50), 1},
	}
	for _, size := range normalizeSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.L2Normalize(nil, size.dim)
			}
		})
	}
}

func BenchmarkTensorL2Normalize_Destination(b *testing.B) {
	normalizeSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"1D_1000", types.NewShape(1000), 0},
		{"2D_100x50_dim0", types.NewShape(100, 50), 0},
		{"2D_100x50_dim1", types.NewShape(100, 50), 1},
	}
	for _, size := range normalizeSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.L2Normalize(dst, size.dim)
			}
		})
	}
}

func BenchmarkTensorAddScaled_InPlace(b *testing.B) {
	alpha := 2.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.AddScaled(nil, t2, alpha)
			}
		})
	}
}

func BenchmarkTensorAddScaled_Destination(b *testing.B) {
	alpha := 2.5
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t1 := makeBenchTensor(size.shape)
			t2 := makeBenchTensorOther(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t1.AddScaled(dst, t2, alpha)
			}
		})
	}
}

// Benchmark Additional Activation Functions
func BenchmarkTensorSoftmax_InPlace(b *testing.B) {
	softmaxSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"1D_1000", types.NewShape(1000), 0},
		{"2D_100x50_dim0", types.NewShape(100, 50), 0},
		{"2D_100x50_dim1", types.NewShape(100, 50), 1},
		{"3D_32x64x128_dim2", types.NewShape(32, 64, 128), 2},
	}
	for _, size := range softmaxSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Softmax(size.dim, nil)
			}
		})
	}
}

func BenchmarkTensorSoftmax_Destination(b *testing.B) {
	softmaxSizes := []struct {
		name  string
		shape types.Shape
		dim   int
	}{
		{"1D_1000", types.NewShape(1000), 0},
		{"2D_100x50_dim0", types.NewShape(100, 50), 0},
		{"2D_100x50_dim1", types.NewShape(100, 50), 1},
		{"3D_32x64x128_dim2", types.NewShape(32, 64, 128), 2},
	}
	for _, size := range softmaxSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Softmax(size.dim, dst)
			}
		})
	}
}

func BenchmarkTensorReLU6_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ReLU6(nil)
			}
		})
	}
}

func BenchmarkTensorReLU6_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ReLU6(dst)
			}
		})
	}
}

func BenchmarkTensorLeakyReLU_InPlace(b *testing.B) {
	alpha := 0.1
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.LeakyReLU(nil, alpha)
			}
		})
	}
}

func BenchmarkTensorLeakyReLU_Destination(b *testing.B) {
	alpha := 0.1
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.LeakyReLU(dst, alpha)
			}
		})
	}
}

func BenchmarkTensorELU_InPlace(b *testing.B) {
	alpha := 1.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ELU(nil, alpha)
			}
		})
	}
}

func BenchmarkTensorELU_Destination(b *testing.B) {
	alpha := 1.0
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			// Mix positive and negative values
			data := t.Data().([]float32)
			for i := range data {
				if i%2 == 0 {
					data[i] = -data[i]
				}
			}
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.ELU(dst, alpha)
			}
		})
	}
}

func BenchmarkTensorSoftplus_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Softplus(nil)
			}
		})
	}
}

func BenchmarkTensorSoftplus_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Softplus(dst)
			}
		})
	}
}

func BenchmarkTensorSwish_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Swish(nil)
			}
		})
	}
}

func BenchmarkTensorSwish_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.Swish(dst)
			}
		})
	}
}

func BenchmarkTensorGELU_InPlace(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.GELU(nil)
			}
		})
	}
}

func BenchmarkTensorGELU_Destination(b *testing.B) {
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.GELU(dst)
			}
		})
	}
}

// Benchmark Dropout Operations
func BenchmarkTensorDropoutForward_InPlace(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			maskTensor := New(types.FP32, size.shape)
			mask := maskTensor.DropoutMask(0.5, 2.0, rng) // 50% dropout, scale by 2.0
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.DropoutForward(nil, mask)
			}
		})
	}
}

func BenchmarkTensorDropoutForward_Destination(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			t := makeBenchTensor(size.shape)
			maskTensor := New(types.FP32, size.shape)
			mask := maskTensor.DropoutMask(0.5, 2.0, rng) // 50% dropout, scale by 2.0
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t.DropoutForward(dst, mask)
			}
		})
	}
}

func BenchmarkTensorDropoutBackward_InPlace(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			gradOutput := makeBenchTensor(size.shape)
			maskTensor := New(types.FP32, size.shape)
			mask := maskTensor.DropoutMask(0.5, 2.0, rng)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				gradOutput.DropoutBackward(nil, gradOutput, mask)
			}
		})
	}
}

func BenchmarkTensorDropoutBackward_Destination(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, size := range benchSizes {
		b.Run(size.name, func(b *testing.B) {
			gradOutput := makeBenchTensor(size.shape)
			maskTensor := New(types.FP32, size.shape)
			mask := maskTensor.DropoutMask(0.5, 2.0, rng)
			dst := New(types.FP32, size.shape)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				gradOutput.DropoutBackward(dst, gradOutput, mask)
			}
		})
	}
}
