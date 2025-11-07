package eager_tensor

import (
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
