package gorgonia

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Benchmark MatMul operations with various matrix sizes

func BenchmarkMatMul_Small_32x32(b *testing.B) {
	a := New(types.FP32, 32, 32)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 32, 32)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_Medium_128x128(b *testing.B) {
	a := New(types.FP32, 128, 128)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 128, 128)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_Large_512x512(b *testing.B) {
	a := New(types.FP32, 512, 512)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 512, 512)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_Rectangular_64x128x256(b *testing.B) {
	// [64, 128] x [128, 256] = [64, 256]
	a := New(types.FP32, 64, 128)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 128, 256)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_BatchedSmall_8x32x32(b *testing.B) {
	// Batched MatMul not fully supported in gorgonia wrapper yet
	// Skipping for now
	b.Skip("Batched MatMul not yet fully supported")

	// Batched: [8, 32, 32] x [8, 32, 32] = [8, 32, 32]
	a := New(types.FP32, 8, 32, 32)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 8, 32, 32)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_Reused_128x128(b *testing.B) {
	a := New(types.FP32, 128, 128)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 128, 128)
	c.Fill(nil, 2.0)
	dst := New(types.FP32, 128, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(dst, c)
	}
}

// Benchmark Add operations with various tensor sizes

func BenchmarkAdd_Small_1K(b *testing.B) {
	// 1K elements
	a := New(types.FP32, 32, 32)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 32, 32)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(nil, c)
	}
}

func BenchmarkAdd_Medium_64K(b *testing.B) {
	// 64K elements
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 256, 256)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(nil, c)
	}
}

func BenchmarkAdd_Large_1M(b *testing.B) {
	// 1M elements
	a := New(types.FP32, 1024, 1024)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 1024, 1024)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(nil, c)
	}
}

func BenchmarkAdd_Reused_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)
	c := New(types.FP32, 256, 256)
	c.Fill(nil, 2.0)
	dst := New(types.FP32, 256, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(dst, c)
	}
}

func BenchmarkAdd_InPlace_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	c := New(types.FP32, 256, 256)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Fill(nil, 1.0)
		_ = a.Add(a, c)
	}
}

// Benchmark 1D Convolutions

func BenchmarkConv1D_Kernel3_Stride1(b *testing.B) {
	// Input: [batch=8, inChannels=32, length=128]
	// Kernel: [outChannels=64, inChannels=32, kernelLen=3]
	// Note: Conv1D not yet implemented in gorgonia wrapper, will panic
	// Skipping for now since it's not implemented
	b.Skip("Conv1D not yet implemented")

	input := New(types.FP32, 8, 32, 128)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 64, 32, 3)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv1D(nil, kernel, nil, 1, 0)
	}
}

func BenchmarkConv1D_Kernel5_Stride2(b *testing.B) {
	b.Skip("Conv1D not yet implemented")

	// Input: [batch=8, inChannels=64, length=256]
	// Kernel: [outChannels=128, inChannels=64, kernelLen=5]
	input := New(types.FP32, 8, 64, 256)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 128, 64, 5)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv1D(nil, kernel, nil, 2, 1)
	}
}

// Benchmark 2D Convolutions

func BenchmarkConv2D_3x3_Stride1_Small(b *testing.B) {
	// Note: Conv2D not yet implemented in gorgonia wrapper, will panic
	// Skipping for now since it's not implemented
	b.Skip("Conv2D not yet implemented")

	// Input: [batch=4, inChannels=32, height=32, width=32]
	// Kernel: [outChannels=64, inChannels=32, kernelH=3, kernelW=3]
	input := New(types.FP32, 4, 32, 32, 32)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 64, 32, 3, 3)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv2D(nil, kernel, nil, []int{1, 1}, []int{1, 1})
	}
}

func BenchmarkConv2D_3x3_Stride1_Medium(b *testing.B) {
	b.Skip("Conv2D not yet implemented")

	// Input: [batch=8, inChannels=64, height=64, width=64]
	// Kernel: [outChannels=128, inChannels=64, kernelH=3, kernelW=3]
	input := New(types.FP32, 8, 64, 64, 64)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 128, 64, 3, 3)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv2D(nil, kernel, nil, []int{1, 1}, []int{1, 1})
	}
}

func BenchmarkConv2D_3x3_Stride2_Large(b *testing.B) {
	b.Skip("Conv2D not yet implemented")

	// Input: [batch=4, inChannels=128, height=128, width=128]
	// Kernel: [outChannels=256, inChannels=128, kernelH=3, kernelW=3]
	input := New(types.FP32, 4, 128, 128, 128)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 256, 128, 3, 3)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv2D(nil, kernel, nil, []int{2, 2}, []int{1, 1})
	}
}

func BenchmarkConv2D_5x5_Stride1(b *testing.B) {
	b.Skip("Conv2D not yet implemented")

	// Input: [batch=8, inChannels=32, height=56, width=56]
	// Kernel: [outChannels=64, inChannels=32, kernelH=5, kernelW=5]
	input := New(types.FP32, 8, 32, 56, 56)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 64, 32, 5, 5)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv2D(nil, kernel, nil, []int{1, 1}, []int{2, 2})
	}
}

func BenchmarkConv2D_1x1_Stride1(b *testing.B) {
	b.Skip("Conv2D not yet implemented")

	// 1x1 convolution (common in modern architectures like ResNet, MobileNet)
	// Input: [batch=8, inChannels=256, height=56, width=56]
	// Kernel: [outChannels=128, inChannels=256, kernelH=1, kernelW=1]
	input := New(types.FP32, 8, 256, 56, 56)
	input.Fill(nil, 1.0)
	kernel := New(types.FP32, 128, 256, 1, 1)
	kernel.Fill(nil, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.Conv2D(nil, kernel, nil, []int{1, 1}, []int{0, 0})
	}
}

// Additional benchmarks for common operations

func BenchmarkMultiply_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.5)
	c := New(types.FP32, 256, 256)
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Multiply(nil, c)
	}
}

func BenchmarkScalarMul_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.ScalarMul(nil, 2.0)
	}
}

func BenchmarkReLU_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, -0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.ReLU(nil)
	}
}

func BenchmarkSigmoid_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Sigmoid(nil)
	}
}

func BenchmarkTranspose_128x256(b *testing.B) {
	a := New(types.FP32, 128, 256)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Transpose(nil, nil)
	}
}

func BenchmarkReshape_1Mx4(b *testing.B) {
	// Reshape from [1024, 1024] to [262144, 4]
	a := New(types.FP32, 1024, 1024)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Reshape(nil, []int{262144, 4})
	}
}

func BenchmarkSum_All_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Sum(nil, nil)
	}
}

func BenchmarkSum_Axis0_256x256(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Sum(nil, []int{0})
	}
}

func BenchmarkMean_All_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Mean(nil, nil)
	}
}

func BenchmarkClone_64K(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Clone()
	}
}

func BenchmarkCopy_64K(b *testing.B) {
	src := New(types.FP32, 256, 256)
	src.Fill(nil, 1.0)
	dst := New(types.FP32, 256, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst.Copy(src)
	}
}

func BenchmarkAtLinear_Sequential(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)
	size := a.Size()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < size; j++ {
			_ = a.At(j)
		}
	}
}

func BenchmarkAtMultiDim_Sequential(b *testing.B) {
	a := New(types.FP32, 256, 256)
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 256; row++ {
			for col := 0; col < 256; col++ {
				_ = a.At(row, col)
			}
		}
	}
}

func BenchmarkSetAtLinear_Sequential(b *testing.B) {
	a := New(types.FP32, 256, 256)
	size := a.Size()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < size; j++ {
			a.SetAt(float64(j), j)
		}
	}
}

