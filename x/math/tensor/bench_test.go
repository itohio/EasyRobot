package tensor

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/gorgonia"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Comparative benchmarks between different tensor implementations
// This allows direct performance comparison of eager_tensor vs gorgonia (graph-based) vs (future) tflite
//
// NOTE: Gorgonia benchmarks use graph-based execution, which amortizes compilation cost
// across many executions. This is the intended usage pattern for Gorgonia.

// ============================================================================
// MatMul Benchmarks
// ============================================================================

func BenchmarkMatMul_Eager_Small_32x32(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(32, 32))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(32, 32))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_GorgoniaGraph_Small_32x32(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 32, 32).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 32, 32).(*gorgonia.GraphTensor)
	result := a.MatMul(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 32*32)
	cData := make([]float32, 32*32)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = 2.0
	}

	b.ResetTimer()
	// Execute many times (amortize compilation cost)
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{32, 32}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{32, 32}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkMatMul_Eager_Medium_128x128(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(128, 128))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(128, 128))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_GorgoniaGraph_Medium_128x128(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 128, 128).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 128, 128).(*gorgonia.GraphTensor)
	result := a.MatMul(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 128*128)
	cData := make([]float32, 128*128)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{128, 128}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{128, 128}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkMatMul_Eager_Large_512x512(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(512, 512))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(512, 512))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_GorgoniaGraph_Large_512x512(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 512, 512).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 512, 512).(*gorgonia.GraphTensor)
	result := a.MatMul(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 512*512)
	cData := make([]float32, 512*512)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{512, 512}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{512, 512}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkMatMul_Eager_Rectangular_64x128x256(b *testing.B) {
	// [64, 128] x [128, 256] = [64, 256]
	a := eager_tensor.New(types.FP32, types.NewShape(64, 128))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(128, 256))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.MatMul(nil, c)
	}
}

func BenchmarkMatMul_GorgoniaGraph_Rectangular_64x128x256(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 64, 128).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 128, 256).(*gorgonia.GraphTensor)
	result := a.MatMul(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 64*128)
	cData := make([]float32, 128*256)
	for i := range aData {
		aData[i] = 1.0
	}
	for i := range cData {
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{64, 128}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{128, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

// ============================================================================
// Add Benchmarks
// ============================================================================

func BenchmarkAdd_Eager_Small_1K(b *testing.B) {
	// 1K elements
	a := eager_tensor.New(types.FP32, types.NewShape(32, 32))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(32, 32))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(nil, c)
	}
}

func BenchmarkAdd_GorgoniaGraph_Small_1K(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 32, 32).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 32, 32).(*gorgonia.GraphTensor)
	result := a.Add(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 32*32)
	cData := make([]float32, 32*32)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{32, 32}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{32, 32}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkAdd_Eager_Medium_64K(b *testing.B) {
	// 64K elements
	a := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(nil, c)
	}
}

func BenchmarkAdd_GorgoniaGraph_Medium_64K(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	result := a.Add(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 256*256)
	cData := make([]float32, 256*256)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{256, 256}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{256, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkAdd_Eager_Large_1M(b *testing.B) {
	// 1M elements
	a := eager_tensor.New(types.FP32, types.NewShape(1024, 1024))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(1024, 1024))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(nil, c)
	}
}

func BenchmarkAdd_GorgoniaGraph_Large_1M(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 1024, 1024).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 1024, 1024).(*gorgonia.GraphTensor)
	result := a.Add(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 1024*1024)
	cData := make([]float32, 1024*1024)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{1024, 1024}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{1024, 1024}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

// ============================================================================
// Composite Operations (showing graph benefits)
// ============================================================================

func BenchmarkComposite_Eager_MatMulReLU_128x128(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(128, 128))
	a.Fill(nil, 1.0)
	c := eager_tensor.New(types.FP32, types.NewShape(128, 128))
	c.Fill(nil, -0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := a.MatMul(nil, c)
		_ = result.ReLU(nil)
	}
}

func BenchmarkComposite_GorgoniaGraph_MatMulReLU_128x128(b *testing.B) {
	// Build graph once - composition happens at graph construction
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 128, 128).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 128, 128).(*gorgonia.GraphTensor)
	// Compose operations in graph
	result := a.MatMul(nil, c).ReLU(nil).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 128*128)
	cData := make([]float32, 128*128)
	for i := range aData {
		aData[i] = 1.0
		cData[i] = -0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{128, 128}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{128, 128}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkComposite_Eager_AddMulSigmoid_64K(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	a.Fill(nil, 0.5)
	c := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tmp := a.Add(nil, c)
		tmp = tmp.MulScalar(nil, 0.5)
		_ = tmp.Sigmoid(nil)
	}
}

func BenchmarkComposite_GorgoniaGraph_AddMulSigmoid_64K(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	// Compose operations in graph
	result := a.Add(nil, c).MulScalar(nil, 0.5).Sigmoid(nil).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 256*256)
	cData := make([]float32, 256*256)
	for i := range aData {
		aData[i] = 0.5
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{256, 256}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{256, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

// ============================================================================
// Additional Common Operations for Comparison
// ============================================================================

func BenchmarkMultiply_Eager_64K(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	a.Fill(nil, 1.5)
	c := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	c.Fill(nil, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Multiply(nil, c)
	}
}

func BenchmarkMultiply_GorgoniaGraph_64K(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	c := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	result := a.Multiply(nil, c).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 256*256)
	cData := make([]float32, 256*256)
	for i := range aData {
		aData[i] = 1.5
		cData[i] = 2.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{256, 256}, dtype: types.FP32})
		c.Copy(&simpleTensor{data: cData, shape: types.Shape{256, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkReLU_Eager_64K(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	a.Fill(nil, -0.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.ReLU(nil)
	}
}

func BenchmarkReLU_GorgoniaGraph_64K(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	result := a.ReLU(nil).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 256*256)
	for i := range aData {
		aData[i] = -0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{256, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkTranspose_Eager_128x256(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(128, 256))
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Transpose(nil, nil)
	}
}

func BenchmarkTranspose_GorgoniaGraph_128x256(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 128, 256).(*gorgonia.GraphTensor)
	result := a.Transpose(nil, nil).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 128*256)
	for i := range aData {
		aData[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{128, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

func BenchmarkSum_Eager_All_64K(b *testing.B) {
	a := eager_tensor.New(types.FP32, types.NewShape(256, 256))
	a.Fill(nil, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Sum(nil, nil)
	}
}

func BenchmarkSum_GorgoniaGraph_All_64K(b *testing.B) {
	// Build graph once
	eg := gorgonia.NewExpressionGraph()
	a := eg.New(types.FP32, 256, 256).(*gorgonia.GraphTensor)
	result := a.Sum(nil, nil).(*gorgonia.GraphTensor)

	// Compile once
	if err := eg.Compile(); err != nil {
		b.Fatal(err)
	}

	// Prepare input data
	aData := make([]float32, 256*256)
	for i := range aData {
		aData[i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Copy(&simpleTensor{data: aData, shape: types.Shape{256, 256}, dtype: types.FP32})
		if err := eg.Compute(); err != nil {
			b.Fatal(err)
		}
		_ = result.Data()
	}
}

// ============================================================================
// Helper type for benchmarks
// ============================================================================

// simpleTensor is a minimal tensor implementation for feeding data to graph tensors
type simpleTensor struct {
	data  any
	shape types.Shape
	dtype types.DataType
}

func (st *simpleTensor) Data() any                                             { return st.data }
func (st *simpleTensor) Shape() types.Shape                                    { return st.shape }
func (st *simpleTensor) DataType() types.DataType                              { return st.dtype }
func (st *simpleTensor) ID() uintptr                                           { return 0 }
func (st *simpleTensor) Rank() int                                             { return len(st.shape) }
func (st *simpleTensor) Size() int                                             { return st.shape.Size() }
func (st *simpleTensor) Empty() bool                                           { return st.Size() == 0 }
func (st *simpleTensor) Strides(dst []int) []int                               { return st.shape.Strides(dst) }
func (st *simpleTensor) IsContiguous() bool                                    { return true }
func (st *simpleTensor) Offset() int                                           { return 0 }
func (st *simpleTensor) DataWithOffset() any                                   { return st.data }
func (st *simpleTensor) At(...int) float64                                     { return 0 }
func (st *simpleTensor) SetAt(float64, ...int)                                 {}
func (st *simpleTensor) Elements(...int) func(func(types.Element) bool)        { return nil }
func (st *simpleTensor) Release()                                              {}
func (st *simpleTensor) Clone() types.Tensor                                   { return st }
func (st *simpleTensor) Copy(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Reshape(types.Tensor, types.Shape) types.Tensor        { return st }
func (st *simpleTensor) Slice(types.Tensor, int, int, int) types.Tensor        { return st }
func (st *simpleTensor) Transpose(types.Tensor, []int) types.Tensor            { return st }
func (st *simpleTensor) Permute(types.Tensor, []int) types.Tensor              { return st }
func (st *simpleTensor) BroadcastTo(types.Tensor, types.Shape) types.Tensor    { return st }
func (st *simpleTensor) Fill(types.Tensor, float64) types.Tensor               { return st }
func (st *simpleTensor) FillFunc(types.Tensor, func() float64) types.Tensor    { return st }
func (st *simpleTensor) Pad(types.Tensor, []int, float64) types.Tensor         { return st }
func (st *simpleTensor) Unpad(types.Tensor, []int) types.Tensor                { return st }
func (st *simpleTensor) Add(types.Tensor, types.Tensor) types.Tensor           { return st }
func (st *simpleTensor) Subtract(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) Multiply(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) Divide(types.Tensor, types.Tensor) types.Tensor        { return st }
func (st *simpleTensor) ScalarMul(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) AddScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) SubScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) MulScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) DivScalar(types.Tensor, float64) types.Tensor          { return st }
func (st *simpleTensor) Square(types.Tensor) types.Tensor                      { return st }
func (st *simpleTensor) Sqrt(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Exp(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Log(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Pow(types.Tensor, float64) types.Tensor                { return st }
func (st *simpleTensor) Abs(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Sign(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Cos(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Sin(types.Tensor) types.Tensor                         { return st }
func (st *simpleTensor) Negative(types.Tensor) types.Tensor                    { return st }
func (st *simpleTensor) Equal(types.Tensor, types.Tensor) types.Tensor         { return st }
func (st *simpleTensor) Greater(types.Tensor, types.Tensor) types.Tensor       { return st }
func (st *simpleTensor) Less(types.Tensor, types.Tensor) types.Tensor          { return st }
func (st *simpleTensor) NotEqual(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) GreaterEqual(types.Tensor, types.Tensor) types.Tensor  { return st }
func (st *simpleTensor) LessEqual(types.Tensor, types.Tensor) types.Tensor     { return st }
func (st *simpleTensor) EqualScalar(types.Tensor, float64) types.Tensor        { return st }
func (st *simpleTensor) NotEqualScalar(types.Tensor, float64) types.Tensor     { return st }
func (st *simpleTensor) GreaterScalar(types.Tensor, float64) types.Tensor      { return st }
func (st *simpleTensor) LessScalar(types.Tensor, float64) types.Tensor         { return st }
func (st *simpleTensor) GreaterEqualScalar(types.Tensor, float64) types.Tensor { return st }
func (st *simpleTensor) LessEqualScalar(types.Tensor, float64) types.Tensor    { return st }
func (st *simpleTensor) Where(types.Tensor, types.Tensor, types.Tensor, types.Tensor) types.Tensor {
	return st
}
func (st *simpleTensor) Sum(types.Tensor, []int) types.Tensor           { return st }
func (st *simpleTensor) ReduceSum(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) Mean(types.Tensor, []int) types.Tensor          { return st }
func (st *simpleTensor) ReduceMean(types.Tensor, []int) types.Tensor    { return st }
func (st *simpleTensor) Max(types.Tensor, []int) types.Tensor           { return st }
func (st *simpleTensor) ReduceMax(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) Min(types.Tensor, []int) types.Tensor           { return st }
func (st *simpleTensor) ReduceMin(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) ArgMax(types.Tensor, int) types.Tensor          { return st }
func (st *simpleTensor) ArgMin(types.Tensor, int) types.Tensor          { return st }
func (st *simpleTensor) MatMul(types.Tensor, types.Tensor) types.Tensor { return st }
func (st *simpleTensor) MatMulTransposed(types.Tensor, types.Tensor, bool, bool) types.Tensor {
	return st
}
func (st *simpleTensor) MatVecMulTransposed(types.Tensor, types.Tensor, types.Tensor, float64, float64) types.Tensor {
	return st
}
func (st *simpleTensor) Dot(types.Tensor) float64                                         { return 0 }
func (st *simpleTensor) Tensordot(types.Tensor) float64                                   { return 0 }
func (st *simpleTensor) Norm(int) float64                                                 { return 0 }
func (st *simpleTensor) L2Normalize(types.Tensor, int) types.Tensor                       { return st }
func (st *simpleTensor) Normalize(types.Tensor, int) types.Tensor                         { return st }
func (st *simpleTensor) AddScaled(types.Tensor, types.Tensor, float64) types.Tensor       { return st }
func (st *simpleTensor) ScatterAdd(types.Tensor, types.Tensor, types.Tensor) types.Tensor { return st }
func (st *simpleTensor) BatchNormForward(types.Tensor, types.Tensor, types.Tensor, float64) types.Tensor {
	return st
}
func (st *simpleTensor) BatchNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) LayerNormForward(types.Tensor, types.Tensor, types.Tensor, float64) types.Tensor {
	return st
}
func (st *simpleTensor) LayerNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) RMSNormForward(types.Tensor, types.Tensor, float64) types.Tensor { return st }
func (st *simpleTensor) RMSNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor) {
	return st, st
}
func (st *simpleTensor) InstanceNorm2D(types.Tensor, types.Tensor, types.Tensor, float64) types.Tensor {
	return st
}
func (st *simpleTensor) InstanceNorm2DGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) GroupNormForward(types.Tensor, types.Tensor, types.Tensor, int, float64) types.Tensor {
	return st
}
func (st *simpleTensor) GroupNormGrad(types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, types.Tensor, int, float64) (types.Tensor, types.Tensor, types.Tensor) {
	return st, st, st
}
func (st *simpleTensor) ReLU(types.Tensor) types.Tensor                           { return st }
func (st *simpleTensor) Sigmoid(types.Tensor) types.Tensor                        { return st }
func (st *simpleTensor) Tanh(types.Tensor) types.Tensor                           { return st }
func (st *simpleTensor) Softmax(int, types.Tensor) types.Tensor                   { return st }
func (st *simpleTensor) ReLU6(types.Tensor) types.Tensor                          { return st }
func (st *simpleTensor) LeakyReLU(types.Tensor, float64) types.Tensor             { return st }
func (st *simpleTensor) ELU(types.Tensor, float64) types.Tensor                   { return st }
func (st *simpleTensor) Softplus(types.Tensor) types.Tensor                       { return st }
func (st *simpleTensor) Swish(types.Tensor) types.Tensor                          { return st }
func (st *simpleTensor) GELU(types.Tensor) types.Tensor                           { return st }
func (st *simpleTensor) ReLUGrad(types.Tensor, types.Tensor) types.Tensor         { return st }
func (st *simpleTensor) SigmoidGrad(types.Tensor, types.Tensor) types.Tensor      { return st }
func (st *simpleTensor) TanhGrad(types.Tensor, types.Tensor) types.Tensor         { return st }
func (st *simpleTensor) SoftmaxGrad(types.Tensor, types.Tensor, int) types.Tensor { return st }
func (st *simpleTensor) Conv1D(types.Tensor, types.Tensor, types.Tensor, int, int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv2D(types.Tensor, types.Tensor, types.Tensor, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv2DTransposed(types.Tensor, types.Tensor, types.Tensor, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv2DKernelGrad(types.Tensor, types.Tensor, types.Tensor, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) Conv1DKernelGrad(types.Tensor, types.Tensor, types.Tensor, int, int) types.Tensor {
	return st
}
func (st *simpleTensor) Im2Col(types.Tensor, []int, []int, []int) types.Tensor { return st }
func (st *simpleTensor) Col2Im(types.Tensor, []int, []int, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) MaxPool2D(types.Tensor, []int, []int, []int) types.Tensor { return st }
func (st *simpleTensor) MaxPool2DWithIndices(types.Tensor, types.Tensor, []int, []int, []int) (types.Tensor, types.Tensor) {
	return st, st
}
func (st *simpleTensor) MaxPool2DBackward(types.Tensor, types.Tensor, types.Tensor, []int, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) AvgPool2D(types.Tensor, []int, []int, []int) types.Tensor { return st }
func (st *simpleTensor) AvgPool2DBackward(types.Tensor, types.Tensor, []int, []int, []int) types.Tensor {
	return st
}
func (st *simpleTensor) GlobalAvgPool2D(types.Tensor) types.Tensor              { return st }
func (st *simpleTensor) AdaptiveAvgPool2D(types.Tensor, []int) types.Tensor     { return st }
func (st *simpleTensor) DropoutForward(types.Tensor, types.Tensor) types.Tensor { return st }
func (st *simpleTensor) DropoutMask(float64, float64, types.RNG) types.Tensor   { return st }
func (st *simpleTensor) DropoutBackward(types.Tensor, types.Tensor, types.Tensor) types.Tensor {
	return st
}
