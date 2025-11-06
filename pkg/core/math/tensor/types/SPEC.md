# Tensor Interface API Specification

## Tensor Interface Methods

### Core Properties and Access (TensorCore)
- ID() uintptr
- DataType() DataType
- Data() any
- Shape() Shape
- Rank() int
- Size() int
- Empty() bool
- At(indices ...int) float64
- SetAt(value float64, indices ...int)
- Elements(fixedAxisValuePairs ...int) func(func(Element) bool)

### Manipulation Operations (TensorManipulation)
- Clone() Tensor
- Copy(src Tensor) Tensor
- Reshape(newShape Shape) Tensor
- Slice(dim int, start int, length int) Tensor
- Transpose(dst Tensor, dims []int) Tensor
- Permute(dims []int) Tensor
- BroadcastTo(shape Shape) (Tensor, error)
- Fill(dst Tensor, value float64) Tensor
- Pad(dst Tensor, padding []int, value float64) Tensor
- Unpad(padding []int) Tensor

### Element-Wise Operations (TensorElementWise)
- Add(dst Tensor, other Tensor) Tensor
- Subtract(dst Tensor, other Tensor) Tensor
- Multiply(dst Tensor, other Tensor) Tensor
- Divide(dst Tensor, other Tensor) Tensor
- ScalarMul(dst Tensor, scalar float64) Tensor
- AddScalar(dst Tensor, scalar float64) Tensor
- SubScalar(dst Tensor, scalar float64) Tensor
- MulScalar(dst Tensor, scalar float64) Tensor
- DivScalar(dst Tensor, scalar float64) Tensor
- Square(dst Tensor) Tensor
- Sqrt(dst Tensor) Tensor
- Exp(dst Tensor) Tensor
- Log(dst Tensor) Tensor
- Pow(dst Tensor, power float64) Tensor
- Abs(dst Tensor) Tensor
- Sign(dst Tensor) Tensor
- Cos(dst Tensor) Tensor
- Sin(dst Tensor) Tensor
- Negative(dst Tensor) Tensor
- Equal(other Tensor) Tensor
- GreaterThan(other Tensor) Tensor
- Greater(other Tensor) Tensor
- Less(other Tensor) Tensor
- NotEqual(other Tensor) Tensor
- GreaterEqual(other Tensor) Tensor
- LessEqual(other Tensor) Tensor
- Where(dst Tensor, condition, a, b Tensor) Tensor

### Math Operations (TensorMath)
- Sum(dst Tensor, dims []int) Tensor
- ReduceSum(dst Tensor, dims []int) Tensor
- Mean(dst Tensor, dims []int) Tensor
- ReduceMean(dst Tensor, dims []int) Tensor
- Max(dst Tensor, dims []int) Tensor
- ReduceMax(dst Tensor, dims []int) Tensor
- Min(dst Tensor, dims []int) Tensor
- ReduceMin(dst Tensor, dims []int) Tensor
- ArgMax(dst Tensor, dim int) Tensor
- ArgMin(dst Tensor, dim int) Tensor
- MatMul(dst Tensor, other Tensor) Tensor
- MatMulTransposed(dst Tensor, other Tensor, transposeA, transposeB bool) Tensor
- MatVecMulTransposed(dst Tensor, matrix, vector Tensor, alpha, beta float64) Tensor
- Dot(other Tensor) float64
- Tensordot(other Tensor) float64
- Norm(ord int) float64
- L2Normalize(dst Tensor, dim int) Tensor
- Normalize(dst Tensor, dim int) Tensor
- AddScaled(dst Tensor, other Tensor, alpha float64) Tensor
- ScatterAdd(dst, index, value Tensor) Tensor

### Activation Functions (TensorActivations)
- ReLU(dst Tensor) Tensor
- Sigmoid(dst Tensor) Tensor
- Tanh(dst Tensor) Tensor
- Softmax(dim int, dst Tensor) Tensor
- ReLU6(dst Tensor) Tensor
- LeakyReLU(dst Tensor, alpha float64) Tensor
- ELU(dst Tensor, alpha float64) Tensor
- Softplus(dst Tensor) Tensor
- Swish(dst Tensor) Tensor
- GELU(dst Tensor) Tensor

### Convolution Operations (TensorConvolutions)
- Conv1D(dst Tensor, kernel, bias Tensor, stride, padding int) Tensor
- Conv2D(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor
- Conv2DTransposed(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor
- Conv2DKernelGrad(outputGrad, kernel Tensor, stride, padding []int) Tensor
- Conv1DKernelGrad(outputGrad, kernel Tensor, stride, padding int) Tensor
- Im2Col(kernelSize, stride, padding []int) Tensor
- Col2Im(outputShape, kernelSize, stride, padding []int) Tensor

### Pooling Operations (TensorPooling)
- MaxPool2D(kernelSize, stride, padding []int) Tensor
- MaxPool2DWithIndices(kernelSize, stride, padding []int) (Tensor, Tensor)
- MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
- AvgPool2D(kernelSize, stride, padding []int) Tensor
- AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor
- GlobalAvgPool2D() Tensor
- AdaptiveAvgPool2D(outputSize []int) Tensor

### Dropout Operations (TensorDropout)
- DropoutForward(dst Tensor, mask Tensor) Tensor
- DropoutMask(p, scale float64, rng RNG) Tensor
- DropoutBackward(dst Tensor, gradOutput, mask Tensor) Tensor

