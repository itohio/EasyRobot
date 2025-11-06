package types

// TensorConvolutions defines convolution operations for neural networks.
// This interface contains operations for performing convolutions on tensors.
type TensorConvolutions interface {
	// Conv1D performs 1D convolution (implemented via 2D conv with width=1).
	// Input: [inChannels, length] or [batch, inChannels, length]
	// Kernel: [outChannels, inChannels, kernelLen]
	// Bias: [outChannels] (optional, can be nil)
	// Output: [outChannels, outLen] or [batch, outChannels, outLen]
	// Returns a new tensor. Panics if shapes are incompatible.
	Conv1D(kernel, bias Tensor, stride, padding int) Tensor

	// Conv1DTo performs 1D convolution and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Conv1DTo(kernel, bias Tensor, dst Tensor, stride, padding int) Tensor

	// Conv2D performs 2D convolution.
	// Input: [batch, inChannels, height, width]
	// Kernel: [outChannels, inChannels, kernelH, kernelW]
	// Bias: [outChannels] (optional, can be nil)
	// Stride: [strideH, strideW]
	// Padding: [padH, padW]
	// Output: [batch, outChannels, outHeight, outWidth]
	// Returns a new tensor. Panics if shapes are incompatible.
	Conv2D(kernel, bias Tensor, stride, padding []int) Tensor

	// Conv2DTo performs 2D convolution and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Conv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor

	// Conv2DTransposed performs transposed 2D convolution (deconvolution).
	// Input: [batch, inChannels, height, width]
	// Kernel: [inChannels, outChannels, kernelH, kernelW] (transposed layout)
	// Bias: [outChannels] (optional, can be nil)
	// Output: [batch, outChannels, outHeight, outWidth]
	// Returns a new tensor. Panics if shapes are incompatible.
	Conv2DTransposed(kernel, bias Tensor, stride, padding []int) Tensor

	// Conv2DKernelGrad computes the gradient of the convolution kernel.
	// Used in backpropagation for training convolutional layers.
	// Returns a new tensor with kernel gradient.
	Conv2DKernelGrad(outputGrad, kernel Tensor, stride, padding []int) Tensor

	// Conv1DKernelGrad computes the gradient of the 1D convolution kernel.
	// Used in backpropagation for training 1D convolutional layers.
	// Returns a new tensor with kernel gradient.
	Conv1DKernelGrad(outputGrad, kernel Tensor, stride, padding int) Tensor

	// Image/Column Conversion
	// These operations convert between image patches and column format for efficient convolution computation.

	// Im2Col converts image patches to columns for GEMM-based convolution.
	// Input: [batch, channels, height, width]
	// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	// Returns a new tensor. Used internally for optimized convolution computation.
	Im2Col(kernelSize, stride, padding []int) Tensor

	// Col2Im converts columns back to image (inverse of Im2Col).
	// Input: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	// Output: [batch, channels, height, width]
	// Returns a new tensor. Used in backpropagation for convolution gradients.
	Col2Im(outputShape, kernelSize, stride, padding []int) Tensor
}

