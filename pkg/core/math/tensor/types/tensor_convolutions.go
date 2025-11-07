package types

// TensorConvolutions defines convolution operations for neural networks.
// This interface contains operations for performing convolutions on tensors.
type TensorConvolutions interface {
	// Conv1D performs 1D convolution (implemented via 2D conv with width=1) (matches tf.nn.conv1d).
	// Input: [inChannels, length] or [batch, inChannels, length]
	// Kernel: [outChannels, inChannels, kernelLen]
	// Bias: [outChannels] (optional, can be nil)
	// Output: [outChannels, outLen] or [batch, outChannels, outLen]
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes are incompatible.
	Conv1D(dst Tensor, kernel, bias Tensor, stride, padding int) Tensor

	// Conv2D performs 2D convolution (matches tf.nn.conv2d).
	// Input: [batch, inChannels, height, width]
	// Kernel: [outChannels, inChannels, kernelH, kernelW]
	// Bias: [outChannels] (optional, can be nil)
	// Stride: [strideH, strideW]
	// Padding: [padH, padW]
	// Output: [batch, outChannels, outHeight, outWidth]
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes are incompatible.
	Conv2D(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor

	// Conv2DTransposed performs transposed 2D convolution (deconvolution) (matches tf.nn.conv2d_transpose).
	// Input: [batch, inChannels, height, width]
	// Kernel: [inChannels, outChannels, kernelH, kernelW] (transposed layout)
	// Bias: [outChannels] (optional, can be nil)
	// Output: [batch, outChannels, outHeight, outWidth]
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes are incompatible.
	Conv2DTransposed(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor

	// Conv2DKernelGrad computes the gradient of the convolution kernel.
	// Used in backpropagation for training convolutional layers.
	// If dst is nil, creates a new tensor with kernel gradient.
	// If dst is provided, writes kernel gradient to dst and returns dst.
	Conv2DKernelGrad(dst Tensor, outputGrad, kernel Tensor, stride, padding []int) Tensor

	// Conv1DKernelGrad computes the gradient of the 1D convolution kernel.
	// Used in backpropagation for training 1D convolutional layers.
	// If dst is nil, creates a new tensor with kernel gradient.
	// If dst is provided, writes kernel gradient to dst and returns dst.
	Conv1DKernelGrad(dst Tensor, outputGrad, kernel Tensor, stride, padding int) Tensor

	// Image/Column Conversion
	// These operations convert between image patches and column format for efficient convolution computation.

	// Im2Col converts image patches to columns for GEMM-based convolution.
	// Input: [batch, channels, height, width]
	// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	// If dst is nil, creates a new tensor. Used internally for optimized convolution computation.
	// If dst is provided, writes result to dst and returns dst.
	Im2Col(dst Tensor, kernelSize, stride, padding []int) Tensor

	// Col2Im converts columns back to image (inverse of Im2Col).
	// Input: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	// Output: [batch, channels, height, width]
	// If dst is nil, creates a new tensor. Used in backpropagation for convolution gradients.
	// If dst is provided, writes result to dst and returns dst.
	Col2Im(dst Tensor, outputShape, kernelSize, stride, padding []int) Tensor
}

