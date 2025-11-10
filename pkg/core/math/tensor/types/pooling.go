package types

// Pooling defines pooling operations for neural networks.
// This interface contains operations for downsampling tensors through pooling.
type Pooling interface {
	// MaxPool2D performs max pooling operation.
	// Input: [batch, channels, height, width]
	// KernelSize: [kernelH, kernelW]
	// Stride: [strideH, strideW]
	// Padding: [padH, padW]
	// Output: [batch, channels, outHeight, outWidth]
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	// Panics if shapes are incompatible.
	MaxPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor

	// MaxPool2DWithIndices performs max pooling and returns both output and indices.
	// Input: [batch, channels, height, width]
	// KernelSize: [kernelH, kernelW]
	// Stride: [strideH, strideW]
	// Padding: [padH, padW]
	// Output: [batch, channels, outHeight, outWidth]
	// Indices: [batch, channels, outHeight, outWidth] (as int16, linear indices into input)
	// Returns: (output Tensor, indices Tensor)
	// If dst is nil, creates a new output tensor. If dst is provided, writes result to dst.
	// If indicesDst is nil, creates a new indices tensor. If indicesDst is provided, writes indices to indicesDst.
	// The indices are used for efficient backward pass computation.
	MaxPool2DWithIndices(dst Tensor, indicesDst Tensor, kernelSize, stride, padding []int) (Tensor, Tensor)

	// MaxPool2DBackward performs backward pass for max pooling using stored indices.
	// gradOutput: input gradient [batch, channels, outHeight, outWidth]
	// indices: indices from forward pass [batch, channels, outHeight, outWidth] (as int16)
	// kernelSize, stride, padding: pooling parameters
	// If dst is nil, creates a new tensor. If dst is provided, writes gradient to dst and returns dst.
	// Returns: gradient w.r.t. input [batch, channels, inHeight, inWidth]
	MaxPool2DBackward(dst Tensor, gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor

	// AvgPool2D performs average pooling operation.
	// Same signature as MaxPool2D.
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	AvgPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor

	// AvgPool2DBackward performs backward pass for average pooling.
	// gradOutput: input gradient [batch, channels, outHeight, outWidth]
	// kernelSize, stride, padding: pooling parameters
	// If dst is nil, creates a new tensor. If dst is provided, writes gradient to dst and returns dst.
	// Returns: gradient w.r.t. input [batch, channels, inHeight, inWidth]
	AvgPool2DBackward(dst Tensor, gradOutput Tensor, kernelSize, stride, padding []int) Tensor

	// GlobalAvgPool2D performs global average pooling.
	// Input: [batch, channels, height, width]
	// Output: [batch, channels]
	// Computes mean over spatial dimensions (height, width).
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	GlobalAvgPool2D(dst Tensor) Tensor

	// AdaptiveAvgPool2D performs adaptive average pooling to fixed output size.
	// Input: [batch, channels, height, width]
	// outputSize: [outHeight, outWidth] - target output spatial dimensions
	// Output: [batch, channels, outHeight, outWidth]
	// Divides input into approximately equal regions and averages each region.
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	AdaptiveAvgPool2D(dst Tensor, outputSize []int) Tensor
}
