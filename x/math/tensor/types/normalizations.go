package types

// Normalizations defines normalization operations commonly used in neural networks.
// This interface contains operations for stabilizing training and improving convergence.
type Normalizations interface {
	// Batch Normalization Operations

	// BatchNormForward performs batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
	// Normalizes across batch dimension (axis 0). Assumes input shape is [batch, ...].
	// gamma and beta are learnable parameters with shape matching the non-batch dimensions.
	// If gamma/beta are nil, uses gamma=1, beta=0.
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	BatchNormForward(dst Tensor, gamma, beta Tensor, eps float64) Tensor

	// BatchNormGrad computes gradients for batch normalization.
	// gradInputDst: destination for input gradient [batch, ...] (can be nil)
	// gradGammaDst: destination for gamma gradient [...] (can be nil)
	// gradBetaDst: destination for beta gradient [...] (can be nil)
	// gradOutput: gradient w.r.t. output [batch, ...]
	// input: original input [batch, ...]
	// gamma: scale parameter [...]
	// eps: epsilon for numerical stability
	// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
	BatchNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma Tensor, eps float64) (Tensor, Tensor, Tensor)

	// Layer Normalization Operations

	// LayerNormForward performs layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
	// Normalizes across the last dimension (feature dimension). Assumes input is contiguous.
	// gamma and beta are learnable parameters with shape matching the last dimension.
	// If gamma/beta are nil, uses gamma=1, beta=0.
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	LayerNormForward(dst Tensor, gamma, beta Tensor, eps float64) Tensor

	// LayerNormGrad computes gradients for layer normalization.
	// gradInputDst: destination for input gradient [...] (can be nil)
	// gradGammaDst: destination for gamma gradient [last_dim] (can be nil)
	// gradBetaDst: destination for beta gradient [last_dim] (can be nil)
	// gradOutput: gradient w.r.t. output [...]
	// input: original input [...]
	// gamma: scale parameter [last_dim]
	// eps: epsilon for numerical stability
	// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
	LayerNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma Tensor, eps float64) (Tensor, Tensor, Tensor)

	// RMS Normalization Operations

	// RMSNormForward performs RMS normalization: x / sqrt(mean(x^2) + eps) * gamma
	// Simpler than layer norm - only scales, no centering. Often used in transformers.
	// gamma is a learnable parameter with shape matching the last dimension.
	// If gamma is nil, uses gamma=1.
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	RMSNormForward(dst Tensor, gamma Tensor, eps float64) Tensor

	// RMSNormGrad computes gradients for RMS normalization.
	// gradInputDst: destination for input gradient [...] (can be nil)
	// gradGammaDst: destination for gamma gradient [last_dim] (can be nil)
	// gradOutput: gradient w.r.t. output [...]
	// input: original input [...]
	// gamma: scale parameter [last_dim]
	// eps: epsilon for numerical stability
	// Returns: (gradInput, gradGamma) - new tensors if dst was nil, otherwise returns dst tensors
	RMSNormGrad(gradInputDst, gradGammaDst, gradOutput, input, gamma Tensor, eps float64) (Tensor, Tensor)

	// L2 Normalization Operations
	// Note: L2Normalize is implemented in TensorMath interface

	// Instance Normalization Operations

	// InstanceNorm2D performs instance normalization for 2D feature maps.
	// Normalizes across spatial dimensions (H, W) for each instance and channel.
	// Input shape: [batch, channels, height, width]
	// gamma/beta shape: [channels] (one per channel)
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	InstanceNorm2D(dst Tensor, gamma, beta Tensor, eps float64) Tensor

	// InstanceNorm2DGrad computes gradients for 2D instance normalization.
	// gradInputDst: destination for input gradient [batch, channels, height, width] (can be nil)
	// gradGammaDst: destination for gamma gradient [channels] (can be nil)
	// gradBetaDst: destination for beta gradient [channels] (can be nil)
	// gradOutput: gradient w.r.t. output [batch, channels, height, width]
	// input: original input [batch, channels, height, width]
	// gamma: scale parameter [channels]
	// eps: epsilon for numerical stability
	// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
	InstanceNorm2DGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma Tensor, eps float64) (Tensor, Tensor, Tensor)

	// Group Normalization Operations

	// GroupNormForward performs group normalization.
	// Divides channels into groups and normalizes within each group.
	// Input shape: [batch, channels, ...] where channels must be divisible by numGroups.
	// gamma/beta shape: [channels] (one per channel)
	// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
	GroupNormForward(dst Tensor, gamma, beta Tensor, numGroups int, eps float64) Tensor

	// GroupNormGrad computes gradients for group normalization.
	// gradInputDst: destination for input gradient [batch, channels, ...] (can be nil)
	// gradGammaDst: destination for gamma gradient [channels] (can be nil)
	// gradBetaDst: destination for beta gradient [channels] (can be nil)
	// gradOutput: gradient w.r.t. output [batch, channels, ...]
	// input: original input [batch, channels, ...]
	// gamma: scale parameter [channels]
	// numGroups: number of groups used in forward pass
	// eps: epsilon for numerical stability
	// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
	GroupNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma Tensor, numGroups int, eps float64) (Tensor, Tensor, Tensor)
}
