package fp32

import (
	"github.com/chewxy/math32"
)

// BatchNormForward performs batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
// Normalizes across batch dimension (axis 0). Assumes input shape is [batch, ...].
// gamma and beta are learnable parameters with shape matching the non-batch dimensions.
// If gamma/beta are nil, uses gamma=1, beta=0.
func BatchNormForward(dst, x, gamma, beta []float32, shape []int, eps float32) {
	if len(shape) < 2 {
		// Copy input if shape is too small for batch norm
		copy(dst, x)
		return
	}

	batchSize := shape[0]
	featureSize := SizeFromShape(shape[1:])

	// Use Pool for temporary arrays - much more efficient than stack allocation
	mean := Pool.Get(featureSize)
	defer Pool.Put(mean)
	variance := Pool.Get(featureSize)
	defer Pool.Put(variance)

	// Initialize to zero
	for i := range mean {
		mean[i] = 0
		variance[i] = 0
	}

	// Compute mean: sum over batch dimension for each feature
	for b := 0; b < batchSize; b++ {
		for f := 0; f < featureSize; f++ {
			idx := b*featureSize + f
			mean[f] += x[idx]
		}
	}
	for f := 0; f < featureSize; f++ {
		mean[f] /= float32(batchSize)
	}

	// Compute variance: sum of squared differences from mean
	for b := 0; b < batchSize; b++ {
		for f := 0; f < featureSize; f++ {
			idx := b*featureSize + f
			diff := x[idx] - mean[f]
			variance[f] += diff * diff
		}
	}
	for f := 0; f < featureSize; f++ {
		variance[f] /= float32(batchSize)
	}

	// Apply normalization: (x - mean) / sqrt(var + eps)
	for b := 0; b < batchSize; b++ {
		for f := 0; f < featureSize; f++ {
			idx := b*featureSize + f
			normalized := (x[idx] - mean[f]) / math32.Sqrt(variance[f]+eps)

			// Apply scale and shift if provided
			if gamma != nil && beta != nil {
				normalized = normalized*gamma[f] + beta[f]
			} else if gamma != nil {
				normalized = normalized * gamma[f]
			} else if beta != nil {
				normalized = normalized + beta[f]
			}

			dst[idx] = normalized
		}
	}
}

// LayerNormForward performs layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
// Normalizes across the last dimension (feature dimension). Assumes input is contiguous.
// gamma and beta are learnable parameters with shape matching the last dimension.
// If gamma/beta are nil, uses gamma=1, beta=0.
func LayerNormForward(dst, x, gamma, beta []float32, shape []int, eps float32) {
	if len(shape) == 0 {
		return
	}

	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	lastDim := shape[len(shape)-1]
	totalElements := size / lastDim // Number of normalization groups

	// For each normalization group (all elements except last dimension)
	for group := 0; group < totalElements; group++ {
		groupStart := group * lastDim

		// Compute mean for this group
		mean := float32(0)
		for i := 0; i < lastDim; i++ {
			mean += x[groupStart+i]
		}
		mean /= float32(lastDim)

		// Compute variance for this group
		variance := float32(0)
		for i := 0; i < lastDim; i++ {
			diff := x[groupStart+i] - mean
			variance += diff * diff
		}
		variance /= float32(lastDim)

		// Apply normalization
		invStd := 1.0 / math32.Sqrt(variance+eps)
		for i := 0; i < lastDim; i++ {
			idx := groupStart + i
			normalized := (x[idx] - mean) * invStd

			// Apply scale and shift if provided
			if gamma != nil && beta != nil {
				normalized = normalized*gamma[i] + beta[i]
			} else if gamma != nil {
				normalized = normalized * gamma[i]
			} else if beta != nil {
				normalized = normalized + beta[i]
			}

			dst[idx] = normalized
		}
	}
}

// RMSNormForward performs RMS normalization: x / sqrt(mean(x^2) + eps) * gamma
// Simpler than layer norm - only scales, no centering. Often used in transformers.
// gamma is a learnable parameter with shape matching the last dimension.
// If gamma is nil, uses gamma=1.
func RMSNormForward(dst, x, gamma []float32, shape []int, eps float32) {
	if len(shape) == 0 {
		return
	}

	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	lastDim := shape[len(shape)-1]
	totalElements := size / lastDim // Number of normalization groups

	// For each normalization group
	for group := 0; group < totalElements; group++ {
		groupStart := group * lastDim

		// Compute RMS (root mean square)
		rms := float32(0)
		for i := 0; i < lastDim; i++ {
			val := x[groupStart+i]
			rms += val * val
		}
		rms = math32.Sqrt(rms/float32(lastDim) + eps)

		// Apply normalization
		for i := 0; i < lastDim; i++ {
			idx := groupStart + i
			normalized := x[idx] / rms

			// Apply scale if provided
			if gamma != nil {
				normalized *= gamma[i]
			}

			dst[idx] = normalized
		}
	}
}

// L2NormForward performs L2 normalization: x / ||x||_2
// Normalizes each sample to unit L2 norm. Normalization axis can be specified.
// If axis < 0, normalizes the entire tensor as one vector.
func L2NormForward(dst, x []float32, shape []int, axis int) {
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if axis < 0 {
		// Normalize entire tensor as one vector
		norm := Nrm2(x, 1, size)
		if norm == 0 {
			copy(dst, x) // Avoid division by zero
			return
		}
		invNorm := 1.0 / norm
		for i := 0; i < size; i++ {
			dst[i] = x[i] * invNorm
		}
		return
	}

	if axis >= len(shape) {
		copy(dst, x) // Invalid axis
		return
	}

	// Normalize along specified axis
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	axisSize := shape[axis]
	innerSize := 1
	for i := axis + 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			// Compute L2 norm for this slice
			norm := float32(0)
			baseIdx := outer*axisSize*innerSize + inner
			for a := 0; a < axisSize; a++ {
				idx := baseIdx + a*innerSize
				val := x[idx]
				norm += val * val
			}
			norm = math32.Sqrt(norm)

			if norm == 0 {
				// Copy without normalization to avoid division by zero
				for a := 0; a < axisSize; a++ {
					idx := baseIdx + a*innerSize
					dst[idx] = x[idx]
				}
			} else {
				invNorm := 1.0 / norm
				for a := 0; a < axisSize; a++ {
					idx := baseIdx + a*innerSize
					dst[idx] = x[idx] * invNorm
				}
			}
		}
	}
}

// InstanceNorm2D performs instance normalization for 2D feature maps.
// Normalizes across spatial dimensions (H, W) for each instance and channel.
// Input shape: [batch, channels, height, width]
// gamma/beta shape: [channels] (one per channel)
func InstanceNorm2D(dst, x, gamma, beta []float32, batchSize, channels, height, width int, eps float32) {
	featureSize := height * width // Spatial dimensions to normalize over

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			channelStart := b*channels*height*width + c*height*width

			// Compute mean for this instance/channel
			mean := float32(0)
			for i := 0; i < featureSize; i++ {
				mean += x[channelStart+i]
			}
			mean /= float32(featureSize)

			// Compute variance for this instance/channel
			variance := float32(0)
			for i := 0; i < featureSize; i++ {
				diff := x[channelStart+i] - mean
				variance += diff * diff
			}
			variance /= float32(featureSize)

			// Apply normalization
			invStd := 1.0 / math32.Sqrt(variance+eps)
			scale := float32(1.0)
			shift := float32(0.0)

			if gamma != nil {
				scale = gamma[c]
			}
			if beta != nil {
				shift = beta[c]
			}

			for i := 0; i < featureSize; i++ {
				idx := channelStart + i
				normalized := (x[idx] - mean) * invStd
				dst[idx] = normalized*scale + shift
			}
		}
	}
}

// GroupNormForward performs group normalization.
// Divides channels into groups and normalizes within each group.
// Input shape: [batch, channels, ...] where channels must be divisible by numGroups.
// gamma/beta shape: [channels] (one per channel)
func GroupNormForward(dst, x, gamma, beta []float32, shape []int, numGroups int, eps float32) {
	if len(shape) < 2 || numGroups <= 0 {
		copy(dst, x)
		return
	}

	batchSize := shape[0]
	channels := shape[1]
	if channels%numGroups != 0 {
		copy(dst, x) // Invalid group configuration
		return
	}

	channelsPerGroup := channels / numGroups
	spatialSize := SizeFromShape(shape[2:]) // H * W * D * ...

	for b := 0; b < batchSize; b++ {
		for g := 0; g < numGroups; g++ {
			groupStartChannel := g * channelsPerGroup

			// Compute mean and variance for this group
			mean := float32(0)
			variance := float32(0)
			groupElements := 0

			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				for i := 0; i < spatialSize; i++ {
					val := x[channelStart+i]
					mean += val
					groupElements++
				}
			}
			mean /= float32(groupElements)

			// Compute variance
			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				for i := 0; i < spatialSize; i++ {
					val := x[channelStart+i]
					diff := val - mean
					variance += diff * diff
				}
			}
			variance /= float32(groupElements)

			// Apply normalization
			invStd := 1.0 / math32.Sqrt(variance+eps)
			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				scale := float32(1.0)
				shift := float32(0.0)
				if gamma != nil {
					scale = gamma[channelIdx]
				}
				if beta != nil {
					shift = beta[channelIdx]
				}

				for i := 0; i < spatialSize; i++ {
					idx := channelStart + i
					normalized := (x[idx] - mean) * invStd
					dst[idx] = normalized*scale + shift
				}
			}
		}
	}
}

// BatchNormGrad computes gradients for batch normalization.
// gradInput: destination for input gradient [batch, ...]
// gradGamma: destination for gamma gradient [...]
// gradBeta: destination for beta gradient [...]
// gradOutput: gradient w.r.t. output [batch, ...]
// input: original input [batch, ...]
// gamma: scale parameter [...]
func BatchNormGrad(gradInput, gradGamma, gradBeta, gradOutput, input, gamma []float32, shape []int, eps float32) {
	if len(shape) < 2 {
		// Zero out gradients for invalid shapes
		size := SizeFromShape(shape)
		for i := 0; i < size; i++ {
			gradInput[i] = 0
		}
		return
	}

	batchSize := shape[0]
	featureSize := SizeFromShape(shape[1:])

	// Use Pool for temporary arrays - much more efficient than stack allocation
	mean := Pool.Get(featureSize)
	defer Pool.Put(mean)
	variance := Pool.Get(featureSize)
	defer Pool.Put(variance)

	// Initialize to zero
	for i := range mean {
		mean[i] = 0
		variance[i] = 0
	}

	// Compute mean
	for b := 0; b < batchSize; b++ {
		for f := 0; f < featureSize; f++ {
			idx := b*featureSize + f
			mean[f] += input[idx]
		}
	}
	for f := 0; f < featureSize; f++ {
		mean[f] /= float32(batchSize)
	}

	// Compute variance
	for b := 0; b < batchSize; b++ {
		for f := 0; f < featureSize; f++ {
			idx := b*featureSize + f
			diff := input[idx] - mean[f]
			variance[f] += diff * diff
		}
	}
	for f := 0; f < featureSize; f++ {
		variance[f] /= float32(batchSize)
	}

	// Compute gradients
	for f := 0; f < featureSize; f++ {
		std := math32.Sqrt(variance[f] + eps)
		invStd := 1.0 / std

		// Compute intermediate values
		var sumGradOutput float32 = 0
		var sumGradOutputTimesXHat float32 = 0

		for b := 0; b < batchSize; b++ {
			idx := b*featureSize + f
			xHat := (input[idx] - mean[f]) * invStd
			sumGradOutput += gradOutput[idx]
			sumGradOutputTimesXHat += gradOutput[idx] * xHat
		}

		// Compute gradGamma and gradBeta
		if gamma != nil && gradGamma != nil && gradBeta != nil {
			gradGamma[f] = sumGradOutputTimesXHat
			gradBeta[f] = sumGradOutput
		}

		// Compute gradInput
		for b := 0; b < batchSize; b++ {
			idx := b*featureSize + f
			xHat := (input[idx] - mean[f]) * invStd

			var dXHat float32
			if gamma != nil {
				dXHat = gradOutput[idx] * gamma[f]
			} else {
				dXHat = gradOutput[idx]
			}

			// Gradient w.r.t. input
			gradInput[idx] = dXHat*invStd - (sumGradOutputTimesXHat*xHat+sumGradOutput)*invStd/float32(batchSize)
		}
	}
}

// LayerNormGrad computes gradients for layer normalization.
// gradInput: destination for input gradient [...]
// gradGamma: destination for gamma gradient [last_dim]
// gradBeta: destination for beta gradient [last_dim]
// gradOutput: gradient w.r.t. output [...]
// input: original input [...]
// gamma: scale parameter [last_dim]
func LayerNormGrad(gradInput, gradGamma, gradBeta, gradOutput, input, gamma []float32, shape []int, eps float32) {
	if len(shape) == 0 {
		return
	}

	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	lastDim := shape[len(shape)-1]
	totalElements := size / lastDim // Number of normalization groups

	// For each normalization group
	for group := 0; group < totalElements; group++ {
		groupStart := group * lastDim

		// Compute mean and variance (same as forward)
		mean := float32(0)
		variance := float32(0)
		for i := 0; i < lastDim; i++ {
			mean += input[groupStart+i]
		}
		mean /= float32(lastDim)

		for i := 0; i < lastDim; i++ {
			diff := input[groupStart+i] - mean
			variance += diff * diff
		}
		variance /= float32(lastDim)

		std := math32.Sqrt(variance + eps)
		invStd := 1.0 / std

		// Compute intermediate sums
		var sumGradOutput float32 = 0
		var sumGradOutputTimesXHat float32 = 0

		for i := 0; i < lastDim; i++ {
			idx := groupStart + i
			xHat := (input[idx] - mean) * invStd
			sumGradOutput += gradOutput[idx]
			sumGradOutputTimesXHat += gradOutput[idx] * xHat
		}

		// Compute gradGamma and gradBeta
		if gamma != nil && gradGamma != nil && gradBeta != nil {
			for i := 0; i < lastDim; i++ {
				xHat := (input[groupStart+i] - mean) * invStd
				gradGamma[i] += gradOutput[groupStart+i] * xHat
				gradBeta[i] += gradOutput[groupStart+i]
			}
		}

		// Compute gradInput
		for i := 0; i < lastDim; i++ {
			idx := groupStart + i
			xHat := (input[idx] - mean) * invStd

			var dXHat float32
			if gamma != nil {
				dXHat = gradOutput[idx] * gamma[i]
			} else {
				dXHat = gradOutput[idx]
			}

			// Gradient w.r.t. input
			gradInput[idx] = dXHat*invStd - (sumGradOutputTimesXHat*xHat+sumGradOutput)*invStd/float32(lastDim)
		}
	}
}

// RMSNormGrad computes gradients for RMS normalization.
// gradInput: destination for input gradient [...]
// gradGamma: destination for gamma gradient [last_dim]
// gradOutput: gradient w.r.t. output [...]
// input: original input [...]
// gamma: scale parameter [last_dim]
func RMSNormGrad(gradInput, gradGamma, gradOutput, input, gamma []float32, shape []int, eps float32) {
	if len(shape) == 0 {
		return
	}

	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	lastDim := shape[len(shape)-1]
	totalElements := size / lastDim // Number of normalization groups

	// For each normalization group
	for group := 0; group < totalElements; group++ {
		groupStart := group * lastDim

		// Compute RMS (same as forward)
		rms := float32(0)
		for i := 0; i < lastDim; i++ {
			val := input[groupStart+i]
			rms += val * val
		}
		rms = math32.Sqrt(rms/float32(lastDim) + eps)

		// Compute gradient
		for i := 0; i < lastDim; i++ {
			idx := groupStart + i
			normalized := input[idx] / rms

			if gamma != nil && gradGamma != nil {
				gradInput[idx] = gradOutput[idx] * gamma[i] / rms
				gradGamma[i] += gradOutput[idx] * normalized
			} else {
				gradInput[idx] = gradOutput[idx] / rms
				if gradGamma != nil {
					gradGamma[i] += gradOutput[idx] * normalized
				}
			}

			// Add the derivative w.r.t. RMS
			sumGradTimesX := float32(0)
			for j := 0; j < lastDim; j++ {
				sumGradTimesX += gradOutput[groupStart+j] * input[groupStart+j]
			}
			gradInput[idx] -= (sumGradTimesX * input[idx]) / (rms * rms * rms * float32(lastDim))
		}
	}
}

// InstanceNorm2DGrad computes gradients for 2D instance normalization.
// gradInput: destination for input gradient [batch, channels, height, width]
// gradGamma: destination for gamma gradient [channels]
// gradBeta: destination for beta gradient [channels]
// gradOutput: gradient w.r.t. output [batch, channels, height, width]
// input: original input [batch, channels, height, width]
// gamma: scale parameter [channels]
func InstanceNorm2DGrad(gradInput, gradGamma, gradBeta, gradOutput, input, gamma []float32, batchSize, channels, height, width int, eps float32) {
	featureSize := height * width // Spatial dimensions to normalize over

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			channelStart := b*channels*height*width + c*height*width

			// Compute mean and variance (same as forward)
			mean := float32(0)
			variance := float32(0)
			for i := 0; i < featureSize; i++ {
				mean += input[channelStart+i]
			}
			mean /= float32(featureSize)

			for i := 0; i < featureSize; i++ {
				diff := input[channelStart+i] - mean
				variance += diff * diff
			}
			variance /= float32(featureSize)

			std := math32.Sqrt(variance + eps)
			invStd := 1.0 / std

			// Compute intermediate sums
			var sumGradOutput float32 = 0
			var sumGradOutputTimesXHat float32 = 0

			for i := 0; i < featureSize; i++ {
				idx := channelStart + i
				xHat := (input[idx] - mean) * invStd
				sumGradOutput += gradOutput[idx]
				sumGradOutputTimesXHat += gradOutput[idx] * xHat
			}

			// Compute gradGamma and gradBeta
			if gamma != nil && gradGamma != nil && gradBeta != nil {
				gradGamma[c] += sumGradOutputTimesXHat
				gradBeta[c] += sumGradOutput
			}

			// Compute gradInput
			for i := 0; i < featureSize; i++ {
				idx := channelStart + i
				xHat := (input[idx] - mean) * invStd

				var dXHat float32
				if gamma != nil {
					dXHat = gradOutput[idx] * gamma[c]
				} else {
					dXHat = gradOutput[idx]
				}

				// Gradient w.r.t. input
				gradInput[idx] = dXHat*invStd - (sumGradOutputTimesXHat*xHat+sumGradOutput)*invStd/float32(featureSize)
			}
		}
	}
}

// GroupNormGrad computes gradients for group normalization.
// gradInput: destination for input gradient [batch, channels, ...]
// gradGamma: destination for gamma gradient [channels]
// gradBeta: destination for beta gradient [channels]
// gradOutput: gradient w.r.t. output [batch, channels, ...]
// input: original input [batch, channels, ...]
// gamma: scale parameter [channels]
// numGroups: number of groups used in forward pass
func GroupNormGrad(gradInput, gradGamma, gradBeta, gradOutput, input, gamma []float32, shape []int, numGroups int, eps float32) {
	if len(shape) < 2 || numGroups <= 0 {
		return
	}

	batchSize := shape[0]
	channels := shape[1]
	if channels%numGroups != 0 {
		return // Invalid group configuration
	}

	channelsPerGroup := channels / numGroups
	spatialSize := SizeFromShape(shape[2:]) // H * W * D * ...

	for b := 0; b < batchSize; b++ {
		for g := 0; g < numGroups; g++ {
			groupStartChannel := g * channelsPerGroup

			// Compute mean and variance (same as forward)
			mean := float32(0)
			variance := float32(0)
			groupElements := 0

			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				for i := 0; i < spatialSize; i++ {
					val := input[channelStart+i]
					mean += val
					groupElements++
				}
			}
			mean /= float32(groupElements)

			// Compute variance
			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				for i := 0; i < spatialSize; i++ {
					val := input[channelStart+i]
					diff := val - mean
					variance += diff * diff
				}
			}
			variance /= float32(groupElements)

			std := math32.Sqrt(variance + eps)
			invStd := 1.0 / std

			// Compute intermediate sums for this group
			var sumGradOutput float32 = 0
			var sumGradOutputTimesXHat float32 = 0

			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				for i := 0; i < spatialSize; i++ {
					idx := channelStart + i
					xHat := (input[idx] - mean) * invStd
					sumGradOutput += gradOutput[idx]
					sumGradOutputTimesXHat += gradOutput[idx] * xHat
				}
			}

			// Compute gradGamma and gradBeta for this group
			if gamma != nil && gradGamma != nil && gradBeta != nil {
				for c := 0; c < channelsPerGroup; c++ {
					channelIdx := groupStartChannel + c
					gradGamma[channelIdx] += sumGradOutputTimesXHat
					gradBeta[channelIdx] += sumGradOutput
				}
			}

			// Compute gradInput for this group
			for c := 0; c < channelsPerGroup; c++ {
				channelIdx := groupStartChannel + c
				channelStart := b*channels*spatialSize + channelIdx*spatialSize

				for i := 0; i < spatialSize; i++ {
					idx := channelStart + i
					xHat := (input[idx] - mean) * invStd

					var dXHat float32
					if gamma != nil {
						dXHat = gradOutput[idx] * gamma[channelIdx]
					} else {
						dXHat = gradOutput[idx]
					}

					// Gradient w.r.t. input
					gradInput[idx] = dXHat*invStd - (sumGradOutputTimesXHat*xHat+sumGradOutput)*invStd/float32(groupElements)
				}
			}
		}
	}
}
