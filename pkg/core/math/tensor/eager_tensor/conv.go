package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Conv2D performs 2D convolution (matches tf.nn.conv2d).
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [outChannels, inChannels, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// Output shape: [batch, outChannels, outHeight, outWidth]
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Conv2D(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	if t.shape == nil {
		return nil
	}
	if IsNil(kernel) {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	// Validate input shape: [batch, inChannels, height, width]
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2D: input must be 4D [batch, inChannels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	// Validate kernel shape: [outChannels, inChannels, kernelH, kernelW]
	if len(kernelShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2D: kernel must be 4D [outChannels, inChannels, kernelH, kernelW], got %v", kernelShape))
	}

	if kernelShape[1] != inChannels {
		panic(fmt.Sprintf("tensor.Conv2D: kernel inChannels %d doesn't match input inChannels %d", kernelShape[1], inChannels))
	}

	outChannels := kernelShape[0]
	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	// Validate stride and padding
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.Conv2D: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.Conv2D: padding must have 2 elements [padH, padW], got %v", padding))
	}

	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	// Calculate output dimensions
	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1

	switch t.Data().(type) {
	case []float32:
		// Extract raw data slices IMMEDIATELY - no tensor wrappers
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)

		// Handle bias - extract data or prepare for scalar handling
		var biasData []float32
		var biasScalar float32
		var hasScalarBias bool
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else if len(biasShape) == 0 {
				// Scalar bias - will apply after fp32 call
				biasRaw := types.GetTensorData[[]float32](bias)
				if len(biasRaw) > 0 {
					biasScalar = biasRaw[0]
					hasScalarBias = true
				}
			} else {
				panic(fmt.Sprintf("tensor.Conv2D: bias must be 1D [outChannels] or scalar, got %v", biasShape))
			}
		}

		// Calculate output shape
		outputShape := types.NewShape(batchSize, outChannels, outHeight, outWidth)

		// Create ONLY the final result tensor (or use dst)
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !dst.Shape().Equal(outputShape) {
				panic(fmt.Sprintf("tensor.Conv2D: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32.Conv2D DIRECTLY with raw data slices
		fp32.Conv2D(
			resultData,
			tData,
			kernelData,
			batchSize,
			inChannels,
			outChannels,
			inHeight,
			inWidth,
			outHeight,
			outWidth,
			kernelH,
			kernelW,
			strideH,
			strideW,
			padH,
			padW,
			biasData, // Can be nil if scalar bias
		)

		// Handle scalar bias if needed - apply directly to output
		if hasScalarBias && biasScalar != 0 {
			var stridesBuf [primitive.MAX_DIMS]int
			outputStrides := outputShape.Strides(stridesBuf[:len(outputShape)])
			fp32.ElemAddScalar(resultData, resultData, biasScalar,
				outputShape, outputStrides, outputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.Conv2D: unsupported data type: %T", t.Data()))
	}
}

// Conv2DTransposed performs transposed 2D convolution (deconvolution)
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [inChannels, outChannels, kernelH, kernelW] (transposed from Conv2D)
// Bias shape: [outChannels] (optional, can be nil)
// Output shape: [batch, outChannels, outHeight, outWidth]
// Uses fp32 primitive.Conv2DTransposed for optimized computation
func (t Tensor) Conv2DTransposed(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	// Validate input shape: [batch, inChannels, height, width]
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2DTransposed: input must be 4D [batch, inChannels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	// Validate kernel shape: [inChannels, outChannels, kernelH, kernelW]
	if len(kernelShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2DTransposed: kernel must be 4D [inChannels, outChannels, kernelH, kernelW], got %v", kernelShape))
	}

	if kernelShape[0] != inChannels {
		panic(fmt.Sprintf("tensor.Conv2DTransposed: kernel inChannels %d doesn't match input inChannels %d", kernelShape[0], inChannels))
	}

	outChannels := kernelShape[1]
	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	// Validate stride and padding
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.Conv2DTransposed: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.Conv2DTransposed: padding must have 2 elements [padH, padW], got %v", padding))
	}

	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	// Calculate base output dimensions for transposed convolution
	// outHeight = (inHeight - 1) * strideH - 2*padH + kernelH
	// outWidth = (inWidth - 1) * strideW - 2*padW + kernelW
	baseOutHeight := (inHeight-1)*strideH - 2*padH + kernelH
	baseOutWidth := (inWidth-1)*strideW - 2*padW + kernelW

	outputPadH := 0
	outputPadW := 0
	outHeight := baseOutHeight
	outWidth := baseOutWidth

	if !IsNil(dst) && dst.Shape() != nil {
		dstShape := dst.Shape()
		if len(dstShape) != 4 {
			panic(fmt.Sprintf("tensor.Conv2DTransposed: destination must be 4D [batch, channels, height, width], got %v", dstShape))
		}
		if dstShape[0] != batchSize {
			panic(fmt.Sprintf("tensor.Conv2DTransposed: destination batch %d doesn't match input batch %d", dstShape[0], batchSize))
		}
		if dstShape[1] != outChannels {
			panic(fmt.Sprintf("tensor.Conv2DTransposed: destination channels %d doesn't match expected %d", dstShape[1], outChannels))
		}

		padHDelta := dstShape[2] - baseOutHeight
		padWDelta := dstShape[3] - baseOutWidth

		if padHDelta < 0 || padWDelta < 0 {
			panic(fmt.Sprintf("tensor.Conv2DTransposed: destination spatial dims %v smaller than base output [%d %d]", dstShape[2:], baseOutHeight, baseOutWidth))
		}
		if padHDelta >= strideH || padWDelta >= strideW {
			panic(fmt.Sprintf("tensor.Conv2DTransposed: required output padding (%d, %d) must be less than stride (%d, %d)", padHDelta, padWDelta, strideH, strideW))
		}

		outputPadH = padHDelta
		outputPadW = padWDelta
		outHeight = baseOutHeight + outputPadH
		outWidth = baseOutWidth + outputPadW
	}

	switch t.Data().(type) {
	case []float32:
		// Extract raw data slices IMMEDIATELY - no tensor wrappers
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)

		// Handle bias - extract data or prepare for scalar handling
		var biasData []float32
		var biasScalar float32
		var hasScalarBias bool
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else if len(biasShape) == 0 {
				// Scalar bias - will apply after fp32 call
				biasRaw := types.GetTensorData[[]float32](bias)
				if len(biasRaw) > 0 {
					biasScalar = biasRaw[0]
					hasScalarBias = true
				}
			} else {
				panic(fmt.Sprintf("tensor.Conv2DTransposed: bias must be 1D [outChannels] or scalar, got %v", biasShape))
			}
		}

		// Calculate expected output shape
		expectedShape := types.NewShape(batchSize, outChannels, outHeight, outWidth)

		// Create ONLY the final result tensor (or use dst)
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), expectedShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !dst.Shape().Equal(expectedShape) {
				panic(fmt.Sprintf("tensor.Conv2DTransposed: destination shape mismatch: expected %v, got %v", expectedShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32 Conv2DTransposed primitive, extending with output padding when needed.
		if outputPadH == 0 && outputPadW == 0 {
			fp32.Conv2DTransposed(
				resultData,
				tData,
				kernelData,
				batchSize,
				inChannels,
				outChannels,
				inHeight,
				inWidth,
				outHeight,
				outWidth,
				kernelH,
				kernelW,
				strideH,
				strideW,
				padH,
				padW,
				biasData, // Can be nil if scalar bias
			)
		} else {
			fp32.Conv2DTransposedWithOutputPadding(
				resultData,
				tData,
				kernelData,
				batchSize,
				inChannels,
				outChannels,
				inHeight,
				inWidth,
				outHeight,
				outWidth,
				kernelH,
				kernelW,
				strideH,
				strideW,
				padH,
				padW,
				outputPadH,
				outputPadW,
				biasData,
			)
		}

		// Handle scalar bias if needed - apply directly to output
		if hasScalarBias && biasScalar != 0 {
			var stridesBuf [primitive.MAX_DIMS]int
			outputStrides := expectedShape.Strides(stridesBuf[:len(expectedShape)])
			fp32.ElemAddScalar(resultData, resultData, biasScalar,
				expectedShape, outputStrides, outputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.Conv2DTransposed: unsupported data type: %T", t.Data()))
	}
}

// Conv1D performs 1D convolution (simplified 2D convolution with width=1)
// Input shape: [batch, inChannels, length] or [inChannels, length]
// Kernel shape: [outChannels, inChannels, kernelLen]
// Bias shape: [outChannels] (optional)
// Output shape: [batch, outChannels, outLen] or [outChannels, outLen]
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
// This implementation works directly with data slices - no intermediate tensor objects.
func (t Tensor) Conv1D(dst types.Tensor, kernel, bias types.Tensor, stride, padding int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	// Handle both 2D [inChannels, length] and 3D [batch, inChannels, length] input
	var batchSize int
	var inChannels int
	var length int

	if len(tShape) == 2 {
		// [inChannels, length]
		inChannels = tShape[0]
		length = tShape[1]
		batchSize = 1
	} else if len(tShape) == 3 {
		// [batch, inChannels, length]
		batchSize = tShape[0]
		inChannels = tShape[1]
		length = tShape[2]
	} else {
		panic(fmt.Sprintf("tensor.Conv1D: input must be 2D [inChannels, length] or 3D [batch, inChannels, length], got %v", tShape))
	}

	// Validate kernel shape: [outChannels, inChannels, kernelLen]
	if len(kernelShape) != 3 {
		panic(fmt.Sprintf("tensor.Conv1D: kernel must be 3D [outChannels, inChannels, kernelLen], got %v", kernelShape))
	}

	if kernelShape[1] != inChannels {
		panic(fmt.Sprintf("tensor.Conv1D: kernel inChannels %d doesn't match input inChannels %d", kernelShape[1], inChannels))
	}

	outChannels := kernelShape[0]
	kernelLen := kernelShape[2]

	// Calculate output length
	outLen := (length+2*padding-kernelLen)/stride + 1

	switch t.Data().(type) {
	case []float32:
		// Extract raw data slices IMMEDIATELY - no tensor wrappers
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)

		// Handle bias - extract data or prepare for scalar handling
		var biasData []float32
		var biasScalar float32
		var hasScalarBias bool
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else if len(biasShape) == 0 {
				// Scalar bias - will apply after fp32 call
				biasRaw := types.GetTensorData[[]float32](bias)
				if len(biasRaw) > 0 {
					biasScalar = biasRaw[0]
					hasScalarBias = true
				}
			} else {
				panic(fmt.Sprintf("tensor.Conv1D: bias must be 1D [outChannels] or scalar, got %v", biasShape))
			}
		}

		// Calculate final output shape
		var finalShape types.Shape
		if len(tShape) == 2 {
			finalShape = types.NewShape(outChannels, outLen)
		} else {
			finalShape = types.NewShape(batchSize, outChannels, outLen)
		}

		// Create ONLY the final result tensor (or use dst)
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), finalShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !dst.Shape().Equal(finalShape) {
				panic(fmt.Sprintf("tensor.Conv1D: destination shape mismatch: expected %v, got %v", finalShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32.Conv2D DIRECTLY with 3D data treated as 4D (width=1)
		fp32.Conv2D(
			resultData, // Output: [batch, outChannels, outLen, 1] conceptually
			tData,      // Input: [batch, inChannels, length, 1] conceptually
			kernelData, // Kernel: [outChannels, inChannels, kernelLen, 1] conceptually
			batchSize,
			inChannels,
			outChannels,
			length,    // inHeight
			1,         // inWidth (treating 1D as 2D with width=1)
			outLen,    // outHeight
			1,         // outWidth
			kernelLen, // kernelH
			1,         // kernelW
			stride,    // strideH
			1,         // strideW
			padding,   // padH
			0,         // padW
			biasData,  // Can be nil if scalar bias
		)

		// Handle scalar bias if needed - apply directly to output
		if hasScalarBias && biasScalar != 0 {
			var stridesBuf [primitive.MAX_DIMS]int
			outputStrides := finalShape.Strides(stridesBuf[:len(finalShape)])
			fp32.ElemAddScalar(resultData, resultData, biasScalar,
				finalShape, outputStrides, outputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.Conv1D: unsupported data type: %T", t.Data()))
	}
}

// Conv2DKernelGrad computes kernel gradients for 2D convolution
// If dst is nil, creates a new kernel gradient tensor with the same shape as the kernel.
// If dst is provided, writes kernel gradient to dst and returns dst.
func (t Tensor) Conv2DKernelGrad(dst types.Tensor, outputGrad types.Tensor, kernel types.Tensor, stride, padding []int) types.Tensor {
	if t.shape == nil || outputGrad == nil || outputGrad.Shape() == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	outputGradShape := outputGrad.Shape()
	kernelShape := kernel.Shape()

	// Validate input shape: [batch, inChannels, height, width]
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: input must be 4D [batch, inChannels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	// Validate output gradient shape: [batch, outChannels, outHeight, outWidth]
	if len(outputGradShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: outputGrad must be 4D [batch, outChannels, outHeight, outWidth], got %v", outputGradShape))
	}

	if outputGradShape[0] != batchSize {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: outputGrad batch size %d doesn't match input batch size %d", outputGradShape[0], batchSize))
	}

	outChannels := outputGradShape[1]
	outHeight := outputGradShape[2]
	outWidth := outputGradShape[3]

	// Validate kernel shape: [outChannels, inChannels, kernelH, kernelW]
	if len(kernelShape) != 4 {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: kernel must be 4D [outChannels, inChannels, kernelH, kernelW], got %v", kernelShape))
	}

	if kernelShape[0] != outChannels || kernelShape[1] != inChannels {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: kernel shape %v doesn't match expected [outChannels=%d, inChannels=%d, kernelH, kernelW]", kernelShape, outChannels, inChannels))
	}

	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	// Validate stride and padding
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: padding must have 2 elements [padH, padW], got %v", padding))
	}

	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var kernelGradData []float32
		if IsNil(dst) {
			// Create kernel gradient tensor
			result = New(t.DataType(), kernelShape)
			kernelGradData = types.GetTensorData[[]float32](result)
		} else {
			// Validate dst shape matches kernel shape
			if !kernelShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.Conv2DKernelGrad: destination shape mismatch: expected %v, got %v", kernelShape, dst.Shape()))
			}
			result = dst
			kernelGradData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32.Conv2DKernelGrad
		tData := types.GetTensorData[[]float32](t)
		outputGradData := types.GetTensorData[[]float32](outputGrad)
		fp32.Conv2DKernelGrad(
			kernelGradData,
			tData,
			outputGradData,
			batchSize,
			inChannels,
			outChannels,
			inHeight,
			inWidth,
			outHeight,
			outWidth,
			kernelH,
			kernelW,
			strideH,
			strideW,
			padH,
			padW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.Conv2DKernelGrad: unsupported data type: %T", t.Data()))
	}
}

// Conv1DKernelGrad computes kernel gradients for 1D convolution
// If dst is nil, creates a new kernel gradient tensor with the same shape as the kernel.
// If dst is provided, writes kernel gradient to dst and returns dst.
func (t Tensor) Conv1DKernelGrad(dst types.Tensor, outputGrad types.Tensor, kernel types.Tensor, stride, padding int) types.Tensor {
	if t.shape == nil || outputGrad == nil || outputGrad.Shape() == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	outputGradShape := outputGrad.Shape()
	kernelShape := kernel.Shape()

	// Handle both 2D [inChannels, length] and 3D [batch, inChannels, length] input
	var batchSize int
	var inChannels int
	var inLength int

	if len(tShape) == 2 {
		// [inChannels, length]
		inChannels = tShape[0]
		inLength = tShape[1]
		batchSize = 1
	} else if len(tShape) == 3 {
		// [batch, inChannels, length]
		batchSize = tShape[0]
		inChannels = tShape[1]
		inLength = tShape[2]
	} else {
		panic(fmt.Sprintf("tensor.Conv1DKernelGrad: input must be 2D [inChannels, length] or 3D [batch, inChannels, length], got %v", tShape))
	}

	// Handle both 2D [outChannels, length] and 3D [batch, outChannels, length] output gradient
	var outChannels int
	var outLength int

	if len(outputGradShape) == 2 {
		// [outChannels, length]
		outChannels = outputGradShape[0]
		outLength = outputGradShape[1]
		if batchSize != 1 {
			panic(fmt.Sprintf("tensor.Conv1DKernelGrad: outputGrad batch size doesn't match input"))
		}
	} else if len(outputGradShape) == 3 {
		// [batch, outChannels, length]
		if outputGradShape[0] != batchSize {
			panic(fmt.Sprintf("tensor.Conv1DKernelGrad: outputGrad batch size %d doesn't match input batch size %d", outputGradShape[0], batchSize))
		}
		outChannels = outputGradShape[1]
		outLength = outputGradShape[2]
	} else {
		panic(fmt.Sprintf("tensor.Conv1DKernelGrad: outputGrad must be 2D [outChannels, length] or 3D [batch, outChannels, length], got %v", outputGradShape))
	}

	// Validate kernel shape: [outChannels, inChannels, kernelLen]
	if len(kernelShape) != 3 {
		panic(fmt.Sprintf("tensor.Conv1DKernelGrad: kernel must be 3D [outChannels, inChannels, kernelLen], got %v", kernelShape))
	}

	if kernelShape[0] != outChannels || kernelShape[1] != inChannels {
		panic(fmt.Sprintf("tensor.Conv1DKernelGrad: kernel shape %v doesn't match expected [outChannels=%d, inChannels=%d, kernelLen]", kernelShape, outChannels, inChannels))
	}

	kernelLen := kernelShape[2]

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var kernelGradData []float32
		if IsNil(dst) {
			// Create kernel gradient tensor
			result = New(t.DataType(), kernelShape)
			kernelGradData = types.GetTensorData[[]float32](result)
		} else {
			// Validate dst shape matches kernel shape
			if !kernelShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.Conv1DKernelGrad: destination shape mismatch: expected %v, got %v", kernelShape, dst.Shape()))
			}
			result = dst
			kernelGradData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32.Conv1DKernelGrad
		tData := types.GetTensorData[[]float32](t)
		outputGradData := types.GetTensorData[[]float32](outputGrad)
		fp32.Conv1DKernelGrad(
			kernelGradData,
			tData,
			outputGradData,
			batchSize,
			inChannels,
			outChannels,
			inLength,
			outLength,
			kernelLen,
			stride,
			padding,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.Conv1DKernelGrad: unsupported data type: %T", t.Data()))
	}
}

// Conv1DTransposed performs transposed 1D convolution (deconvolution)
// Input shape: [batch, inChannels, inLength]
// Kernel shape: [outChannels, inChannels, kernelLen] (forward kernel format)
// Output shape: [batch, inChannels, outLength] (note: outputs inChannels, not outChannels)
// This accepts the forward kernel format and internally transposes it
func (t Tensor) Conv1DTransposed(kernel types.Tensor, bias types.Tensor, stride, padding int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	// Handle both 2D [inChannels, length] and 3D [batch, inChannels, length] input
	var batchSize int
	var inChannels int
	var inLength int

	if len(tShape) == 2 {
		// [inChannels, length]
		inChannels = tShape[0]
		inLength = tShape[1]
		batchSize = 1
	} else if len(tShape) == 3 {
		// [batch, inChannels, length]
		batchSize = tShape[0]
		inChannels = tShape[1]
		inLength = tShape[2]
	} else {
		panic(fmt.Sprintf("tensor.Conv1DTransposed: input must be 2D [inChannels, length] or 3D [batch, inChannels, length], got %v", tShape))
	}

	// Validate kernel shape: [outChannels, inChannels, kernelLen] (forward format)
	if len(kernelShape) != 3 {
		panic(fmt.Sprintf("tensor.Conv1DTransposed: kernel must be 3D [outChannels, inChannels, kernelLen], got %v", kernelShape))
	}

	inChannelsKernel := kernelShape[1]
	if kernelShape[0] != inChannels {
		panic(fmt.Sprintf("tensor.Conv1DTransposed: kernel outChannels %d doesn't match input inChannels %d", kernelShape[0], inChannels))
	}

	kernelLen := kernelShape[2]

	// Calculate output length for transposed convolution
	// outLength = (inLength - 1) * stride - 2*padding + kernelLen
	outLength := (inLength-1)*stride - 2*padding + kernelLen

	switch t.Data().(type) {
	case []float32:
		// Extract raw data slices IMMEDIATELY - no tensor wrappers
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)

		// Handle bias - extract data or prepare for scalar handling
		var biasData []float32
		var biasScalar float32
		var hasScalarBias bool
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == inChannelsKernel {
				biasData = types.GetTensorData[[]float32](bias)
			} else if len(biasShape) == 0 {
				// Scalar bias - will apply after fp32 call
				biasRaw := types.GetTensorData[[]float32](bias)
				if len(biasRaw) > 0 {
					biasScalar = biasRaw[0]
					hasScalarBias = true
				}
			} else {
				panic(fmt.Sprintf("tensor.Conv1DTransposed: bias must be 1D [outChannels] or scalar, got %v", biasShape))
			}
		}

		// Calculate final output shape
		var finalShape types.Shape
		if len(tShape) == 2 {
			finalShape = types.NewShape(inChannelsKernel, outLength)
		} else {
			finalShape = types.NewShape(batchSize, inChannelsKernel, outLength)
		}

		// Create ONLY the final result tensor
		result := New(t.DataType(), finalShape)
		resultData := types.GetTensorData[[]float32](result)

		// Call fp32.Conv2DTransposed DIRECTLY with 3D data treated as 4D (width=1)
		fp32.Conv2DTransposed(
			resultData, // Output: [batch, inChannelsKernel, outLength, 1] conceptually
			tData,      // Input: [batch, inChannels, inLength, 1] conceptually
			kernelData, // Kernel: [inChannels, inChannelsKernel, kernelLen, 1] conceptually
			batchSize,
			inChannels,
			inChannelsKernel, // outChannels for transposed conv
			inLength,         // inHeight
			1,                // inWidth (treating 1D as 2D with width=1)
			outLength,        // outHeight
			1,                // outWidth
			kernelLen,        // kernelH
			1,                // kernelW
			stride,           // strideH
			1,                // strideW
			padding,          // padH
			0,                // padW
			biasData,         // Can be nil if scalar bias
		)

		// Handle scalar bias if needed - apply directly to output
		if hasScalarBias && biasScalar != 0 {
			var stridesBuf [primitive.MAX_DIMS]int
			outputStrides := finalShape.Strides(stridesBuf[:len(finalShape)])
			fp32.ElemAddScalar(resultData, resultData, biasScalar,
				finalShape, outputStrides, outputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.Conv1DTransposed: unsupported data type: %T", t.Data()))
	}
}

// MaxPool2D performs max pooling operation
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels, outHeight, outWidth]
func (t Tensor) MaxPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	tShape := t.Shape()
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.MaxPool2D: input must be 4D [batch, channels, height, width], got %v", tShape))
	}

	if len(kernelSize) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2D: kernelSize must have 2 elements [kernelH, kernelW], got %v", kernelSize))
	}
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2D: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2D: padding must have 2 elements [padH, padW], got %v", padding))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	// Calculate output dimensions
	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1
	outputShape := types.NewShape(batchSize, channels, outHeight, outWidth)

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !outputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.MaxPool2D: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Perform max pooling using fp32
		tData := types.GetTensorData[[]float32](t)
		fp32.MaxPool2D(
			resultData,
			tData,
			batchSize, channels, inHeight, inWidth,
			kernelH, kernelW, strideH, strideW, padH, padW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.MaxPool2D: unsupported data type: %T", t.Data()))
	}
}

// MaxPool2DWithIndices performs max pooling and returns both output and indices
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels, outHeight, outWidth]
// Indices shape: [batch, channels, outHeight, outWidth] as int32 (linear indices into input)
// Returns: (output Tensor, indices Tensor)
func (t Tensor) MaxPool2DWithIndices(dst types.Tensor, indicesDst types.Tensor, kernelSize, stride, padding []int) (types.Tensor, types.Tensor) {
	if t.shape == nil {
		return nil, nil
	}

	tShape := t.Shape()
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: input must be 4D [batch, channels, height, width], got %v", tShape))
	}

	if len(kernelSize) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: kernelSize must have 2 elements [kernelH, kernelW], got %v", kernelSize))
	}
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: padding must have 2 elements [padH, padW], got %v", padding))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	// Calculate output dimensions
	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1
	outputShape := types.NewShape(batchSize, channels, outHeight, outWidth)
	indicesShape := outputShape

	switch t.Data().(type) {
	case []float32:
		// Handle destination for output
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !outputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Handle destination for indices
		var indices types.Tensor
		var indicesDataInt32 []int32
		if IsNil(indicesDst) {
			indicesSize := batchSize * channels * outHeight * outWidth
			indicesDataInt32 = make([]int32, indicesSize)
			indices = FromArray(indicesShape, indicesDataInt32)
		} else {
			if !indicesShape.Equal(indicesDst.Shape()) {
				panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: indices destination shape mismatch: expected %v, got %v", indicesShape, indicesDst.Shape()))
			}
			indices = indicesDst
			indicesDataInt32 = types.GetTensorData[[]int32](indicesDst)
		}

		// Perform max pooling with indices using fp32
		tData := types.GetTensorData[[]float32](t)
		fp32.MaxPool2DWithIndices(
			resultData,
			tData,
			indicesDataInt32,
			batchSize, channels, inHeight, inWidth,
			kernelH, kernelW, strideH, strideW, padH, padW,
		)

		return result, indices
	default:
		panic(fmt.Sprintf("tensor.MaxPool2DWithIndices: unsupported data type: %T", t.Data()))
	}
}

// MaxPool2DBackward performs backward pass for max pooling using stored indices
// gradOutput: input gradient [batch, channels, outHeight, outWidth]
// indices: indices from forward pass [batch, channels, outHeight, outWidth] as int32
// kernelSize, stride, padding: pooling parameters
// Returns: gradient w.r.t. input [batch, channels, inHeight, inWidth]
func (t Tensor) MaxPool2DBackward(dst types.Tensor, gradOutput types.Tensor, indices types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	if t.shape == nil || gradOutput == nil || gradOutput.Shape() == nil || indices == nil || indices.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	gradOutputShape := gradOutput.Shape()
	indicesShape := indices.Shape()

	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: input must be 4D [batch, channels, height, width], got %v", tShape))
	}
	if len(gradOutputShape) != 4 {
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: gradOutput must be 4D [batch, channels, outHeight, outWidth], got %v", gradOutputShape))
	}
	if len(indicesShape) != 4 {
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: indices must be 4D [batch, channels, outHeight, outWidth], got %v", indicesShape))
	}

	if len(kernelSize) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: kernelSize must have 2 elements [kernelH, kernelW], got %v", kernelSize))
	}
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: padding must have 2 elements [padH, padW], got %v", padding))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	outHeight := gradOutputShape[2]
	outWidth := gradOutputShape[3]
	expectedShape := types.NewShape(batchSize, channels, inHeight, inWidth)

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var gradInputData []float32
		if IsNil(dst) {
			result = New(t.DataType(), expectedShape)
			gradInputData = types.GetTensorData[[]float32](result)
		} else {
			if !expectedShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.MaxPool2DBackward: destination shape mismatch: expected %v, got %v", expectedShape, dst.Shape()))
			}
			result = dst
			gradInputData = types.GetTensorData[[]float32](dst)
		}

		// Get data
		gradOutputData := types.GetTensorData[[]float32](gradOutput)
		indicesDataInt32 := types.GetTensorData[[]int32](indices)
		tData := types.GetTensorData[[]float32](t)

		// Call fp32.MaxPool2DBackward
		fp32.MaxPool2DBackward(
			gradInputData,
			gradOutputData,
			indicesDataInt32,
			tData,
			batchSize, channels, inHeight, inWidth,
			outHeight, outWidth,
			kernelH, kernelW, strideH, strideW, padH, padW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.MaxPool2DBackward: unsupported data type: %T", t.Data()))
	}
}

// AvgPool2D performs average pooling operation
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels, outHeight, outWidth]
func (t Tensor) AvgPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	tShape := t.Shape()
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.AvgPool2D: input must be 4D [batch, channels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1
	outputShape := types.NewShape(batchSize, channels, outHeight, outWidth)

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !outputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.AvgPool2D: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Perform average pooling using fp32
		tData := types.GetTensorData[[]float32](t)
		fp32.AvgPool2D(
			resultData,
			tData,
			batchSize, channels, inHeight, inWidth,
			kernelH, kernelW, strideH, strideW, padH, padW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.AvgPool2D: unsupported data type: %T", t.Data()))
	}
}

// AvgPool2DBackward performs backward pass for average pooling
// gradOutput: input gradient [batch, channels, outHeight, outWidth]
// kernelSize, stride, padding: pooling parameters
// Returns: gradient w.r.t. input [batch, channels, inHeight, inWidth]
func (t Tensor) AvgPool2DBackward(dst types.Tensor, gradOutput types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	if t.shape == nil || gradOutput == nil || gradOutput.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	gradOutputShape := gradOutput.Shape()

	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.AvgPool2DBackward: input must be 4D [batch, channels, height, width], got %v", tShape))
	}
	if len(gradOutputShape) != 4 {
		panic(fmt.Sprintf("tensor.AvgPool2DBackward: gradOutput must be 4D [batch, channels, outHeight, outWidth], got %v", gradOutputShape))
	}

	if len(kernelSize) != 2 {
		panic(fmt.Sprintf("tensor.AvgPool2DBackward: kernelSize must have 2 elements [kernelH, kernelW], got %v", kernelSize))
	}
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.AvgPool2DBackward: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.AvgPool2DBackward: padding must have 2 elements [padH, padW], got %v", padding))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	outHeight := gradOutputShape[2]
	outWidth := gradOutputShape[3]
	expectedShape := types.NewShape(batchSize, channels, inHeight, inWidth)

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var gradInputData []float32
		if IsNil(dst) {
			result = New(t.DataType(), expectedShape)
			gradInputData = types.GetTensorData[[]float32](result)
		} else {
			if !expectedShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.AvgPool2DBackward: destination shape mismatch: expected %v, got %v", expectedShape, dst.Shape()))
			}
			result = dst
			gradInputData = types.GetTensorData[[]float32](dst)
		}

		// Get data
		gradOutputData := types.GetTensorData[[]float32](gradOutput)

		// Call fp32.AvgPool2DBackward
		fp32.AvgPool2DBackward(
			gradInputData,
			gradOutputData,
			batchSize, channels, inHeight, inWidth,
			outHeight, outWidth,
			kernelH, kernelW, strideH, strideW, padH, padW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.AvgPool2DBackward: unsupported data type: %T", t.Data()))
	}
}

// GlobalAvgPool2D performs global average pooling
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels]
func (t Tensor) GlobalAvgPool2D(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	tShape := t.Shape()
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.GlobalAvgPool2D: input must be 4D [batch, channels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	height := tShape[2]
	width := tShape[3]
	outputShape := types.NewShape(batchSize, channels)

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !outputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.GlobalAvgPool2D: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Perform global average pooling using fp32
		tData := types.GetTensorData[[]float32](t)
		fp32.GlobalAvgPool2D(
			resultData,
			tData,
			batchSize, channels, height, width,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.GlobalAvgPool2D: unsupported data type: %T", t.Data()))
	}
}

// DepthwiseConv2D performs depthwise separable 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [inChannels, 1, kernelH, kernelW] or [inChannels, kernelH, kernelW]
// Bias shape: [inChannels] (optional, can be nil)
// Output shape: [batch, inChannels, outHeight, outWidth]
// Each input channel is convolved with its own kernel (depth multiplier = 1)
func (t Tensor) DepthwiseConv2D(kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.DepthwiseConv2D: input must be 4D [batch, inChannels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	// Kernel shape can be [inChannels, 1, kernelH, kernelW] or [inChannels, kernelH, kernelW]
	var kernelH, kernelW int
	if len(kernelShape) == 4 {
		if kernelShape[0] != inChannels || kernelShape[1] != 1 {
			panic(fmt.Sprintf("tensor.DepthwiseConv2D: kernel must be [inChannels, 1, kernelH, kernelW], got %v", kernelShape))
		}
		kernelH = kernelShape[2]
		kernelW = kernelShape[3]
	} else if len(kernelShape) == 3 {
		if kernelShape[0] != inChannels {
			panic(fmt.Sprintf("tensor.DepthwiseConv2D: kernel must be [inChannels, kernelH, kernelW], got %v", kernelShape))
		}
		kernelH = kernelShape[1]
		kernelW = kernelShape[2]
	} else {
		panic(fmt.Sprintf("tensor.DepthwiseConv2D: kernel must be 3D or 4D, got %v", kernelShape))
	}

	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.DepthwiseConv2D: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.DepthwiseConv2D: padding must have 2 elements [padH, padW], got %v", padding))
	}

	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1

	switch t.Data().(type) {
	case []float32:
		var biasData []float32
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == inChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else {
				panic(fmt.Sprintf("tensor.DepthwiseConv2D: bias must be 1D [inChannels], got %v", biasShape))
			}
		}

		result := New(t.DataType(), types.NewShape(batchSize, inChannels, outHeight, outWidth))
		resultPtr := &result

		// Convert kernel to 3D format [channels, kernelH, kernelW] for fp32 function
		kernelData := types.GetTensorData[[]float32](kernel)
		if len(kernelShape) == 4 {
			// Convert from [channels, 1, kernelH, kernelW] to [channels, kernelH, kernelW]
			kernel3D := make([]float32, inChannels*kernelH*kernelW)
			for c := 0; c < inChannels; c++ {
				dstOffset := c * kernelH * kernelW
				copy(kernel3D[dstOffset:dstOffset+kernelH*kernelW], kernelData[dstOffset:dstOffset+kernelH*kernelW])
			}
			kernelData = kernel3D
		}

		// Perform depthwise convolution using fp32
		resultData := types.GetTensorData[[]float32](resultPtr)
		tData := types.GetTensorData[[]float32](t)
		fp32.DepthwiseConv2D(
			resultData,
			tData,
			kernelData,
			biasData,
			batchSize, inChannels, inHeight, inWidth,
			kernelH, kernelW, strideH, strideW, padH, padW,
		)

		return resultPtr
	default:
		panic(fmt.Sprintf("tensor.DepthwiseConv2D: unsupported data type: %T", t.Data()))
	}
}

// GroupConv2D performs grouped 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [outChannels, inChannels/groups, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// groups: number of groups (inChannels must be divisible by groups)
// Output shape: [batch, outChannels, outHeight, outWidth]
func (t Tensor) GroupConv2D(kernel, bias types.Tensor, stride, padding []int, groups int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.GroupConv2D: input must be 4D [batch, inChannels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	if inChannels%groups != 0 {
		panic(fmt.Sprintf("tensor.GroupConv2D: inChannels %d must be divisible by groups %d", inChannels, groups))
	}

	channelsPerGroup := inChannels / groups

	if len(kernelShape) != 4 {
		panic(fmt.Sprintf("tensor.GroupConv2D: kernel must be 4D [outChannels, inChannels/groups, kernelH, kernelW], got %v", kernelShape))
	}

	outChannels := kernelShape[0]
	if kernelShape[1] != channelsPerGroup {
		panic(fmt.Sprintf("tensor.GroupConv2D: kernel inChannels/groups %d doesn't match expected %d", kernelShape[1], channelsPerGroup))
	}

	if outChannels%groups != 0 {
		panic(fmt.Sprintf("tensor.GroupConv2D: outChannels %d must be divisible by groups %d", outChannels, groups))
	}

	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.GroupConv2D: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.GroupConv2D: padding must have 2 elements [padH, padW], got %v", padding))
	}

	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1

	switch t.Data().(type) {
	case []float32:
		var biasData []float32
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else {
				panic(fmt.Sprintf("tensor.GroupConv2D: bias must be 1D [outChannels], got %v", biasShape))
			}
		}

		result := New(t.DataType(), types.NewShape(batchSize, outChannels, outHeight, outWidth))
		resultPtr := &result

		// Perform grouped convolution using fp32
		resultData := types.GetTensorData[[]float32](resultPtr)
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)
		fp32.GroupConv2D(
			resultData,
			tData,
			kernelData,
			biasData,
			batchSize, inChannels, outChannels, inHeight, inWidth,
			kernelH, kernelW, strideH, strideW, padH, padW, groups,
		)

		return resultPtr
	default:
		panic(fmt.Sprintf("tensor.GroupConv2D: unsupported data type: %T", t.Data()))
	}
}

// DilatedConv2D performs dilated (atrous) 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [outChannels, inChannels, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// dilation: [dilationH, dilationW] - dilation rates
// Output shape: [batch, outChannels, outHeight, outWidth]
func (t Tensor) DilatedConv2D(kernel, bias types.Tensor, stride, padding, dilation []int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.DilatedConv2D: input must be 4D [batch, inChannels, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	if len(kernelShape) != 4 {
		panic(fmt.Sprintf("tensor.DilatedConv2D: kernel must be 4D [outChannels, inChannels, kernelH, kernelW], got %v", kernelShape))
	}

	if kernelShape[1] != inChannels {
		panic(fmt.Sprintf("tensor.DilatedConv2D: kernel inChannels %d doesn't match input inChannels %d", kernelShape[1], inChannels))
	}

	outChannels := kernelShape[0]
	kernelH := kernelShape[2]
	kernelW := kernelShape[3]

	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.DilatedConv2D: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.DilatedConv2D: padding must have 2 elements [padH, padW], got %v", padding))
	}
	if len(dilation) != 2 {
		panic(fmt.Sprintf("tensor.DilatedConv2D: dilation must have 2 elements [dilationH, dilationW], got %v", dilation))
	}

	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]
	dilationH := dilation[0]
	dilationW := dilation[1]

	// Calculate effective kernel size with dilation
	effKernelH := (kernelH-1)*dilationH + 1
	effKernelW := (kernelW-1)*dilationW + 1

	outHeight := (inHeight+2*padH-effKernelH)/strideH + 1
	outWidth := (inWidth+2*padW-effKernelW)/strideW + 1

	switch t.Data().(type) {
	case []float32:
		var biasData []float32
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else {
				panic(fmt.Sprintf("tensor.DilatedConv2D: bias must be 1D [outChannels], got %v", biasShape))
			}
		}

		result := New(t.DataType(), types.NewShape(batchSize, outChannels, outHeight, outWidth))
		resultPtr := &result

		// Perform dilated convolution using fp32
		resultData := types.GetTensorData[[]float32](resultPtr)
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)
		fp32.DilatedConv2D(
			resultData,
			tData,
			kernelData,
			biasData,
			batchSize, inChannels, outChannels, inHeight, inWidth,
			kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW,
		)

		return resultPtr
	default:
		panic(fmt.Sprintf("tensor.DilatedConv2D: unsupported data type: %T", t.Data()))
	}
}

// Conv3D performs 3D convolution
// Input shape: [batch, inChannels, depth, height, width]
// Kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// Stride: [strideD, strideH, strideW]
// Padding: [padD, padH, padW]
// Output shape: [batch, outChannels, outDepth, outHeight, outWidth]
func (t Tensor) Conv3D(kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	if t.shape == nil || kernel == nil || kernel.Shape() == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	if len(tShape) != 5 {
		panic(fmt.Sprintf("tensor.Conv3D: input must be 5D [batch, inChannels, depth, height, width], got %v", tShape))
	}

	batchSize := tShape[0]
	inChannels := tShape[1]
	depth := tShape[2]
	height := tShape[3]
	width := tShape[4]

	if len(kernelShape) != 5 {
		panic(fmt.Sprintf("tensor.Conv3D: kernel must be 5D [outChannels, inChannels, kernelD, kernelH, kernelW], got %v", kernelShape))
	}

	if kernelShape[1] != inChannels {
		panic(fmt.Sprintf("tensor.Conv3D: kernel inChannels %d doesn't match input inChannels %d", kernelShape[1], inChannels))
	}

	outChannels := kernelShape[0]
	kernelD := kernelShape[2]
	kernelH := kernelShape[3]
	kernelW := kernelShape[4]

	if len(stride) != 3 {
		panic(fmt.Sprintf("tensor.Conv3D: stride must have 3 elements [strideD, strideH, strideW], got %v", stride))
	}
	if len(padding) != 3 {
		panic(fmt.Sprintf("tensor.Conv3D: padding must have 3 elements [padD, padH, padW], got %v", padding))
	}

	strideD := stride[0]
	strideH := stride[1]
	strideW := stride[2]
	padD := padding[0]
	padH := padding[1]
	padW := padding[2]

	outDepth := (depth+2*padD-kernelD)/strideD + 1
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	switch t.Data().(type) {
	case []float32:
		var biasData []float32
		if bias != nil && bias.Shape() != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outChannels {
				biasData = types.GetTensorData[[]float32](bias)
			} else {
				panic(fmt.Sprintf("tensor.Conv3D: bias must be 1D [outChannels], got %v", biasShape))
			}
		}

		result := New(t.DataType(), types.NewShape(batchSize, outChannels, outDepth, outHeight, outWidth))
		resultPtr := &result

		// Perform 3D convolution using fp32
		resultData := types.GetTensorData[[]float32](resultPtr)
		tData := types.GetTensorData[[]float32](t)
		kernelData := types.GetTensorData[[]float32](kernel)
		fp32.Conv3D(
			resultData,
			tData,
			kernelData,
			biasData,
			batchSize, inChannels, outChannels, depth, height, width,
			kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW,
		)

		return resultPtr
	default:
		panic(fmt.Sprintf("tensor.Conv3D: unsupported data type: %T", t.Data()))
	}
}

// AdaptiveAvgPool2D performs adaptive average pooling to a fixed output size
// Input shape: [batch, channels, height, width]
// outputSize: [outHeight, outWidth] - target output spatial dimensions
// Output shape: [batch, channels, outHeight, outWidth]
func (t Tensor) AdaptiveAvgPool2D(dst types.Tensor, outputSize []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	tShape := t.Shape()
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.AdaptiveAvgPool2D: input must be 4D [batch, channels, height, width], got %v", tShape))
	}

	if len(outputSize) != 2 {
		panic(fmt.Sprintf("tensor.AdaptiveAvgPool2D: outputSize must have 2 elements [outHeight, outWidth], got %v", outputSize))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	height := tShape[2]
	width := tShape[3]

	outHeight := outputSize[0]
	outWidth := outputSize[1]

	if outHeight <= 0 || outWidth <= 0 {
		panic(fmt.Sprintf("tensor.AdaptiveAvgPool2D: outputSize must be positive, got %v", outputSize))
	}

	outputShape := types.NewShape(batchSize, channels, outHeight, outWidth)

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			if !outputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.AdaptiveAvgPool2D: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Perform adaptive average pooling using fp32
		tData := types.GetTensorData[[]float32](t)
		fp32.AdaptiveAvgPool2D(
			resultData,
			tData,
			batchSize, channels, height, width, outHeight, outWidth,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.AdaptiveAvgPool2D: unsupported data type: %T", t.Data()))
	}
}

// Im2Col converts image patches to columns (for convolution)
// Input shape: [batch, channels, height, width]
// Output shape: [batch*outHeight*outWidth, channels*kernelH*kernelW]
// Uses fp32.Im2Col for optimized computation
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Im2Col(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	tShape := t.Shape()
	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.Im2Col: input must be 4D [batch, channels, height, width], got %v", tShape))
	}

	if len(kernelSize) != 2 {
		panic(fmt.Sprintf("tensor.Im2Col: kernelSize must have 2 elements [kernelH, kernelW], got %v", kernelSize))
	}
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.Im2Col: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.Im2Col: padding must have 2 elements [padH, padW], got %v", padding))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	height := tShape[2]
	width := tShape[3]

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	// Calculate output dimensions
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Output shape: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	colHeight := batchSize * outHeight * outWidth
	colWidth := channels * kernelH * kernelW
	outputShape := types.NewShape(colHeight, colWidth)

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), outputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			// Validate dst shape matches output shape
			if !outputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.Im2Col: destination shape mismatch: expected %v, got %v", outputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32.Im2Col
		tData := types.GetTensorData[[]float32](t)
		fp32.Im2Col(
			resultData,
			tData,
			batchSize,
			channels,
			height,
			width,
			kernelH,
			kernelW,
			padH,
			padW,
			strideH,
			strideW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.Im2Col: unsupported data type: %T", t.Data()))
	}
}

// Col2Im converts columns back to image (inverse of Im2Col)
// Input shape: [batch*outHeight*outWidth, channels*kernelH*kernelW]
// Output shape: [batch, channels, height, width]
// Uses fp32.Col2Im for optimized computation
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Col2Im(dst types.Tensor, outputShape, kernelSize, stride, padding []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	colShape := t.Shape()
	if len(colShape) != 2 {
		panic(fmt.Sprintf("tensor.Col2Im: input must be 2D [batch*outHeight*outWidth, channels*kernelH*kernelW], got %v", colShape))
	}

	if len(outputShape) != 4 {
		panic(fmt.Sprintf("tensor.Col2Im: outputShape must be 4D [batch, channels, height, width], got %v", outputShape))
	}
	if len(kernelSize) != 2 {
		panic(fmt.Sprintf("tensor.Col2Im: kernelSize must have 2 elements [kernelH, kernelW], got %v", kernelSize))
	}
	if len(stride) != 2 {
		panic(fmt.Sprintf("tensor.Col2Im: stride must have 2 elements [strideH, strideW], got %v", stride))
	}
	if len(padding) != 2 {
		panic(fmt.Sprintf("tensor.Col2Im: padding must have 2 elements [padH, padW], got %v", padding))
	}

	batchSize := outputShape[0]
	channels := outputShape[1]
	height := outputShape[2]
	width := outputShape[3]

	kernelH := kernelSize[0]
	kernelW := kernelSize[1]
	strideH := stride[0]
	strideW := stride[1]
	padH := padding[0]
	padW := padding[1]

	// Calculate output dimensions from input
	colHeight := colShape[0]
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Validate dimensions
	if colHeight != batchSize*outHeight*outWidth {
		panic(fmt.Sprintf("tensor.Col2Im: col height %d doesn't match expected %d", colHeight, batchSize*outHeight*outWidth))
	}

	expectedOutputShape := types.NewShape(batchSize, channels, height, width)

	switch t.Data().(type) {
	case []float32:
		// Handle destination
		var result types.Tensor
		var resultData []float32
		if IsNil(dst) {
			result = New(t.DataType(), expectedOutputShape)
			resultData = types.GetTensorData[[]float32](result)
		} else {
			// Validate dst shape matches expected output shape
			if !expectedOutputShape.Equal(dst.Shape()) {
				panic(fmt.Sprintf("tensor.Col2Im: destination shape mismatch: expected %v, got %v", expectedOutputShape, dst.Shape()))
			}
			result = dst
			resultData = types.GetTensorData[[]float32](dst)
		}

		// Call fp32.Col2Im (which clears the destination before accumulation)
		tData := types.GetTensorData[[]float32](t)
		fp32.Col2Im(
			resultData,
			tData,
			batchSize,
			channels,
			height,
			width,
			kernelH,
			kernelW,
			padH,
			padW,
			strideH,
			strideW,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.Col2Im: unsupported data type: %T", t.Data()))
	}
}

// ScatterAdd adds values to destination tensor at positions specified by indices
// dst: destination tensor (modified in-place). The slice is cleared inside this
// helper so callers do not need to zero pooled buffers manually.
// index: indices tensor [batch, channels, outHeight, outWidth] as int32 (linear indices into dst)
// value: values to add [batch, channels, outHeight, outWidth]
// For each position in index, adds the corresponding value from value to dst[index[i]]
// This is a general scatter operation useful for gradient routing in backpropagation
func (t Tensor) ScatterAdd(dst types.Tensor, index types.Tensor, value types.Tensor) types.Tensor {
	if t.shape == nil || dst == nil || dst.Shape() == nil || index == nil || index.Shape() == nil || value == nil || value.Shape() == nil {
		return nil
	}

	tShape := t.Shape()     // Source tensor shape
	dstShape := dst.Shape() // Destination tensor shape
	indexShape := index.Shape()
	valueShape := value.Shape()

	if len(tShape) != 4 {
		panic(fmt.Sprintf("tensor.ScatterAdd: source must be 4D [batch, channels, height, width], got %v", tShape))
	}
	if len(dstShape) != 4 {
		panic(fmt.Sprintf("tensor.ScatterAdd: destination must be 4D [batch, channels, inHeight, inWidth], got %v", dstShape))
	}
	if len(indexShape) != 4 {
		panic(fmt.Sprintf("tensor.ScatterAdd: index must be 4D [batch, channels, outHeight, outWidth], got %v", indexShape))
	}
	if len(valueShape) != 4 {
		panic(fmt.Sprintf("tensor.ScatterAdd: value must be 4D [batch, channels, outHeight, outWidth], got %v", valueShape))
	}

	batchSize := tShape[0]
	channels := tShape[1]
	inHeight := tShape[2]
	inWidth := tShape[3]

	outHeight := indexShape[2]
	outWidth := indexShape[3]

	switch t.Data().(type) {
	case []float32:
		// Get data
		dstData := types.GetTensorData[[]float32](dst)
		for i := range dstData {
			dstData[i] = 0
		}
		indexDataInt32 := types.GetTensorData[[]int32](index)
		valueData := types.GetTensorData[[]float32](value)

		// Call fp32.ScatterAdd
		fp32.ScatterAdd(
			dstData,
			indexDataInt32,
			valueData,
			batchSize, channels, inHeight, inWidth,
			outHeight, outWidth,
		)

		return dst
	default:
		panic(fmt.Sprintf("tensor.ScatterAdd: unsupported data type: %T", t.Data()))
	}
}

// Unpad removes padding from tensor
// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
// If dst is nil, creates a new tensor with padding removed.
// If dst is provided, copies unpadded data to dst and returns dst.
func (t Tensor) Unpad(dst types.Tensor, padding []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape()
	rank := shape.Rank()

	if len(padding) != 2*rank {
		panic(fmt.Sprintf("tensor.Unpad: padding must have %d elements (2 per dimension), got %d", 2*rank, len(padding)))
	}

	// Compute unpadded shape
	newShape := make(types.Shape, rank)
	for i := 0; i < rank; i++ {
		padBefore := padding[2*i]
		padAfter := padding[2*i+1]
		if padBefore < 0 || padAfter < 0 {
			panic(fmt.Sprintf("tensor.Unpad: padding values must be non-negative, got padBefore=%d, padAfter=%d for dim %d", padBefore, padAfter, i))
		}
		newShape[i] = shape[i] - padBefore - padAfter
		if newShape[i] <= 0 {
			panic(fmt.Sprintf("tensor.Unpad: unpadded dimension %d would be %d (invalid)", i, newShape[i]))
		}
	}

	if len(newShape) == 0 {
		if IsNil(dst) {
			result := New(t.DataType(), types.NewShape(newShape...))
			return &result
		}
		if !newShape.Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Unpad: destination shape mismatch: expected %v, got %v", newShape, dst.Shape()))
		}
		return dst
	}

	// Handle destination
	var result types.Tensor
	var resultData []float32
	if IsNil(dst) {
		// Create new tensor
		result = New(t.DataType(), types.NewShape(newShape...))
		resultData = types.GetTensorData[[]float32](result)
	} else {
		// Validate dst shape matches unpadded shape
		if !newShape.Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Unpad: destination shape mismatch: expected %v, got %v", newShape, dst.Shape()))
		}
		result = dst
		resultData = types.GetTensorData[[]float32](dst)
	}

	switch t.Data().(type) {
	case []float32:
		// Compute strides
		srcStrides := t.Strides(nil)
		dstStrides := types.NewShape(newShape...).Strides(nil)

		// Calculate source offset (skip padding at the beginning of each dimension)
		srcOffset := 0
		for i := 0; i < rank; i++ {
			padBefore := padding[2*i]
			srcOffset += padBefore * srcStrides[i]
		}

		// Get data slices starting from offset
		tData := types.GetTensorData[[]float32](t)

		// Create source view starting at offset
		srcView := tData[srcOffset:]
		fp32.ElemCopy(
			resultData,
			srcView,
			newShape.ToSlice(),
			dstStrides,
			srcStrides, // Use original strides (offset is handled by srcView)
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.Unpad: unsupported data type: %T", t.Data()))
	}
}

// Pad adds padding to tensor with constant value (matches tf.pad).
// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
// value: constant value to pad with
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Pad(dst types.Tensor, padding []int, value float64) types.Tensor {
	return t.PadTo(dst, padding, value)
}

// PadTo adds padding to tensor with constant value and stores result in dst.
// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
// value: constant value to pad with
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match output shape).
func (t Tensor) PadTo(dst types.Tensor, padding []int, value float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape()
	rank := shape.Rank()

	if len(padding) != 2*rank {
		panic(fmt.Sprintf("tensor.Pad: padding must have %d elements (2 per dimension), got %d", 2*rank, len(padding)))
	}

	// Compute padded shape
	paddedShape := make(types.Shape, rank)
	for i := 0; i < rank; i++ {
		padBefore := padding[2*i]
		padAfter := padding[2*i+1]
		if padBefore < 0 || padAfter < 0 {
			panic(fmt.Sprintf("tensor.Pad: padding values must be non-negative, got padBefore=%d, padAfter=%d for dim %d", padBefore, padAfter, i))
		}
		paddedShape[i] = shape[i] + padBefore + padAfter
		if paddedShape[i] <= 0 {
			panic(fmt.Sprintf("tensor.Pad: padded dimension %d would be %d (invalid)", i, paddedShape[i]))
		}
	}

	// Create or validate result tensor
	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = New(t.DataType(), types.NewShape(paddedShape...))
	} else {
		if !dst.Shape().Equal(types.NewShape(paddedShape...)) {
			panic("tensor.PadTo: destination shape mismatch")
		}
		result = dst
	}

	// Fill result with padding value
	result.Fill(result, value)

	// Copy input data to the appropriate position in result
	srcStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch t.Data().(type) {
	case []float32:
		// Calculate destination offset (skip padding at the beginning of each dimension)
		dstOffset := 0
		for i := 0; i < rank; i++ {
			padBefore := padding[2*i]
			dstOffset += padBefore * dstStrides[i]
		}

		// Get data slices
		resultData := types.GetTensorData[[]float32](result)
		tData := types.GetTensorData[[]float32](t)

		// Create destination view starting at offset
		dstView := resultData[dstOffset:]

		// Copy input data to padded region
		fp32.ElemCopy(
			dstView,
			tData,
			shape.ToSlice(),
			dstStrides,
			srcStrides,
		)

		return result
	default:
		panic(fmt.Sprintf("tensor.PadTo: unsupported data type: %T", t.Data()))
	}
}
