package tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// Conv2D performs 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [outChannels, inChannels, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// Output shape: [batch, outChannels, outHeight, outWidth]
// Uses fp32 primitive.Conv2D for optimized computation
func (t *Tensor) Conv2D(kernel, bias *Tensor, stride, padding []int) *Tensor {
	if t == nil || kernel == nil {
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

	// Validate bias if provided
	var biasData []float32
	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outChannels {
			biasData = bias.Data()
		} else if len(biasShape) == 0 {
			// Scalar bias - expand to [outChannels]
			biasData = make([]float32, outChannels)
			val := bias.Data()[0]
			for i := range biasData {
				biasData[i] = val
			}
		} else {
			panic(fmt.Sprintf("tensor.Conv2D: bias must be 1D [outChannels] or scalar, got %v", biasShape))
		}
	}

	// Create output tensor
	result := New(t.dtype, NewShape(batchSize, outChannels, outHeight, outWidth))

	// Call fp32.Conv2D
	fp32.Conv2D(
		result.data,
		t.data,
		kernel.data,
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
		biasData,
	)

	return result
}

// Conv2DTo performs 2D convolution and stores result in dst (or creates new tensor if dst is nil)
func (t *Tensor) Conv2DTo(kernel, bias, dst *Tensor, stride, padding []int) *Tensor {
	if t == nil || kernel == nil {
		return nil
	}

	result := t.Conv2D(kernel, bias, stride, padding)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.sameShape(dst) {
		panic(fmt.Sprintf("tensor.Conv2DTo: destination shape mismatch: %v vs %v", dst.Shape(), result.Shape()))
	}

	result.copyTo(dst)
	return dst
}

// Conv2DTransposed performs transposed 2D convolution (deconvolution)
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [inChannels, outChannels, kernelH, kernelW] (transposed from Conv2D)
// Bias shape: [outChannels] (optional, can be nil)
// Output shape: [batch, outChannels, outHeight, outWidth]
// Uses fp32 primitive.Conv2DTransposed for optimized computation
func (t *Tensor) Conv2DTransposed(kernel, bias *Tensor, stride, padding []int) *Tensor {
	if t == nil || kernel == nil {
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

	// Calculate output dimensions for transposed convolution
	// outHeight = (inHeight - 1) * strideH - 2*padH + kernelH
	// outWidth = (inWidth - 1) * strideW - 2*padW + kernelW
	outHeight := (inHeight-1)*strideH - 2*padH + kernelH
	outWidth := (inWidth-1)*strideW - 2*padW + kernelW

	// Validate bias if provided
	var biasData []float32
	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outChannels {
			biasData = bias.Data()
		} else {
			panic(fmt.Sprintf("tensor.Conv2DTransposed: bias must be 1D [outChannels], got %v", biasShape))
		}
	}

	// Create output tensor
	result := New(t.dtype, NewShape(batchSize, outChannels, outHeight, outWidth))

	// Call fp32.Conv2DTransposed
	fp32.Conv2DTransposed(
		result.data,
		t.data,
		kernel.data,
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
		biasData,
	)

	return result
}

// Conv1D performs 1D convolution (simplified 2D convolution with width=1)
// Input shape: [batch, inChannels, length] or [inChannels, length]
// Kernel shape: [outChannels, inChannels, kernelLen]
// Bias shape: [outChannels] (optional)
// Output shape: [batch, outChannels, outLen] or [outChannels, outLen]
func (t *Tensor) Conv1D(kernel, bias *Tensor, stride, padding int) *Tensor {
	if t == nil || kernel == nil {
		return nil
	}

	tShape := t.Shape()
	kernelShape := kernel.Shape()

	// Handle both 2D [batch, inChannels, length] and 3D [inChannels, length] input
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

	// Reshape to 4D for Conv2D: add width=1 dimension
	// Input: [batch, inChannels, length] -> [batch, inChannels, length, 1]
	// Kernel: [outChannels, inChannels, kernelLen] -> [outChannels, inChannels, kernelLen, 1]
	// Output: [batch, outChannels, outLen] -> [batch, outChannels, outLen, 1]

	// Reshape to 4D for Conv2D: add width=1 dimension
	// Input: [batch, inChannels, length] -> [batch, inChannels, length, 1]
	input4D := t.Clone().Reshape([]int{batchSize, inChannels, length, 1})

	// Kernel: [outChannels, inChannels, kernelLen] -> [outChannels, inChannels, kernelLen, 1]
	kernel4D := kernel.Clone().Reshape([]int{outChannels, inChannels, kernelLen, 1})

	// Use Conv2D with width=1
	result4D := input4D.Conv2D(kernel4D, bias, []int{stride, 1}, []int{padding, 0})

	// Reshape back to 3D: [batch, outChannels, outLen, 1] -> [batch, outChannels, outLen]
	result := result4D.Reshape([]int{batchSize, outChannels, outLen})

	// If original was 2D, remove batch dimension
	if len(tShape) == 2 {
		result = result.Reshape([]int{outChannels, outLen})
	}

	return result
}

// MaxPool2D performs max pooling operation
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels, outHeight, outWidth]
func (t *Tensor) MaxPool2D(kernelSize, stride, padding []int) *Tensor {
	if t == nil {
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

	result := New(t.dtype, NewShape(batchSize, channels, outHeight, outWidth))

	// Perform max pooling using fp32
	fp32.MaxPool2D(
		result.Data(),
		t.Data(),
		batchSize, channels, inHeight, inWidth,
		kernelH, kernelW, strideH, strideW, padH, padW,
	)

	return result
}

// AvgPool2D performs average pooling operation
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels, outHeight, outWidth]
func (t *Tensor) AvgPool2D(kernelSize, stride, padding []int) *Tensor {
	if t == nil {
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

	result := New(t.dtype, NewShape(batchSize, channels, outHeight, outWidth))

	// Perform average pooling using fp32
	fp32.AvgPool2D(
		result.Data(),
		t.Data(),
		batchSize, channels, inHeight, inWidth,
		kernelH, kernelW, strideH, strideW, padH, padW,
	)

	return result
}

// GlobalAvgPool2D performs global average pooling
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels]
func (t *Tensor) GlobalAvgPool2D() *Tensor {
	if t == nil {
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

	result := New(t.dtype, NewShape(batchSize, channels))

	// Perform global average pooling using fp32
	fp32.GlobalAvgPool2D(
		result.Data(),
		t.Data(),
		batchSize, channels, height, width,
	)

	return result
}

// DepthwiseConv2D performs depthwise separable 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [inChannels, 1, kernelH, kernelW] or [inChannels, kernelH, kernelW]
// Bias shape: [inChannels] (optional, can be nil)
// Output shape: [batch, inChannels, outHeight, outWidth]
// Each input channel is convolved with its own kernel (depth multiplier = 1)
func (t *Tensor) DepthwiseConv2D(kernel, bias *Tensor, stride, padding []int) *Tensor {
	if t == nil || kernel == nil {
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

	var biasData []float32
	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == inChannels {
			biasData = bias.Data()
		} else {
			panic(fmt.Sprintf("tensor.DepthwiseConv2D: bias must be 1D [inChannels], got %v", biasShape))
		}
	}

	result := New(t.dtype, NewShape(batchSize, inChannels, outHeight, outWidth))

	// Convert kernel to 3D format [channels, kernelH, kernelW] for fp32 function
	kernelData := kernel.Data()
	if len(kernelShape) == 4 {
		// Convert from [channels, 1, kernelH, kernelW] to [channels, kernelH, kernelW]
		kernel3D := make([]float32, inChannels*kernelH*kernelW)
		for c := 0; c < inChannels; c++ {
			srcOffset := c * kernelH * kernelW
			dstOffset := c * kernelH * kernelW
			copy(kernel3D[dstOffset:dstOffset+kernelH*kernelW], kernelData[srcOffset:srcOffset+kernelH*kernelW])
		}
		kernelData = kernel3D
	}

	// Perform depthwise convolution using fp32
	fp32.DepthwiseConv2D(
		result.Data(),
		t.Data(),
		kernelData,
		biasData,
		batchSize, inChannels, inHeight, inWidth,
		kernelH, kernelW, strideH, strideW, padH, padW,
	)

	return result
}

// GroupConv2D performs grouped 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [outChannels, inChannels/groups, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// groups: number of groups (inChannels must be divisible by groups)
// Output shape: [batch, outChannels, outHeight, outWidth]
func (t *Tensor) GroupConv2D(kernel, bias *Tensor, stride, padding []int, groups int) *Tensor {
	if t == nil || kernel == nil {
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

	var biasData []float32
	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outChannels {
			biasData = bias.Data()
		} else {
			panic(fmt.Sprintf("tensor.GroupConv2D: bias must be 1D [outChannels], got %v", biasShape))
		}
	}

	result := New(t.dtype, NewShape(batchSize, outChannels, outHeight, outWidth))

	// Perform grouped convolution using fp32
	fp32.GroupConv2D(
		result.Data(),
		t.Data(),
		kernel.Data(),
		biasData,
		batchSize, inChannels, outChannels, inHeight, inWidth,
		kernelH, kernelW, strideH, strideW, padH, padW, groups,
	)

	return result
}

// DilatedConv2D performs dilated (atrous) 2D convolution
// Input shape: [batch, inChannels, height, width]
// Kernel shape: [outChannels, inChannels, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// dilation: [dilationH, dilationW] - dilation rates
// Output shape: [batch, outChannels, outHeight, outWidth]
func (t *Tensor) DilatedConv2D(kernel, bias *Tensor, stride, padding, dilation []int) *Tensor {
	if t == nil || kernel == nil {
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

	var biasData []float32
	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outChannels {
			biasData = bias.Data()
		} else {
			panic(fmt.Sprintf("tensor.DilatedConv2D: bias must be 1D [outChannels], got %v", biasShape))
		}
	}

	result := New(t.dtype, NewShape(batchSize, outChannels, outHeight, outWidth))

	// Perform dilated convolution using fp32
	fp32.DilatedConv2D(
		result.Data(),
		t.Data(),
		kernel.Data(),
		biasData,
		batchSize, inChannels, outChannels, inHeight, inWidth,
		kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW,
	)

	return result
}

// Conv3D performs 3D convolution
// Input shape: [batch, inChannels, depth, height, width]
// Kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
// Bias shape: [outChannels] (optional, can be nil)
// Stride: [strideD, strideH, strideW]
// Padding: [padD, padH, padW]
// Output shape: [batch, outChannels, outDepth, outHeight, outWidth]
func (t *Tensor) Conv3D(kernel, bias *Tensor, stride, padding []int) *Tensor {
	if t == nil || kernel == nil {
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

	var biasData []float32
	if bias != nil {
		biasShape := bias.Shape()
		if len(biasShape) == 1 && biasShape[0] == outChannels {
			biasData = bias.Data()
		} else {
			panic(fmt.Sprintf("tensor.Conv3D: bias must be 1D [outChannels], got %v", biasShape))
		}
	}

	result := New(t.dtype, NewShape(batchSize, outChannels, outDepth, outHeight, outWidth))

	// Perform 3D convolution using fp32
	fp32.Conv3D(
		result.Data(),
		t.Data(),
		kernel.Data(),
		biasData,
		batchSize, inChannels, outChannels, depth, height, width,
		kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW,
	)

	return result
}

// AdaptiveAvgPool2D performs adaptive average pooling to a fixed output size
// Input shape: [batch, channels, height, width]
// outputSize: [outHeight, outWidth] - target output spatial dimensions
// Output shape: [batch, channels, outHeight, outWidth]
func (t *Tensor) AdaptiveAvgPool2D(outputSize []int) *Tensor {
	if t == nil {
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

	result := New(t.dtype, NewShape(batchSize, channels, outHeight, outWidth))

	// Perform adaptive average pooling using fp32
	fp32.AdaptiveAvgPool2D(
		result.Data(),
		t.Data(),
		batchSize, channels, height, width, outHeight, outWidth,
	)

	return result
}

// Im2Col converts image patches to columns (for convolution)
// Input shape: [batch, channels, height, width]
// Output shape: [batch*outHeight*outWidth, channels*kernelH*kernelW]
// Uses fp32.Im2Col for optimized computation
func (t *Tensor) Im2Col(kernelSize, stride, padding []int) *Tensor {
	if t == nil {
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

	result := New(t.dtype, NewShape(colHeight, colWidth))

	// Call fp32.Im2Col
	fp32.Im2Col(
		result.data,
		t.data,
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
}

// Col2Im converts columns back to image (inverse of Im2Col)
// Input shape: [batch*outHeight*outWidth, channels*kernelH*kernelW]
// Output shape: [batch, channels, height, width]
// Uses fp32.Col2Im for optimized computation
func (t *Tensor) Col2Im(outputShape, kernelSize, stride, padding []int) *Tensor {
	if t == nil {
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

	result := New(t.dtype, NewShape(batchSize, channels, height, width))

	// Call fp32.Col2Im
	fp32.Col2Im(
		result.data,
		t.data,
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
}
