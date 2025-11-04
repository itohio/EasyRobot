package fp32

// Im2Col converts image patches to columns for GEMM-based convolution
// This is used to convert convolution operations to matrix multiplication
// im: input image [batchSize, channels, height, width] in row-major layout
// col: output columns [batchSize*outHeight*outWidth, channels*kernelH*kernelW]
// The output format is optimized for matrix multiplication
func Im2Col(
	col, im []float32,
	batchSize, channels int,
	height, width int,
	kernelH, kernelW int,
	padH, padW int,
	strideH, strideW int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	// Calculate output dimensions
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Index for output column
	colIdx := 0

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width

		// For each output position (outH, outW)
		for outH := 0; outH < outHeight; outH++ {
			for outW := 0; outW < outWidth; outW++ {
				// For each channel
				for c := 0; c < channels; c++ {
					channelOffset := batchOffset + c*height*width

					// For each kernel position
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							// Calculate input position
							inH := outH*strideH + kh - padH
							inW := outW*strideW + kw - padW

							// Check bounds
							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								// Get pixel value from input
								imIdx := channelOffset + inH*width + inW
								col[colIdx] = im[imIdx]
							} else {
								// Padding: zero
								col[colIdx] = 0
							}
							colIdx++
						}
					}
				}
			}
		}
	}
}

// Col2Im converts columns back to image (inverse of Im2Col)
// col: input columns [batchSize*outHeight*outWidth, channels*kernelH*kernelW]
// im: output image [batchSize, channels, height, width] in row-major layout
// This accumulates values (for backpropagation or gradient accumulation)
func Col2Im(
	im, col []float32,
	batchSize, channels int,
	height, width int,
	kernelH, kernelW int,
	padH, padW int,
	strideH, strideW int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	// Calculate output dimensions
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Index for input column
	colIdx := 0

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width

		// For each output position (outH, outW)
		for outH := 0; outH < outHeight; outH++ {
			for outW := 0; outW < outWidth; outW++ {
				// For each channel
				for c := 0; c < channels; c++ {
					channelOffset := batchOffset + c*height*width

					// For each kernel position
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							// Calculate input position
							inH := outH*strideH + kh - padH
							inW := outW*strideW + kw - padW

							// Check bounds
							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								// Accumulate to output image
								imIdx := channelOffset + inH*width + inW
								im[imIdx] += col[colIdx]
							}
							colIdx++
						}
					}
				}
			}
		}
	}
}

// Conv2D performs 2D convolution with batched input using Im2Col + GEMM
// This is optimized for neural networks using matrix multiplication
// output: [batchSize, outChannels, outHeight, outWidth]
// input: [batchSize, inChannels, inHeight, inWidth]
// weights: [outChannels, inChannels, kernelH, kernelW]
// bias: optional bias [outChannels] (can be nil)
func Conv2D(
	output, input, weights []float32,
	batchSize, inChannels, outChannels int,
	inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW int,
	strideH, strideW int,
	padH, padW int,
	bias []float32,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 {
		return
	}

	// Calculate dimensions for Im2Col
	kernelSize := inChannels * kernelH * kernelW
	im2colSize := batchSize * outHeight * outWidth

	// Allocate temporary buffer for Im2Col output
	// Format: [batchSize*outHeight*outWidth, inChannels*kernelH*kernelW]
	im2col := make([]float32, im2colSize*kernelSize)

	// Step 1: Convert input to columns using Im2Col
	Im2Col(im2col, input, batchSize, inChannels, inHeight, inWidth,
		kernelH, kernelW, padH, padW, strideH, strideW)

	// Step 2: Reshape weights for GEMM
	// weights: [outChannels, inChannels, kernelH, kernelW]
	// Reshape to: [outChannels, inChannels*kernelH*kernelW]
	// For GEMM: W * im2col^T -> [outChannels, batchSize*outHeight*outWidth]

	// Step 3: Reshape weights for GEMM
	// weights: [outChannels, inChannels, kernelH, kernelW]
	// Need to reshape to: [outChannels, inChannels*kernelH*kernelW] for GEMM
	// weights[oc][ic][kh][kw] = weights[oc*inChannels*kernelH*kernelW + ic*kernelH*kernelW + kh*kernelW + kw]
	// For GEMM, we need: weights[oc][ic*kernelH*kernelW + kh*kernelW + kw]
	// This is already in the right layout, we just need to access it correctly

	// Step 4: Perform GEMM: output = weights * im2col^T
	// We need: output = weights * im2col^T
	// weights: [outChannels, inChannels*kernelH*kernelW] with ldA = kernelSize
	// im2col: [im2colSize, kernelSize] with ldB = kernelSize (we treat it as N×K for NT)
	// We use GEMM_NT: C = alpha*A*B^T + beta*C
	// A (weights): M=outChannels, K=kernelSize
	// B (im2col): N=im2colSize, K=kernelSize
	// Result C: [outChannels, im2colSize] with ldC = im2colSize
	gemmOutput := make([]float32, outChannels*im2colSize)

	// Perform GEMM_NT: gemmOutput = weights * im2col^T
	// weights: [outChannels, kernelSize] with ldA = kernelSize
	// im2col: [im2colSize, kernelSize] with ldB = kernelSize
	// gemmOutput: [outChannels, im2colSize] with ldC = im2colSize
	Gemm_NT(gemmOutput, weights, im2col,
		im2colSize,  // ldC
		kernelSize,  // ldA (weights leading dimension = kernelSize)
		kernelSize,  // ldB (im2col leading dimension = kernelSize)
		outChannels, // M (number of output channels)
		im2colSize,  // N (number of output positions = batchSize*outHeight*outWidth)
		kernelSize,  // K (kernel size = inChannels*kernelH*kernelW)
		1.0,         // alpha
		0.0,         // beta
	)

	// Step 5: Add bias if provided
	if bias != nil {
		for oc := 0; oc < outChannels; oc++ {
			biasVal := bias[oc]
			for i := 0; i < im2colSize; i++ {
				// Index in gemmOutput: [outChannels, im2colSize]
				gemmIdx := oc*im2colSize + i
				gemmOutput[gemmIdx] += biasVal
			}
		}
	}

	// Step 6: Reshape and transpose gemmOutput to final output format
	// gemmOutput: [outChannels, batchSize*outHeight*outWidth] (row-major)
	// output: [batchSize, outChannels, outHeight, outWidth] (row-major)
	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Source index in gemmOutput
					gemmIdx := oc*im2colSize + b*outHeight*outWidth + outH*outWidth + outW
					// Destination index in output
					outIdx := b*outChannels*outHeight*outWidth +
						oc*outHeight*outWidth +
						outH*outWidth + outW
					output[outIdx] = gemmOutput[gemmIdx]
				}
			}
		}
	}
}

// Conv2DTransposed performs transposed 2D convolution (deconvolution)
// This is the inverse operation of Conv2D
// output: [batchSize, outChannels, outHeight, outWidth]
// input: [batchSize, inChannels, inHeight, inWidth]
// weights: [inChannels, outChannels, kernelH, kernelW] (transposed from Conv2D)
func Conv2DTransposed(
	output, input, weights []float32,
	batchSize, inChannels, outChannels int,
	inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW int,
	strideH, strideW int,
	padH, padW int,
	bias []float32,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 {
		return
	}

	// Initialize output to zero
	outputSize := batchSize * outChannels * outHeight * outWidth
	for i := 0; i < outputSize; i++ {
		output[i] = 0
	}

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchInOffset := b * inChannels * inHeight * inWidth
		batchOutOffset := b * outChannels * outHeight * outWidth

		// For each input channel
		for ic := 0; ic < inChannels; ic++ {
			icOffset := batchInOffset + ic*inHeight*inWidth

			// For each output channel
			for oc := 0; oc < outChannels; oc++ {
				ocOffset := batchOutOffset + oc*outHeight*outWidth

				// Get weights for this filter: [inChannels, outChannels, kernelH, kernelW]
				// weights[ic][oc][kh][kw]
				weightOffset := ic*outChannels*kernelH*kernelW +
					oc*kernelH*kernelW

				// For each input position
				for inH := 0; inH < inHeight; inH++ {
					for inW := 0; inW < inWidth; inW++ {
						// Calculate corresponding output position
						outH := inH*strideH - padH
						outW := inW*strideW - padW

						// Get input value
						inIdx := icOffset + inH*inWidth + inW
						inVal := input[inIdx]

						// Apply kernel
						for kh := 0; kh < kernelH; kh++ {
							for kw := 0; kw < kernelW; kw++ {
								// Calculate output position
								oh := outH + kh
								ow := outW + kw

								// Check bounds
								if oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth {
									// Get weight
									weightIdx := weightOffset + kh*kernelW + kw
									weight := weights[weightIdx]

									// Accumulate to output
									outIdx := ocOffset + oh*outWidth + ow
									output[outIdx] += inVal * weight
								}
							}
						}
					}
				}
			}
		}
	}

	// Add bias if provided
	if bias != nil {
		for b := 0; b < batchSize; b++ {
			for oc := 0; oc < outChannels; oc++ {
				biasVal := bias[oc]
				ocOffset := b*outChannels*outHeight*outWidth + oc*outHeight*outWidth
				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						outIdx := ocOffset + outH*outWidth + outW
						output[outIdx] += biasVal
					}
				}
			}
		}
	}
}

// MaxPool2D performs 2D max pooling on batched input
// dst: output [batchSize, channels, outHeight, outWidth]
// src: input [batchSize, channels, height, width]
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func MaxPool2D(
	dst, src []float32,
	batchSize, channels, height, width int,
	kernelH, kernelW, strideH, strideW, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width
		dstBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*height*width
			dstChannelOffset := dstBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Calculate input window position
					startH := outH*strideH - padH
					startW := outW*strideW - padW

					// Find max in window
					maxVal := float32(-1e30)
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							inH := startH + kh
							inW := startW + kw

							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								idx := channelOffset + inH*width + inW
								val := src[idx]
								if val > maxVal {
									maxVal = val
								}
							}
						}
					}

					// Store max
					dstIdx := dstChannelOffset + outH*outWidth + outW
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// AvgPool2D performs 2D average pooling on batched input
// dst: output [batchSize, channels, outHeight, outWidth]
// src: input [batchSize, channels, height, width]
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func AvgPool2D(
	dst, src []float32,
	batchSize, channels, height, width int,
	kernelH, kernelW, strideH, strideW, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width
		dstBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*height*width
			dstChannelOffset := dstBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					startH := outH*strideH - padH
					startW := outW*strideW - padW

					var sum float32
					var count int

					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							inH := startH + kh
							inW := startW + kw

							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								idx := channelOffset + inH*width + inW
								sum += src[idx]
								count++
							}
						}
					}

					// Store average
					dstIdx := dstChannelOffset + outH*outWidth + outW
					if count > 0 {
						dst[dstIdx] = sum / float32(count)
					} else {
						dst[dstIdx] = 0
					}
				}
			}
		}
	}
}

// GlobalAvgPool2D performs global average pooling
// dst: output [batchSize, channels]
// src: input [batchSize, channels, height, width]
func GlobalAvgPool2D(
	dst, src []float32,
	batchSize, channels, height, width int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	// Process each batch and channel
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width
		dstBatchOffset := b * channels

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*height*width

			var sum float32
			count := height * width

			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					idx := channelOffset + h*width + w
					sum += src[idx]
				}
			}

			// Store average
			dstIdx := dstBatchOffset + c
			dst[dstIdx] = sum / float32(count)
		}
	}
}

// AdaptiveAvgPool2D performs adaptive average pooling to a fixed output size
// dst: output [batchSize, channels, outHeight, outWidth]
// src: input [batchSize, channels, height, width]
// This divides the input into approximately equal regions and averages each region
func AdaptiveAvgPool2D(
	dst, src []float32,
	batchSize, channels, height, width, outHeight, outWidth int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width
		dstBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*height*width
			dstChannelOffset := dstBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Calculate the region in input to average
					// We divide the input evenly across output positions
					hStart := (outH * height) / outHeight
					hEnd := ((outH + 1) * height) / outHeight
					wStart := (outW * width) / outWidth
					wEnd := ((outW + 1) * width) / outWidth

					var sum float32
					count := 0

					for h := hStart; h < hEnd; h++ {
						for w := wStart; w < wEnd; w++ {
							idx := channelOffset + h*width + w
							sum += src[idx]
							count++
						}
					}

					// Store average
					dstIdx := dstChannelOffset + outH*outWidth + outW
					if count > 0 {
						dst[dstIdx] = sum / float32(count)
					} else {
						dst[dstIdx] = 0
					}
				}
			}
		}
	}
}

// DepthwiseConv2D performs depthwise 2D convolution
// dst: output [batchSize, channels, outHeight, outWidth]
// src: input [batchSize, channels, height, width]
// kernel: depthwise kernels [channels, kernelH, kernelW]
// bias: bias terms [channels] (optional, can be nil)
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func DepthwiseConv2D(
	dst, src, kernel, bias []float32,
	batchSize, channels, height, width int,
	kernelH, kernelW, strideH, strideW, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Process each batch, channel separately (depthwise)
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width
		dstBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*height*width
			dstChannelOffset := dstBatchOffset + c*outHeight*outWidth
			kernelOffset := c * kernelH * kernelW

			// Perform 2D convolution for this channel
			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					var sum float32

					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							inH := outH*strideH + kh - padH
							inW := outW*strideW + kw - padW

							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								inIdx := channelOffset + inH*width + inW
								kernelIdx := kernelOffset + kh*kernelW + kw
								sum += src[inIdx] * kernel[kernelIdx]
							}
						}
					}

					// Add bias if provided
					if bias != nil {
						sum += bias[c]
					}

					// Store result
					outIdx := dstChannelOffset + outH*outWidth + outW
					dst[outIdx] = sum
				}
			}
		}
	}
}

// GroupConv2D performs grouped 2D convolution
// dst: output [batchSize, outChannels, outHeight, outWidth]
// src: input [batchSize, inChannels, height, width]
// kernel: grouped kernels [outChannels, inChannels/groups, kernelH, kernelW]
// bias: bias terms [outChannels] (optional, can be nil)
// groups: number of groups (inChannels and outChannels must be divisible by groups)
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func GroupConv2D(
	dst, src, kernel, bias []float32,
	batchSize, inChannels, outChannels, height, width int,
	kernelH, kernelW, strideH, strideW, padH, padW, groups int,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 || height == 0 || width == 0 {
		return
	}

	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	channelsPerGroup := inChannels / groups
	outChannelsPerGroup := outChannels / groups

	// Process each group separately
	for g := 0; g < groups; g++ {
		inputChanStart := g * channelsPerGroup
		outputChanStart := g * outChannelsPerGroup
		kernelChanStart := g * outChannelsPerGroup * channelsPerGroup * kernelH * kernelW

		// For each output channel in this group
		for oc := 0; oc < outChannelsPerGroup; oc++ {
			outChan := outputChanStart + oc
			kernelOffset := kernelChanStart + oc*channelsPerGroup*kernelH*kernelW

			// Process each batch
			for b := 0; b < batchSize; b++ {
				dstBatchOffset := b * outChannels * outHeight * outWidth
				dstChannelOffset := dstBatchOffset + outChan*outHeight*outWidth

				// Perform convolution over input channels in this group
				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						var sum float32

						for ic := 0; ic < channelsPerGroup; ic++ {
							inChan := inputChanStart + ic
							srcBatchOffset := b * inChannels * height * width
							srcChannelOffset := srcBatchOffset + inChan*height*width
							kernelChanOffset := kernelOffset + ic*kernelH*kernelW

							for kh := 0; kh < kernelH; kh++ {
								for kw := 0; kw < kernelW; kw++ {
									inH := outH*strideH + kh - padH
									inW := outW*strideW + kw - padW

									if inH >= 0 && inH < height && inW >= 0 && inW < width {
										srcIdx := srcChannelOffset + inH*width + inW
										kernelIdx := kernelChanOffset + kh*kernelW + kw
										sum += src[srcIdx] * kernel[kernelIdx]
									}
								}
							}
						}

						// Add bias if provided
						if bias != nil {
							sum += bias[outChan]
						}

						// Store result
						dstIdx := dstChannelOffset + outH*outWidth + outW
						dst[dstIdx] = sum
					}
				}
			}
		}
	}
}

// DilatedConv2D performs dilated 2D convolution
// dst: output [batchSize, outChannels, outHeight, outWidth]
// src: input [batchSize, inChannels, height, width]
// kernel: kernels [outChannels, inChannels, kernelH, kernelW]
// bias: bias terms [outChannels] (optional, can be nil)
// dilationH, dilationW: dilation factors
// outHeight = (height + 2*padH - ((kernelH-1)*dilationH + 1)) / strideH + 1
// outWidth = (width + 2*padW - ((kernelW-1)*dilationW + 1)) / strideW + 1
func DilatedConv2D(
	dst, src, kernel, bias []float32,
	batchSize, inChannels, outChannels, height, width int,
	kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW int,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 || height == 0 || width == 0 {
		return
	}

	outHeight := (height+2*padH-((kernelH-1)*dilationH+1))/strideH + 1
	outWidth := (width+2*padW-((kernelW-1)*dilationW+1))/strideW + 1

	// Process each batch and output channel
	for b := 0; b < batchSize; b++ {
		srcBatchOffset := b * inChannels * height * width
		dstBatchOffset := b * outChannels * outHeight * outWidth

		for oc := 0; oc < outChannels; oc++ {
			kernelOffset := oc * inChannels * kernelH * kernelW
			dstChannelOffset := dstBatchOffset + oc*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					var sum float32

					for ic := 0; ic < inChannels; ic++ {
						srcChannelOffset := srcBatchOffset + ic*height*width
						kernelChannelOffset := kernelOffset + ic*kernelH*kernelW

						for kh := 0; kh < kernelH; kh++ {
							for kw := 0; kw < kernelW; kw++ {
								// Apply dilation
								inH := outH*strideH + kh*dilationH - padH
								inW := outW*strideW + kw*dilationW - padW

								if inH >= 0 && inH < height && inW >= 0 && inW < width {
									srcIdx := srcChannelOffset + inH*width + inW
									kernelIdx := kernelChannelOffset + kh*kernelW + kw
									sum += src[srcIdx] * kernel[kernelIdx]
								}
							}
						}
					}

					// Add bias if provided
					if bias != nil {
						sum += bias[oc]
					}

					// Store result
					dstIdx := dstChannelOffset + outH*outWidth + outW
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// Conv3D performs 3D convolution
// dst: output [batchSize, outChannels, outDepth, outHeight, outWidth]
// src: input [batchSize, inChannels, depth, height, width]
// kernel: kernels [outChannels, inChannels, kernelD, kernelH, kernelW]
// bias: bias terms [outChannels] (optional, can be nil)
// outDepth = (depth + 2*padD - kernelD) / strideD + 1
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func Conv3D(
	dst, src, kernel, bias []float32,
	batchSize, inChannels, outChannels, depth, height, width int,
	kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW int,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 || depth == 0 || height == 0 || width == 0 {
		return
	}

	outDepth := (depth+2*padD-kernelD)/strideD + 1
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Process each batch and output channel
	for b := 0; b < batchSize; b++ {
		srcBatchOffset := b * inChannels * depth * height * width
		dstBatchOffset := b * outChannels * outDepth * outHeight * outWidth

		for oc := 0; oc < outChannels; oc++ {
			kernelOffset := oc * inChannels * kernelD * kernelH * kernelW
			dstChannelOffset := dstBatchOffset + oc*outDepth*outHeight*outWidth

			for outD := 0; outD < outDepth; outD++ {
				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						var sum float32

						for ic := 0; ic < inChannels; ic++ {
							srcChannelOffset := srcBatchOffset + ic*depth*height*width
							kernelChannelOffset := kernelOffset + ic*kernelD*kernelH*kernelW

							for kd := 0; kd < kernelD; kd++ {
								for kh := 0; kh < kernelH; kh++ {
									for kw := 0; kw < kernelW; kw++ {
										inD := outD*strideD + kd - padD
										inH := outH*strideH + kh - padH
										inW := outW*strideW + kw - padW

										if inD >= 0 && inD < depth && inH >= 0 && inH < height && inW >= 0 && inW < width {
											srcIdx := srcChannelOffset + inD*height*width + inH*width + inW
											kernelIdx := kernelChannelOffset + kd*kernelH*kernelW + kh*kernelW + kw
											sum += src[srcIdx] * kernel[kernelIdx]
										}
									}
								}
							}
						}

						// Add bias if provided
						if bias != nil {
							sum += bias[oc]
						}

						// Store result
						dstIdx := dstChannelOffset + outD*outHeight*outWidth + outH*outWidth + outW
						dst[dstIdx] = sum
					}
				}
			}
		}
	}
}
