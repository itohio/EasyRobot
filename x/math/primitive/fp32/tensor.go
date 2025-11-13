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
	channelKernelSize := kernelH * kernelW
	channelsKernelSize := channels * channelKernelSize

	// Index for output column
	colIdx := 0

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width

		// For each output position (outH, outW)
		for outH := 0; outH < outHeight; outH++ {
			inTop := outH*strideH - padH
			for outW := 0; outW < outWidth; outW++ {
				inLeft := outW*strideW - padW

				dstBase := colIdx
				colIdx += channelsKernelSize

				// Use a fast path when the kernel window is fully inside
				if inTop >= 0 && inTop+kernelH <= height && inLeft >= 0 && inLeft+kernelW <= width {
					for c := 0; c < channels; c++ {
						channelOffset := batchOffset + c*height*width
						srcStart := channelOffset + inTop*width + inLeft
						dstChannel := dstBase + c*channelKernelSize

						for kh := 0; kh < kernelH; kh++ {
							srcRow := srcStart + kh*width
							dstRow := dstChannel + kh*kernelW
							copy(col[dstRow:dstRow+kernelW], im[srcRow:srcRow+kernelW])
						}
					}
					continue
				}

				// For each channel
				for c := 0; c < channels; c++ {
					channelOffset := batchOffset + c*height*width
					dstChannel := dstBase + c*channelKernelSize

					// For each kernel position
					for kh := 0; kh < kernelH; kh++ {
						dstRow := dstChannel + kh*kernelW
						inH := inTop + kh

						if inH < 0 || inH >= height {
							for kw := 0; kw < kernelW; kw++ {
								col[dstRow+kw] = 0
							}
							continue
						}

						rowOffset := channelOffset + inH*width
						for kw := 0; kw < kernelW; kw++ {
							inW := inLeft + kw
							if inW >= 0 && inW < width {
								col[dstRow+kw] = im[rowOffset+inW]
							} else {
								col[dstRow+kw] = 0
							}
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
// This function zeroes the destination slice before accumulating values so
// callers can safely reuse pooled buffers without manual clearing.
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

	total := batchSize * channels * height * width
	for i := 0; i < total && i < len(im); i++ {
		im[i] = 0
	}

	// Calculate output dimensions
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1
	channelKernelSize := kernelH * kernelW
	channelsKernelSize := channels * channelKernelSize

	// Index for input column
	colIdx := 0

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * height * width

		// For each output position (outH, outW)
		for outH := 0; outH < outHeight; outH++ {
			inTop := outH*strideH - padH
			for outW := 0; outW < outWidth; outW++ {
				inLeft := outW*strideW - padW

				srcBase := colIdx
				colIdx += channelsKernelSize

				// Fast path: entire kernel fits within the bounds
				if inTop >= 0 && inTop+kernelH <= height && inLeft >= 0 && inLeft+kernelW <= width {
					for c := 0; c < channels; c++ {
						channelOffset := batchOffset + c*height*width
						dstStart := channelOffset + inTop*width + inLeft
						srcChannel := srcBase + c*channelKernelSize

						for kh := 0; kh < kernelH; kh++ {
							dstRow := dstStart + kh*width
							srcRow := srcChannel + kh*kernelW
							for kw := 0; kw < kernelW; kw++ {
								im[dstRow+kw] += col[srcRow+kw]
							}
						}
					}
					continue
				}

				// For each channel
				for c := 0; c < channels; c++ {
					channelOffset := batchOffset + c*height*width
					srcChannel := srcBase + c*channelKernelSize

					// For each kernel position
					for kh := 0; kh < kernelH; kh++ {
						srcRow := srcChannel + kh*kernelW
						inH := inTop + kh

						if inH < 0 || inH >= height {
							continue
						}

						rowOffset := channelOffset + inH*width
						for kw := 0; kw < kernelW; kw++ {
							inW := inLeft + kw
							if inW >= 0 && inW < width {
								im[rowOffset+inW] += col[srcRow+kw]
							}
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
	// Direct convolution implementation - optimized for performance
	// This avoids the massive memory allocation and inefficient GEMM operations of Im2Col+GEMM

	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outHeight; oh++ {
				for ow := 0; ow < outWidth; ow++ {
					// Calculate output position
					outIdx := b*outChannels*outHeight*outWidth +
						oc*outHeight*outWidth +
						oh*outWidth + ow

					sum := float32(0.0)

					// Convolution kernel loop
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							// Input position with padding
							ih := oh*strideH + kh - padH
							iw := ow*strideW + kw - padW

							// Skip if outside input bounds
							if ih < 0 || ih >= inHeight || iw < 0 || iw >= inWidth {
								continue
							}

							// Sum over input channels
							for ic := 0; ic < inChannels; ic++ {
								// Input index
								inIdx := b*inChannels*inHeight*inWidth +
									ic*inHeight*inWidth +
									ih*inWidth + iw

								// Weight index
								weightIdx := oc*inChannels*kernelH*kernelW +
									ic*kernelH*kernelW +
									kh*kernelW + kw

								sum += input[inIdx] * weights[weightIdx]
							}
						}
					}

					// Add bias if provided
					if bias != nil {
						sum += bias[oc]
					}

					output[outIdx] = sum
				}
			}
		}
	}
}

// Conv2DKernelGrad computes kernel gradients for 2D convolution
// input: [batchSize, inChannels, inHeight, inWidth]
// outputGrad: [batchSize, outChannels, outHeight, outWidth]
// kernelGrad: [outChannels, inChannels, kernelH, kernelW] (output)
func Conv2DKernelGrad(
	kernelGrad, input, outputGrad []float32,
	batchSize, inChannels, outChannels int,
	inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW int,
	strideH, strideW int,
	padH, padW int,
) {
	// Direct kernel gradient computation - much more efficient than Im2Col + GEMM
	// This avoids the massive memory allocation and inefficient GEMM operations

	// Initialize kernel gradients to zero
	kernelSize := outChannels * inChannels * kernelH * kernelW
	for i := 0; i < kernelSize; i++ {
		kernelGrad[i] = 0.0
	}

	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outHeight; oh++ {
				for ow := 0; ow < outWidth; ow++ {
					// Get output gradient value
					gradIdx := b*outChannels*outHeight*outWidth +
						oc*outHeight*outWidth +
						oh*outWidth + ow
					gradVal := outputGrad[gradIdx]

					// Convolution kernel loop for gradient accumulation
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							// Input position with padding
							ih := oh*strideH + kh - padH
							iw := ow*strideW + kw - padW

							// Skip if outside input bounds
							if ih < 0 || ih >= inHeight || iw < 0 || iw >= inWidth {
								continue
							}

							// Accumulate gradients over input channels
							for ic := 0; ic < inChannels; ic++ {
								// Input index
								inIdx := b*inChannels*inHeight*inWidth +
									ic*inHeight*inWidth +
									ih*inWidth + iw

								// Kernel gradient index
								kernelIdx := oc*inChannels*kernelH*kernelW +
									ic*kernelH*kernelW +
									kh*kernelW + kw

								kernelGrad[kernelIdx] += input[inIdx] * gradVal
							}
						}
					}
				}
			}
		}
	}
}

// Conv1DKernelGrad computes kernel gradients for 1D convolution
// input: [batchSize, inChannels, inLength]
// outputGrad: [batchSize, outChannels, outLength]
// kernelGrad: [outChannels, inChannels, kernelLen] (output)
func Conv1DKernelGrad(
	kernelGrad, input, outputGrad []float32,
	batchSize, inChannels, outChannels int,
	inLength, outLength int,
	kernelLen int,
	stride, padding int,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 {
		return
	}

	kernelSize := inChannels * kernelLen
	im2colSize := batchSize * outLength

	// Allocate temporary buffers
	inputIm2Col := Pool.Get(im2colSize * kernelSize)
	defer Pool.Put(inputIm2Col)
	outputGradFlat := Pool.Get(im2colSize * outChannels)
	defer Pool.Put(outputGradFlat)

	// Step 1: Convert input to columns using Im2Col (treating as 2D with width=1)
	Im2Col(inputIm2Col, input, batchSize, inChannels, inLength, 1,
		kernelLen, 1, padding, 0, stride, 1)

	// Step 2: Flatten output gradient to matrix format
	// outputGrad: [batchSize, outChannels, outLength] -> [batchSize*outLength, outChannels]
	for b := 0; b < batchSize; b++ {
		for outL := 0; outL < outLength; outL++ {
			for oc := 0; oc < outChannels; oc++ {
				srcIdx := b*outChannels*outLength +
					oc*outLength + outL
				dstIdx := (b*outLength+outL)*outChannels + oc
				outputGradFlat[dstIdx] = outputGrad[srcIdx]
			}
		}
	}

	// Step 3: Compute kernel gradients using GEMM
	// Same as 2D case
	Gemm_NN(kernelGrad, outputGradFlat, inputIm2Col,
		kernelSize,  // ldC
		im2colSize,  // ldA
		kernelSize,  // ldB
		outChannels, // M
		kernelSize,  // N
		im2colSize,  // K
		1.0,         // alpha
		0.0,         // beta
	)
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

// MaxPool2DWithIndices performs 2D max pooling and stores indices of max elements
// dst: output [batchSize, channels, outHeight, outWidth]
// indices: output indices [batchSize, channels, outHeight, outWidth] as int32 (linear indices into src)
// src: input [batchSize, channels, height, width]
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
// Indices are stored as linear indices into the flattened input tensor for efficient backward pass
func MaxPool2DWithIndices(
	dst, src []float32,
	indices []int32,
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
		indicesBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*height*width
			dstChannelOffset := dstBatchOffset + c*outHeight*outWidth
			indicesChannelOffset := indicesBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Calculate input window position
					startH := outH*strideH - padH
					startW := outW*strideW - padW

					// Find max in window and store index
					maxVal := float32(-1e30)
					maxIdx := int32(-1)
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							inH := startH + kh
							inW := startW + kw

							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								idx := channelOffset + inH*width + inW
								val := src[idx]
								if val > maxVal {
									maxVal = val
									maxIdx = int32(idx)
								}
							}
						}
					}

					// Store max value and index
					dstIdx := dstChannelOffset + outH*outWidth + outW
					dst[dstIdx] = maxVal
					indicesIdx := indicesChannelOffset + outH*outWidth + outW
					indices[indicesIdx] = maxIdx
				}
			}
		}
	}
}

// MaxPool2DBackward performs backward pass for 2D max pooling.
// gradInput: output gradient [batchSize, channels, inHeight, inWidth]. The slice
// is cleared before gradients are accumulated so callers do not need to
// pre-initialize pooled buffers.
// gradOutput: input gradient [batchSize, channels, outHeight, outWidth]
// indices: indices from forward pass [batchSize, channels, outHeight, outWidth] as int32
// src: original input [batchSize, channels, inHeight, inWidth] (used to resolve ties when multiple positions have same max)
// batchSize, channels, inHeight, inWidth: input dimensions
// outHeight, outWidth: output dimensions
// kernelH, kernelW, strideH, strideW, padH, padW: pooling parameters
// Routes gradients to input positions that produced the maximum value during forward pass.
// If multiple positions had the same max value, the gradient is divided equally among them.
func MaxPool2DBackward(
	gradInput, gradOutput []float32,
	indices []int32,
	src []float32,
	batchSize, channels, inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW, strideH, strideW, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || inHeight == 0 || inWidth == 0 {
		return
	}

	gradTotal := batchSize * channels * inHeight * inWidth
	for i := 0; i < gradTotal && i < len(gradInput); i++ {
		gradInput[i] = 0
	}

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * inHeight * inWidth
		gradOutputBatchOffset := b * channels * outHeight * outWidth
		indicesBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*inHeight*inWidth
			gradOutputChannelOffset := gradOutputBatchOffset + c*outHeight*outWidth
			indicesChannelOffset := indicesBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Get output gradient value
					gradOutputIdx := gradOutputChannelOffset + outH*outWidth + outW
					gradVal := gradOutput[gradOutputIdx]

					// Get index from forward pass
					indicesIdx := indicesChannelOffset + outH*outWidth + outW
					maxIdx := indices[indicesIdx]

					if maxIdx >= 0 {
						// Get max value from original input to identify all positions with same value
						maxVal := src[maxIdx]

						// Calculate input window position
						startH := outH*strideH - padH
						startW := outW*strideW - padW

						// Count how many input positions had the max value (for tie-breaking)
						maxCount := 0
						for kh := 0; kh < kernelH; kh++ {
							for kw := 0; kw < kernelW; kw++ {
								inH := startH + kh
								inW := startW + kw

								if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
									idx := channelOffset + inH*inWidth + inW
									val := src[idx]
									// Use epsilon for floating point comparison
									epsilon := float32(1e-6)
									diff := val - maxVal
									if diff > -epsilon && diff < epsilon {
										maxCount++
									}
								}
							}
						}

						// Route gradient equally to all positions that had the max value
						if maxCount > 0 {
							gradPerPosition := gradVal / float32(maxCount)
							for kh := 0; kh < kernelH; kh++ {
								for kw := 0; kw < kernelW; kw++ {
									inH := startH + kh
									inW := startW + kw

									if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
										idx := channelOffset + inH*inWidth + inW
										val := src[idx]
										// Use epsilon for floating point comparison
										epsilon := float32(1e-6)
										diff := val - maxVal
										if diff > -epsilon && diff < epsilon {
											gradInput[idx] += gradPerPosition
										}
									}
								}
							}
						}
					}
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

// AvgPool2DBackward performs backward pass for 2D average pooling.
// gradInput: output gradient [batchSize, channels, inHeight, inWidth]. The slice
// is cleared before gradients are accumulated.
// gradOutput: input gradient [batchSize, channels, outHeight, outWidth]
// batchSize, channels, inHeight, inWidth: input dimensions
// outHeight, outWidth: output dimensions
// kernelH, kernelW, strideH, strideW, padH, padW: pooling parameters
// Routes gradient equally to all input positions in each pooling window, divided by kernel area.
func AvgPool2DBackward(
	gradInput, gradOutput []float32,
	batchSize, channels, inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW, strideH, strideW, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || inHeight == 0 || inWidth == 0 {
		return
	}

	gradTotal := batchSize * channels * inHeight * inWidth
	for i := 0; i < gradTotal && i < len(gradInput); i++ {
		gradInput[i] = 0
	}

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * inHeight * inWidth
		gradOutputBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*inHeight*inWidth
			gradOutputChannelOffset := gradOutputBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					gradOutputIdx := gradOutputChannelOffset + outH*outWidth + outW
					gradVal := gradOutput[gradOutputIdx]

					startH := outH*strideH - padH
					startW := outW*strideW - padW

					// Count valid positions in the pooling window to mirror forward averaging
					var count int
					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							inH := startH + kh
							inW := startW + kw
							if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
								count++
							}
						}
					}

					if count == 0 {
						continue
					}

					gradPerPosition := gradVal / float32(count)

					for kh := 0; kh < kernelH; kh++ {
						for kw := 0; kw < kernelW; kw++ {
							inH := startH + kh
							inW := startW + kw

							if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
								idx := channelOffset + inH*inWidth + inW
								gradInput[idx] += gradPerPosition
							}
						}
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

// MaxPool1D performs 1D max pooling on batched input
// dst: output [batchSize, channels, outLength]
// src: input [batchSize, channels, length]
// outLength = (length + 2*padding - kernelLen) / stride + 1
func MaxPool1D(
	dst, src []float32,
	batchSize, channels, length int,
	kernelLen, stride, padding int,
) {
	if batchSize == 0 || channels == 0 || length == 0 {
		return
	}

	outLength := (length+2*padding-kernelLen)/stride + 1

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * length
		dstBatchOffset := b * channels * outLength

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*length
			dstChannelOffset := dstBatchOffset + c*outLength

			for outL := 0; outL < outLength; outL++ {
				// Calculate input window position
				startL := outL*stride - padding

				// Find max in window
				maxVal := float32(-1e30)
				for kl := 0; kl < kernelLen; kl++ {
					inL := startL + kl

					if inL >= 0 && inL < length {
						idx := channelOffset + inL
						val := src[idx]
						if val > maxVal {
							maxVal = val
						}
					}
				}

				// Store max
				dstIdx := dstChannelOffset + outL
				dst[dstIdx] = maxVal
			}
		}
	}
}

// MaxPool3D performs 3D max pooling on batched input
// dst: output [batchSize, channels, outDepth, outHeight, outWidth]
// src: input [batchSize, channels, depth, height, width]
// outDepth = (depth + 2*padD - kernelD) / strideD + 1
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func MaxPool3D(
	dst, src []float32,
	batchSize, channels, depth, height, width int,
	kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || depth == 0 || height == 0 || width == 0 {
		return
	}

	outDepth := (depth+2*padD-kernelD)/strideD + 1
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * depth * height * width
		dstBatchOffset := b * channels * outDepth * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*depth*height*width
			dstChannelOffset := dstBatchOffset + c*outDepth*outHeight*outWidth

			for outD := 0; outD < outDepth; outD++ {
				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						// Calculate input window position
						startD := outD*strideD - padD
						startH := outH*strideH - padH
						startW := outW*strideW - padW

						// Find max in window
						maxVal := float32(-1e30)
						for kd := 0; kd < kernelD; kd++ {
							for kh := 0; kh < kernelH; kh++ {
								for kw := 0; kw < kernelW; kw++ {
									inD := startD + kd
									inH := startH + kh
									inW := startW + kw

									if inD >= 0 && inD < depth && inH >= 0 && inH < height && inW >= 0 && inW < width {
										idx := channelOffset + inD*height*width + inH*width + inW
										val := src[idx]
										if val > maxVal {
											maxVal = val
										}
									}
								}
							}
						}

						// Store max
						dstIdx := dstChannelOffset + outD*outHeight*outWidth + outH*outWidth + outW
						dst[dstIdx] = maxVal
					}
				}
			}
		}
	}
}

// AvgPool1D performs 1D average pooling on batched input
// dst: output [batchSize, channels, outLength]
// src: input [batchSize, channels, length]
// outLength = (length + 2*padding - kernelLen) / stride + 1
func AvgPool1D(
	dst, src []float32,
	batchSize, channels, length int,
	kernelLen, stride, padding int,
) {
	if batchSize == 0 || channels == 0 || length == 0 {
		return
	}

	outLength := (length+2*padding-kernelLen)/stride + 1

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * length
		dstBatchOffset := b * channels * outLength

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*length
			dstChannelOffset := dstBatchOffset + c*outLength

			for outL := 0; outL < outLength; outL++ {
				startL := outL*stride - padding

				var sum float32
				var count int

				for kl := 0; kl < kernelLen; kl++ {
					inL := startL + kl

					if inL >= 0 && inL < length {
						idx := channelOffset + inL
						sum += src[idx]
						count++
					}
				}

				// Store average
				dstIdx := dstChannelOffset + outL
				if count > 0 {
					dst[dstIdx] = sum / float32(count)
				} else {
					dst[dstIdx] = 0
				}
			}
		}
	}
}

// AvgPool3D performs 3D average pooling on batched input
// dst: output [batchSize, channels, outDepth, outHeight, outWidth]
// src: input [batchSize, channels, depth, height, width]
// outDepth = (depth + 2*padD - kernelD) / strideD + 1
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func AvgPool3D(
	dst, src []float32,
	batchSize, channels, depth, height, width int,
	kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || depth == 0 || height == 0 || width == 0 {
		return
	}

	outDepth := (depth+2*padD-kernelD)/strideD + 1
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * depth * height * width
		dstBatchOffset := b * channels * outDepth * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*depth*height*width
			dstChannelOffset := dstBatchOffset + c*outDepth*outHeight*outWidth

			for outD := 0; outD < outDepth; outD++ {
				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						startD := outD*strideD - padD
						startH := outH*strideH - padH
						startW := outW*strideW - padW

						var sum float32
						var count int

						for kd := 0; kd < kernelD; kd++ {
							for kh := 0; kh < kernelH; kh++ {
								for kw := 0; kw < kernelW; kw++ {
									inD := startD + kd
									inH := startH + kh
									inW := startW + kw

									if inD >= 0 && inD < depth && inH >= 0 && inH < height && inW >= 0 && inW < width {
										idx := channelOffset + inD*height*width + inH*width + inW
										sum += src[idx]
										count++
									}
								}
							}
						}

						// Store average
						dstIdx := dstChannelOffset + outD*outHeight*outWidth + outH*outWidth + outW
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
}

// GlobalMaxPool1D performs global max pooling over 1D spatial dimensions
// dst: output [batchSize, channels]
// src: input [batchSize, channels, length]
func GlobalMaxPool1D(
	dst, src []float32,
	batchSize, channels, length int,
) {
	if batchSize == 0 || channels == 0 || length == 0 {
		return
	}

	// Process each batch and channel
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * length
		dstBatchOffset := b * channels

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*length

			maxVal := float32(-1e30)
			for l := 0; l < length; l++ {
				idx := channelOffset + l
				val := src[idx]
				if val > maxVal {
					maxVal = val
				}
			}

			// Store max
			dstIdx := dstBatchOffset + c
			dst[dstIdx] = maxVal
		}
	}
}

// GlobalMaxPool2D performs global max pooling over 2D spatial dimensions
// dst: output [batchSize, channels]
// src: input [batchSize, channels, height, width]
func GlobalMaxPool2D(
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

			maxVal := float32(-1e30)
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					idx := channelOffset + h*width + w
					val := src[idx]
					if val > maxVal {
						maxVal = val
					}
				}
			}

			// Store max
			dstIdx := dstBatchOffset + c
			dst[dstIdx] = maxVal
		}
	}
}

// GlobalMaxPool3D performs global max pooling over 3D spatial dimensions
// dst: output [batchSize, channels]
// src: input [batchSize, channels, depth, height, width]
func GlobalMaxPool3D(
	dst, src []float32,
	batchSize, channels, depth, height, width int,
) {
	if batchSize == 0 || channels == 0 || depth == 0 || height == 0 || width == 0 {
		return
	}

	// Process each batch and channel
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * depth * height * width
		dstBatchOffset := b * channels

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*depth*height*width

			maxVal := float32(-1e30)
			for d := 0; d < depth; d++ {
				for h := 0; h < height; h++ {
					for w := 0; w < width; w++ {
						idx := channelOffset + d*height*width + h*width + w
						val := src[idx]
						if val > maxVal {
							maxVal = val
						}
					}
				}
			}

			// Store max
			dstIdx := dstBatchOffset + c
			dst[dstIdx] = maxVal
		}
	}
}

// AdaptiveMaxPool1D performs adaptive max pooling to fixed output size
// dst: output [batchSize, channels, outLength]
// src: input [batchSize, channels, inLength]
// This divides the input into approximately equal regions and takes max of each region
func AdaptiveMaxPool1D(
	dst, src []float32,
	batchSize, channels, inLength, outLength int,
) {
	if batchSize == 0 || channels == 0 || inLength == 0 || outLength == 0 {
		return
	}

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * inLength
		dstBatchOffset := b * channels * outLength

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*inLength
			dstChannelOffset := dstBatchOffset + c*outLength

			for outL := 0; outL < outLength; outL++ {
				// Calculate the region in input to max pool
				// We divide the input evenly across output positions
				lStart := (outL * inLength) / outLength
				lEnd := ((outL + 1) * inLength) / outLength

				maxVal := float32(-1e30)
				for l := lStart; l < lEnd; l++ {
					idx := channelOffset + l
					val := src[idx]
					if val > maxVal {
						maxVal = val
					}
				}

				// Store max
				dstIdx := dstChannelOffset + outL
				dst[dstIdx] = maxVal
			}
		}
	}
}

// AdaptiveMaxPool2D performs adaptive max pooling to fixed output size
// dst: output [batchSize, channels, outHeight, outWidth]
// src: input [batchSize, channels, inHeight, inWidth]
// This divides the input into approximately equal regions and takes max of each region
func AdaptiveMaxPool2D(
	dst, src []float32,
	batchSize, channels, inHeight, inWidth, outHeight, outWidth int,
) {
	if batchSize == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * inHeight * inWidth
		dstBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*inHeight*inWidth
			dstChannelOffset := dstBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Calculate the region in input to max pool
					// We divide the input evenly across output positions
					hStart := (outH * inHeight) / outHeight
					hEnd := ((outH + 1) * inHeight) / outHeight
					wStart := (outW * inWidth) / outWidth
					wEnd := ((outW + 1) * inWidth) / outWidth

					maxVal := float32(-1e30)
					for h := hStart; h < hEnd; h++ {
						for w := wStart; w < wEnd; w++ {
							idx := channelOffset + h*inWidth + w
							val := src[idx]
							if val > maxVal {
								maxVal = val
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

// AdaptiveMaxPool3D performs adaptive max pooling to fixed output size
// dst: output [batchSize, channels, outDepth, outHeight, outWidth]
// src: input [batchSize, channels, inDepth, inHeight, inWidth]
// This divides the input into approximately equal regions and takes max of each region
func AdaptiveMaxPool3D(
	dst, src []float32,
	batchSize, channels, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth int,
) {
	if batchSize == 0 || channels == 0 || inDepth == 0 || inHeight == 0 || inWidth == 0 || outDepth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	// Process each batch, channel, and output position
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * inDepth * inHeight * inWidth
		dstBatchOffset := b * channels * outDepth * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*inDepth*inHeight*inWidth
			dstChannelOffset := dstBatchOffset + c*outDepth*outHeight*outWidth

			for outD := 0; outD < outDepth; outD++ {
				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						// Calculate the region in input to max pool
						// We divide the input evenly across output positions
						dStart := (outD * inDepth) / outDepth
						dEnd := ((outD + 1) * inDepth) / outDepth
						hStart := (outH * inHeight) / outHeight
						hEnd := ((outH + 1) * inHeight) / outHeight
						wStart := (outW * inWidth) / outWidth
						wEnd := ((outW + 1) * inWidth) / outWidth

						maxVal := float32(-1e30)
						for d := dStart; d < dEnd; d++ {
							for h := hStart; h < hEnd; h++ {
								for w := wStart; w < wEnd; w++ {
									idx := channelOffset + d*inHeight*inWidth + h*inWidth + w
									val := src[idx]
									if val > maxVal {
										maxVal = val
									}
								}
							}
						}

						// Store max
						dstIdx := dstChannelOffset + outD*outHeight*outWidth + outH*outWidth + outW
						dst[dstIdx] = maxVal
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

// ScatterAdd adds values to destination tensor at positions specified by indices
// dst: destination tensor [batchSize, channels, inHeight, inWidth] (modified in-place).
// The destination is not cleared automatically; callers that require a fresh
// buffer must zero it before invoking this function.
// index: indices tensor [batchSize, channels, outHeight, outWidth] as int32 (linear indices into dst)
// value: values to add [batchSize, channels, outHeight, outWidth]
// batchSize, channels, inHeight, inWidth: destination tensor dimensions
// outHeight, outWidth: output dimensions (size of index/value tensors)
// For each position in index, adds the corresponding value from value to dst[index[i]]
// This is a general scatter operation useful for gradient routing in backpropagation
func ScatterAdd(
	dst []float32,
	index []int32,
	value []float32,
	batchSize, channels, inHeight, inWidth int,
	outHeight, outWidth int,
) {
	if batchSize == 0 || channels == 0 || inHeight == 0 || inWidth == 0 {
		return
	}

	// Process each batch, channel, and output position
	// Note: indices are linear indices into the full dst tensor, so we iterate over all positions
	for b := 0; b < batchSize; b++ {
		valueBatchOffset := b * channels * outHeight * outWidth
		indexBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			valueChannelOffset := valueBatchOffset + c*outHeight*outWidth
			indexChannelOffset := indexBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Get index and value
					indexIdx := indexChannelOffset + outH*outWidth + outW
					valueIdx := valueChannelOffset + outH*outWidth + outW
					dstIdx := index[indexIdx]
					val := value[valueIdx]

					// Add value to destination if index is valid
					// Note: indices are already linear indices into the full tensor
					if dstIdx >= 0 && int(dstIdx) < len(dst) {
						dst[dstIdx] += val
					}
				}
			}
		}
	}
}

// Fill fills a tensor with a constant value
// dst: destination tensor (modified in-place)
// value: value to fill
// num: number of elements to fill
// stride: access stride (for non-contiguous tensors)
func Fill(
	dst []float32,
	value float32,
	num, stride int,
) {
	if num == 0 {
		return
	}

	idx := 0
	for i := 0; i < num; i++ {
		dst[idx] = value
		idx += stride
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

// Conv2DTransposedWithOutputPadding performs transposed 2D convolution with output padding
// This extends Conv2DTransposed to support output padding, which is useful in some GAN architectures
// where the output size needs fine-grained control.
// output: [batchSize, outChannels, outHeight, outWidth]
// input: [batchSize, inChannels, inHeight, inWidth]
// weights: [inChannels, outChannels, kernelH, kernelW]
// outputPadH, outputPadW: additional padding applied to output dimensions
func Conv2DTransposedWithOutputPadding(
	output, input, weights []float32,
	batchSize, inChannels, outChannels int,
	inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW int,
	strideH, strideW int,
	padH, padW int,
	outputPadH, outputPadW int,
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
				weightOffset := ic*outChannels*kernelH*kernelW + oc*kernelH*kernelW

				// For each input position
				for inH := 0; inH < inHeight; inH++ {
					for inW := 0; inW < inWidth; inW++ {
						// Calculate corresponding output position with output padding
						outH := inH*strideH - padH + outputPadH
						outW := inW*strideW - padW + outputPadW

						// Get input value
						inIdx := icOffset + inH*inWidth + inW
						inVal := input[inIdx]

						// Apply kernel
						for kh := 0; kh < kernelH; kh++ {
							for kw := 0; kw < kernelW; kw++ {
								// Calculate output position
								oh := outH + kh
								ow := outW + kw

								// Check bounds (output padding may cause positions outside valid range)
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

// SeparableConv2D performs separable 2D convolution (depthwise + pointwise)
// This is an optimized version that combines depthwise and pointwise convolutions
// into a single operation for efficiency.
// dst: output [batchSize, channels, outHeight, outWidth]
// src: input [batchSize, channels, height, width]
// depthwiseKernel: depthwise kernels [channels, kernelH, kernelW]
// pointwiseKernel: pointwise kernel [channels, channels, 1, 1] (can be reshaped to [channels, channels])
// bias: bias terms [channels] (optional, can be nil)
// outHeight = (height + 2*padH - kernelH) / strideH + 1
// outWidth = (width + 2*padW - kernelW) / strideW + 1
func SeparableConv2D(
	dst, src, depthwiseKernel, pointwiseKernel, bias []float32,
	batchSize, channels, height, width int,
	kernelH, kernelW, strideH, strideW, padH, padW int,
) {
	if batchSize == 0 || channels == 0 || height == 0 || width == 0 {
		return
	}

	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	// Temporary buffer for depthwise output: [batchSize, channels, outHeight, outWidth]
	depthwiseOutput := Pool.Get(batchSize * channels * outHeight * outWidth)
	defer Pool.Put(depthwiseOutput)

	// Step 1: Perform depthwise convolution
	DepthwiseConv2D(depthwiseOutput, src, depthwiseKernel, nil,
		batchSize, channels, height, width,
		kernelH, kernelW, strideH, strideW, padH, padW)

	// Step 2: Perform pointwise convolution (1x1 convolution)
	// This is equivalent to a matrix multiplication across channels
	for b := 0; b < batchSize; b++ {
		depthwiseBatchOffset := b * channels * outHeight * outWidth
		dstBatchOffset := b * channels * outHeight * outWidth

		for outH := 0; outH < outHeight; outH++ {
			for outW := 0; outW < outWidth; outW++ {
				// For each output channel
				for oc := 0; oc < channels; oc++ {
					var sum float32

					// For each input channel (pointwise convolution)
					for ic := 0; ic < channels; ic++ {
						// Get depthwise output value
						depthwiseIdx := depthwiseBatchOffset + ic*outHeight*outWidth + outH*outWidth + outW
						depthwiseVal := depthwiseOutput[depthwiseIdx]

						// Get pointwise kernel weight: [channels, channels, 1, 1]
						// pointwiseKernel[oc][ic][0][0] = pointwiseKernel[oc*channels + ic]
						pointwiseIdx := oc*channels + ic
						weight := pointwiseKernel[pointwiseIdx]

						sum += depthwiseVal * weight
					}

					// Add bias if provided
					if bias != nil {
						sum += bias[oc]
					}

					// Store result
					dstIdx := dstBatchOffset + oc*outHeight*outWidth + outH*outWidth + outW
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// Conv3DTransposed performs transposed 3D convolution (deconvolution)
// This is the inverse operation of Conv3D, extended to 3D spatial dimensions.
// dst: output [batchSize, outChannels, outDepth, outHeight, outWidth]
// src: input [batchSize, inChannels, inDepth, inHeight, inWidth]
// kernel: transposed kernels [inChannels, outChannels, kernelD, kernelH, kernelW]
// bias: bias terms [outChannels] (optional, can be nil)
func Conv3DTransposed(
	dst, src, kernel, bias []float32,
	batchSize, inChannels, outChannels int,
	inDepth, inHeight, inWidth int,
	outDepth, outHeight, outWidth int,
	kernelD, kernelH, kernelW int,
	strideD, strideH, strideW int,
	padD, padH, padW int,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 {
		return
	}

	// Initialize output to zero
	outputSize := batchSize * outChannels * outDepth * outHeight * outWidth
	for i := 0; i < outputSize; i++ {
		dst[i] = 0
	}

	// For each batch
	for b := 0; b < batchSize; b++ {
		batchInOffset := b * inChannels * inDepth * inHeight * inWidth
		batchOutOffset := b * outChannels * outDepth * outHeight * outWidth

		// For each input channel
		for ic := 0; ic < inChannels; ic++ {
			icOffset := batchInOffset + ic*inDepth*inHeight*inWidth

			// For each output channel
			for oc := 0; oc < outChannels; oc++ {
				ocOffset := batchOutOffset + oc*outDepth*outHeight*outWidth

				// Get weights for this filter: [inChannels, outChannels, kernelD, kernelH, kernelW]
				weightOffset := ic*outChannels*kernelD*kernelH*kernelW + oc*kernelD*kernelH*kernelW

				// For each input position
				for inD := 0; inD < inDepth; inD++ {
					for inH := 0; inH < inHeight; inH++ {
						for inW := 0; inW < inWidth; inW++ {
							// Calculate corresponding output position
							outD := inD*strideD - padD
							outH := inH*strideH - padH
							outW := inW*strideW - padW

							// Get input value
							inIdx := icOffset + inD*inHeight*inWidth + inH*inWidth + inW
							inVal := src[inIdx]

							// Apply kernel
							for kd := 0; kd < kernelD; kd++ {
								for kh := 0; kh < kernelH; kh++ {
									for kw := 0; kw < kernelW; kw++ {
										// Calculate output position
										od := outD + kd
										oh := outH + kh
										ow := outW + kw

										// Check bounds
										if od >= 0 && od < outDepth && oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth {
											// Get weight
											weightIdx := weightOffset + kd*kernelH*kernelW + kh*kernelW + kw
											weight := kernel[weightIdx]

											// Accumulate to output
											outIdx := ocOffset + od*outHeight*outWidth + oh*outWidth + ow
											dst[outIdx] += inVal * weight
										}
									}
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
				ocOffset := b*outChannels*outDepth*outHeight*outWidth + oc*outDepth*outHeight*outWidth
				for outD := 0; outD < outDepth; outD++ {
					for outH := 0; outH < outHeight; outH++ {
						for outW := 0; outW < outWidth; outW++ {
							outIdx := ocOffset + outD*outHeight*outWidth + outH*outWidth + outW
							dst[outIdx] += biasVal
						}
					}
				}
			}
		}
	}
}
