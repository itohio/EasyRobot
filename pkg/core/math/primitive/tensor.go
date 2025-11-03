package primitive

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
