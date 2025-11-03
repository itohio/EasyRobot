package primitive

// This file implements quantized (INT4/UINT4) versions of BLAS operations
// for ultra-efficient low-precision neural network inference.
//
// Quantization follows the asymmetric scheme:
//   real_value = scale * (quantized_value - zero_point)
//
// All quantized operations use uint8 storage (unpacked), with values clamped to [0,15]
// which represents the range:
//   - For symmetric quantization: [-8, 7] with zero_point = 8
//   - For asymmetric quantization: [min, max] mapped to [0, 15]
//
// Intermediate calculations use int32 to avoid overflow.
//
// Note: Unlike true 4-bit packing where 2 values share a byte, this implementation
// stores each 4-bit value in its own byte for simplicity and compatibility with
// existing BLAS-style interfaces.
//
// Reference: "Efficient Inference with TensorRT-LLM Quantization" (NVIDIA, 2024)

// Copy_Q4 copies quantized vector: y = x
// y: destination uint8 vector
// x: source uint8 vector
// strideY, strideX: access strides
// n: vector length
func Copy_Q4(y, x []uint8, strideY, strideX, n int) {
	if n == 0 {
		return
	}

	py := 0
	px := 0

	for i := 0; i < n; i++ {
		y[py] = x[px]
		py += strideY
		px += strideX
	}
}

// Gemm_NN_Q4 computes quantized matrix multiplication: C = A*B
// This implements the quantized GEMM with zero-point corrections
//
// output: M × N matrix (uint8, stored as row-major, ldOutput ≥ N)
// input: M × K matrix (uint8, row-major, ldInput ≥ K)
// weight: K × N matrix (uint8, row-major, ldWeight ≥ N)
//
// Quantization parameters:
//
//	inputScale, weightScale, outputScale: scales for input, weight, output
//	inputZero, weightZero, outputZero: zero points (int32, range 0-15)
//
// The zero-point correction formula:
//
//	C_int[i,j] = (sum(A_int[i,k] * B_int[k,j])
//	             - inputZero * sum(B_int[k,j])
//	             - weightZero * sum(A_int[i,k])
//	             + inputZero * weightZero * K) * scale
//
// where scale = inputScale * weightScale / outputScale
func Gemm_NN_Q4(
	output, input, weight []uint8,
	ldOutput, ldInput, ldWeight, M, N, K int,
	inputScale, weightScale, outputScale float32,
	inputZero, weightZero, outputZero int32,
) {
	if M == 0 || N == 0 || K == 0 {
		return
	}

	// Compute scale factor for requantization
	scale := (inputScale * weightScale) / outputScale

	// Initialize output to zero
	for i := 0; i < M; i++ {
		pc := i * ldOutput
		for j := 0; j < N; j++ {
			output[pc+j] = uint8(outputZero)
		}
	}

	// Compute quantized GEMM with zero-point corrections
	pi := 0
	po := 0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := int32(0)

			// Compute dot product of row i of input with column j of weight
			pw := 0
			for k := 0; k < K; k++ {
				sum += int32(input[pi+k]) * int32(weight[pw+j])
				pw += ldWeight
			}

			// Apply zero-point corrections
			// Correction 1: -inputZero * sum(weight[k,j] for all k)
			sumB := int32(0)
			pw = 0
			for k := 0; k < K; k++ {
				sumB += int32(weight[pw+j])
				pw += ldWeight
			}
			sum -= inputZero * sumB

			// Correction 2: -weightZero * sum(input[i,k] for all k)
			sumA := int32(0)
			for k := 0; k < K; k++ {
				sumA += int32(input[pi+k])
			}
			sum -= weightZero * sumA

			// Correction 3: +inputZero * weightZero * K
			sum += inputZero * weightZero * int32(K)

			// Requantize: multiply by scale and add output zero-point
			// result = sum * scale + outputZero
			// Since we're working in quantized space, we need to convert to output space
			outputVal := int32(float32(sum)*scale) + outputZero

			// Clamp to uint4 range [0, 15]
			if outputVal < 0 {
				outputVal = 0
			} else if outputVal > 15 {
				outputVal = 15
			}

			output[po+j] = uint8(outputVal)
		}
		pi += ldInput
		po += ldOutput
	}
}

// Gemm_NN_Q4_Accum computes quantized matrix multiplication with int32 accumulator
// This variant outputs int32 values instead of requantizing to uint4
// Useful for layers that need higher precision before quantization
//
// output: M × N matrix (int32, row-major, ldOutput ≥ N) - accumulator buffer
// input: M × K matrix (uint8, row-major, ldInput ≥ K)
// weight: K × N matrix (uint8, row-major, ldWeight ≥ N)
func Gemm_NN_Q4_Accum(
	output []int32, input, weight []uint8,
	ldOutput, ldInput, ldWeight, M, N, K int,
	inputZero, weightZero int32,
) {
	if M == 0 || N == 0 || K == 0 {
		return
	}

	// Initialize output to zero
	for i := 0; i < M; i++ {
		pc := i * ldOutput
		for j := 0; j < N; j++ {
			output[pc+j] = 0
		}
	}

	// Compute quantized GEMM with zero-point corrections
	pi := 0
	po := 0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := int32(0)

			// Compute dot product of row i of input with column j of weight
			pw := 0
			for k := 0; k < K; k++ {
				sum += int32(input[pi+k]) * int32(weight[pw+j])
				pw += ldWeight
			}

			// Apply zero-point corrections
			// Correction 1: -inputZero * sum(weight[k,j] for all k)
			sumB := int32(0)
			pw = 0
			for k := 0; k < K; k++ {
				sumB += int32(weight[pw+j])
				pw += ldWeight
			}
			sum -= inputZero * sumB

			// Correction 2: -weightZero * sum(input[i,k] for all k)
			sumA := int32(0)
			for k := 0; k < K; k++ {
				sumA += int32(input[pi+k])
			}
			sum -= weightZero * sumA

			// Correction 3: +inputZero * weightZero * K
			sum += inputZero * weightZero * int32(K)

			output[po+j] = sum
		}
		pi += ldInput
		po += ldOutput
	}
}

// Conv2D_Q4 performs quantized 2D convolution with batched input
// Uses Im2Col + quantized GEMM approach
//
// output: [batchSize, outChannels, outHeight, outWidth] (uint8)
// input: [batchSize, inChannels, inHeight, inWidth] (uint8)
// weights: [outChannels, inChannels, kernelH, kernelW] (uint8)
// bias: optional bias [outChannels] (int32, can be nil)
//
// Quantization parameters:
//
//	inputScale, weightScale, outputScale: scales
//	inputZero, weightZero, outputZero: zero points
func Conv2D_Q4(
	output, input, weights []uint8,
	batchSize, inChannels, outChannels int,
	inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW int,
	strideH, strideW int,
	padH, padW int,
	bias []int32,
	inputScale, weightScale, outputScale float32,
	inputZero, weightZero, outputZero int32,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 {
		return
	}

	// Calculate dimensions for Im2Col
	kernelSize := inChannels * kernelH * kernelW
	im2colSize := batchSize * outHeight * outWidth

	// Allocate temporary buffer for Im2Col output (uint8)
	im2col := make([]uint8, im2colSize*kernelSize)

	// Step 1: Convert input to columns using Im2Col (uint8 version)
	Im2Col_Q4(im2col, input, batchSize, inChannels, inHeight, inWidth,
		kernelH, kernelW, padH, padW, strideH, strideW)

	// Step 2: Reshape weights for GEMM
	// weights: [outChannels, inChannels, kernelH, kernelW]
	// Reshape to: [outChannels, inChannels*kernelH*kernelW]

	// Step 3: Allocate int32 accumulator for GEMM output
	gemmAccum := make([]int32, outChannels*im2colSize)

	// Step 4: Perform quantized GEMM with accumulator
	// gemmOutput: [outChannels, im2colSize]
	Gemm_NN_Q4_Accum(gemmAccum, weights, im2col,
		im2colSize,  // ldOutput
		kernelSize,  // ldInput (weights)
		kernelSize,  // ldWeight (im2col)
		outChannels, // M
		im2colSize,  // N
		kernelSize,  // K
		weightZero,  // inputZero (weights)
		inputZero,   // weightZero (im2col/input)
	)

	// Step 5: Add bias if provided
	if bias != nil {
		for oc := 0; oc < outChannels; oc++ {
			biasVal := bias[oc]
			for i := 0; i < im2colSize; i++ {
				// Index in gemmAccum: [outChannels, im2colSize]
				gemmIdx := oc*im2colSize + i
				gemmAccum[gemmIdx] += biasVal
			}
		}
	}

	// Step 6: Requantize to output uint8
	scale := (inputScale * weightScale) / outputScale
	for oc := 0; oc < outChannels; oc++ {
		for i := 0; i < im2colSize; i++ {
			gemmIdx := oc*im2colSize + i

			// Requantize
			outputVal := int32(float32(gemmAccum[gemmIdx])*scale) + outputZero

			// Clamp to uint4 range [0, 15]
			if outputVal < 0 {
				outputVal = 0
			} else if outputVal > 15 {
				outputVal = 15
			}

			gemmAccum[gemmIdx] = outputVal
		}
	}

	// Step 7: Reshape and transpose gemmAccum to final output format
	// gemmAccum: [outChannels, batchSize*outHeight*outWidth] (int32, treated as uint8)
	// output: [batchSize, outChannels, outHeight, outWidth] (uint8)
	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Source index in gemmAccum
					gemmIdx := oc*im2colSize + b*outHeight*outWidth + outH*outWidth + outW
					// Destination index in output
					outIdx := b*outChannels*outHeight*outWidth +
						oc*outHeight*outWidth +
						outH*outWidth + outW
					output[outIdx] = uint8(gemmAccum[gemmIdx])
				}
			}
		}
	}
}

// Im2Col_Q4 converts image patches to columns for quantized GEMM-based convolution
// This is the uint8 version of Im2Col (4-bit values stored in uint8)
//
// col: output columns [batchSize*outHeight*outWidth, channels*kernelH*kernelW] (uint8)
// im: input image [batchSize, channels, height, width] (uint8)
// batchSize, channels: batch and channel dimensions
// height, width: input spatial dimensions
// kernelH, kernelW: kernel spatial dimensions
// padH, padW: padding values
// strideH, strideW: stride values
func Im2Col_Q4(
	col, im []uint8,
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
								// Padding: zero (which represents the quantization zero point for zero value)
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

// Col2Im_Q4 converts columns back to image for quantized convolution
// This accumulates uint8 values (used in backpropagation)
//
// im: output image [batchSize, channels, height, width] (uint8)
// col: input columns [batchSize*outHeight*outWidth, channels*kernelH*kernelW] (uint8)
func Col2Im_Q4(
	im, col []uint8,
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

	// Initialize output to zero
	for i := range im {
		im[i] = 0
	}

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

								// Saturating addition for uint8 (clamped to 4-bit range)
								sum := int16(im[imIdx]) + int16(col[colIdx])
								if sum > 15 {
									im[imIdx] = 15
								} else if sum < 0 {
									im[imIdx] = 0
								} else {
									im[imIdx] = uint8(sum)
								}
							}
							colIdx++
						}
					}
				}
			}
		}
	}
}

// GemmBatched_Q4 computes batched quantized matrix multiplication
// This processes multiple quantized matrices simultaneously
//
// output, input, weight: batched matrices (uint8)
// All matrices have the same dimensions within each batch
func GemmBatched_Q4(
	output, input, weight []uint8,
	ldOutput, ldInput, ldWeight, M, N, K int,
	inputScale, weightScale, outputScale float32,
	inputZero, weightZero, outputZero int32,
	batchCount int,
	strideOutput, strideInput, strideWeight int,
) {
	if batchCount == 0 || M == 0 || N == 0 || K == 0 {
		return
	}

	for k := 0; k < batchCount; k++ {
		// Get offsets for batch k
		offsetOutput := k * strideOutput
		offsetInput := k * strideInput
		offsetWeight := k * strideWeight

		// Call quantized GEMM for this batch
		Gemm_NN_Q4(
			output[offsetOutput:],
			input[offsetInput:],
			weight[offsetWeight:],
			ldOutput, ldInput, ldWeight,
			M, N, K,
			inputScale, weightScale, outputScale,
			inputZero, weightZero, outputZero,
		)
	}
}

// Conv2DTransposed_Q4 performs quantized 2D transposed convolution with batched input
// Uses direct implementation without Im2Col
//
// output: [batchSize, outChannels, outHeight, outWidth] (uint8)
// input: [batchSize, inChannels, inHeight, inWidth] (uint8)
// weights: [inChannels, outChannels, kernelH, kernelW] (uint8)
// bias: optional bias [outChannels] (int32, can be nil)
//
// Quantization parameters:
//
//	inputScale, weightScale, outputScale: scales
//	inputZero, weightZero, outputZero: zero points (int32, range 0-15)
func Conv2DTransposed_Q4(
	output, input, weights []uint8,
	batchSize, inChannels, outChannels int,
	inHeight, inWidth int,
	outHeight, outWidth int,
	kernelH, kernelW int,
	strideH, strideW int,
	padH, padW int,
	bias []int32,
	inputScale, weightScale, outputScale float32,
	inputZero, weightZero, outputZero int32,
) {
	if batchSize == 0 || inChannels == 0 || outChannels == 0 {
		return
	}

	// Initialize output to zero
	outputSize := batchSize * outChannels * outHeight * outWidth
	for i := 0; i < outputSize; i++ {
		output[i] = uint8(outputZero)
	}

	// Compute scale factor for requantization
	scale := (inputScale * weightScale) / outputScale

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
									wVal := weights[weightIdx]

									// Accumulate to output
									outIdx := ocOffset + oh*outWidth + ow

									// Compute in int32 to avoid overflow
									accum := int32(output[outIdx])
									product := int32(inVal) * int32(wVal)

									// Apply zero-point corrections
									product = product - inputZero*int32(wVal) - weightZero*int32(inVal) + inputZero*weightZero
									accum += product

									output[outIdx] = uint8(accum)
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
						// Requantize bias addition
						outputVal := int32(output[outIdx]) + biasVal
						// Clamp to uint4 range [0, 15]
						if outputVal < 0 {
							outputVal = 0
						} else if outputVal > 15 {
							outputVal = 15
						}
						output[outIdx] = uint8(outputVal)
					}
				}
			}
		}
	}

	// Final requantization pass
	for i := 0; i < outputSize; i++ {
		outputVal := int32(float32(output[i])*scale) + outputZero
		// Clamp to uint4 range [0, 15]
		if outputVal < 0 {
			outputVal = 0
		} else if outputVal > 15 {
			outputVal = 15
		}
		output[i] = uint8(outputVal)
	}
}
