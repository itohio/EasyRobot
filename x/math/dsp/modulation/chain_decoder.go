package modulation

// DecoderChain chains multiple ByteDecoders together in reverse order.
// Decoding flow: data -> decoderN -> ... -> decoder2 -> decoder1 -> output
// This undoes the encoding chain (e.g., manchester -> reed solomon -> decryption).
//
// The decoders must be provided in reverse order of the encoders:
//   - If encoders were: encryption -> reed solomon -> manchester
//   - Decoders should be: manchester -> reed solomon -> decryption
type DecoderChain struct {
	decoders []ByteDecoder
	buffers  [][]byte // Intermediate buffers for chaining
}

// NewDecoderChain creates a chain decoder that applies multiple ByteDecoders in sequence.
// The decoders are applied in the order provided (which should be reverse of encoding order):
//   - First decoder processes the input data
//   - Each subsequent decoder processes the output of the previous decoder
//   - Final output is from the last decoder
//
// Example: NewDecoderChain(manchester, reedSolomon, decryption)
//
//	Flow: data -> manchester -> reedSolomon -> decryption -> output
func NewDecoderChain(decoders ...ByteDecoder) *DecoderChain {
	if len(decoders) == 0 {
		panic("ChainDecoder: at least one decoder is required")
	}

	return &DecoderChain{
		decoders: decoders,
		buffers:  make([][]byte, len(decoders)),
	}
}

// Decode decodes bytes through all chained decoders in sequence.
// Decode implements modulation.ByteDecoder interface.
// Returns the number of bytes written to dst.
// Returns expected dst length if dst is nil.
func (c *DecoderChain) Decode(dst, src []byte) int {
	if len(src) == 0 {
		return 0
	}

	if dst == nil {
		// Calculate expected output length by chaining through all decoders
		// Each decoder may change the size, so we chain the size calculations
		expectedSize := len(src)
		for i, decoder := range c.decoders {
			// Query decoder for expected output size
			var tempSize int
			if i == 0 {
				// First decoder: use actual src data
				tempSize = decoder.Decode(nil, src)
			} else {
				// Subsequent decoders: use dummy data of the right size
				// Fill with pattern to avoid all zeros
				dummy := make([]byte, expectedSize)
				for j := range dummy {
					dummy[j] = byte(j % 256)
				}
				tempSize = decoder.Decode(nil, dummy)
			}
			if tempSize == 0 && expectedSize > 0 {
				// If decoder returns 0 for nil dst, estimate conservatively
				// Most decoders halve size (e.g., Manchester)
				expectedSize = expectedSize / 2
				if expectedSize == 0 {
					return 0
				}
			} else {
				expectedSize = tempSize
			}
		}
		return expectedSize
	}

	// Chain through all decoders
	currentInput := src
	lastOutputSize := len(src)

	for i, decoder := range c.decoders {
		isLast := i == len(c.decoders)-1

		// Determine buffer size for this stage
		var stageBuffer []byte
		if isLast {
			// Last decoder writes to dst
			stageBuffer = dst
		} else {
			// Intermediate stages need their own buffer
			// Estimate size: decoding typically reduces size (e.g., Manchester halves)
			estimatedSize := lastOutputSize
			if len(c.buffers[i]) < estimatedSize {
				c.buffers[i] = make([]byte, estimatedSize)
			}
			stageBuffer = c.buffers[i]
		}

		// Decode through current decoder
		bytesWritten := decoder.Decode(stageBuffer, currentInput)
		if bytesWritten == 0 {
			expected := decoder.Decode(nil, currentInput)
			if expected == 0 && !isLast {
				continue
			}
			return 0
		}

		// Update for next iteration
		if isLast {
			// Final output is written to dst
			return bytesWritten
		}
		// Intermediate output becomes next input
		currentInput = stageBuffer[:bytesWritten]
		lastOutputSize = bytesWritten
	}

	return 0 // Should never reach here
}

// Reset resets all chained decoders.
// Reset implements modulation.ByteDecoder interface.
func (c *DecoderChain) Reset() {
	for _, decoder := range c.decoders {
		decoder.Reset()
	}
	// Clear buffers (keep allocated capacity)
	for i := range c.buffers {
		c.buffers[i] = c.buffers[i][:0]
	}
}
