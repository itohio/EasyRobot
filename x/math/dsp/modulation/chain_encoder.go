package modulation

// EncoderChain chains multiple ByteEncoders together.
// Encoding flow: data -> encoder1 -> encoder2 -> ... -> encoderN -> output
// This allows combining multiple encoding layers (e.g., encryption -> reed solomon -> manchester).
type EncoderChain struct {
	encoders []ByteEncoder
	buffers  [][]byte // Intermediate buffers for chaining
}

// NewEncoderChain creates a chain encoder that applies multiple ByteEncoders in sequence.
// The encoders are applied in the order provided:
//   - First encoder processes the input data
//   - Each subsequent encoder processes the output of the previous encoder
//   - Final output is from the last encoder
//
// Example: NewEncoderChain(encryption, reedSolomon, manchester)
//
//	Flow: data -> encryption -> reedSolomon -> manchester -> output
func NewEncoderChain(encoders ...ByteEncoder) *EncoderChain {
	if len(encoders) == 0 {
		panic("ChainEncoder: at least one encoder is required")
	}

	return &EncoderChain{
		encoders: encoders,
		buffers:  make([][]byte, len(encoders)),
	}
}

// Encode encodes bytes through all chained encoders in sequence.
// Encode implements modulation.ByteEncoder interface.
// Returns the number of bytes written to dst.
// Returns expected dst length if dst is nil.
func (c *EncoderChain) Encode(dst, src []byte) int {
	if len(src) == 0 {
		return 0
	}

	if dst == nil {
		// Calculate expected output length by chaining through all encoders
		// Each encoder may change the size, so we chain the size calculations
		expectedSize := len(src)
		for i, encoder := range c.encoders {
			// Query encoder for expected output size
			var tempSize int
			if i == 0 {
				// First encoder: use actual src data
				tempSize = encoder.Encode(nil, src)
			} else {
				// Subsequent encoders: use dummy data of the right size
				// Fill with pattern to avoid all zeros (some encoders might reject all zeros)
				dummy := make([]byte, expectedSize)
				for j := range dummy {
					dummy[j] = byte(j % 256)
				}
				tempSize = encoder.Encode(nil, dummy)
			}
			if tempSize == 0 && expectedSize > 0 {
				// If encoder returns 0 for nil dst, estimate conservatively
				// Most encoders double size (e.g., Manchester)
				expectedSize = expectedSize * 2
			} else {
				expectedSize = tempSize
			}
		}
		return expectedSize
	}

	// Chain through all encoders
	currentInput := src
	lastOutputSize := len(src)

	for i, encoder := range c.encoders {
		isLast := i == len(c.encoders)-1

		// Determine buffer size for this stage
		var stageBuffer []byte
		if isLast {
			// Last encoder writes to dst
			stageBuffer = dst
		} else {
			// Intermediate stages need their own buffer
			// Estimate size: conservatively assume 2x for encoders like Manchester
			estimatedSize := lastOutputSize * 2
			if len(c.buffers[i]) < estimatedSize {
				c.buffers[i] = make([]byte, estimatedSize)
			}
			stageBuffer = c.buffers[i]
		}

		// Encode through current encoder
		bytesWritten := encoder.Encode(stageBuffer, currentInput)
		if bytesWritten == 0 {
			expected := encoder.Encode(nil, currentInput)
			if expected == 0 && i > 0 {
				if isLast {
					if len(dst) < len(currentInput) {
						return 0
					}
					copy(dst, currentInput)
					return len(currentInput)
				}
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

// Reset resets all chained encoders.
// Reset implements modulation.ByteEncoder interface.
func (c *EncoderChain) Reset() {
	for _, encoder := range c.encoders {
		encoder.Reset()
	}
	// Clear buffers (keep allocated capacity)
	for i := range c.buffers {
		c.buffers[i] = c.buffers[i][:0]
	}
}
