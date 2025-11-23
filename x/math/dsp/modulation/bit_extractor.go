package modulation

// BitExtractor converts symbols to bits and bits to bytes
// Reusable component for all digital demodulators
type BitExtractor struct {
	bitBuffer []byte
}

// NewBitExtractor creates a new bit extractor
func NewBitExtractor() *BitExtractor {
	return &BitExtractor{
		bitBuffer: make([]byte, 0, 64),
	}
}

// AddSymbol adds a symbol value and extracts bits based on bits per symbol
func (be *BitExtractor) AddSymbol(symbol int, bitsPerSymbol int) {
	// Convert symbol to bits (MSB first)
	for i := bitsPerSymbol - 1; i >= 0; i-- {
		bit := (symbol >> uint(i)) & 1
		be.bitBuffer = append(be.bitBuffer, byte(bit))
	}
}

// ExtractBytes extracts complete bytes from the bit buffer
// Returns number of bytes extracted
func (be *BitExtractor) ExtractBytes(output []byte) int {
	bytesWritten := 0
	
	for len(be.bitBuffer) >= 8 && bytesWritten < len(output) {
		byteVal := byte(0)
		for i := 0; i < 8; i++ {
			byteVal |= be.bitBuffer[i] << uint(7-i)
		}
		output[bytesWritten] = byteVal
		bytesWritten++
		be.bitBuffer = be.bitBuffer[8:]
	}
	
	return bytesWritten
}

// HasCompleteByte returns true if there's at least one complete byte
func (be *BitExtractor) HasCompleteByte() bool {
	return len(be.bitBuffer) >= 8
}

// Reset resets the bit extractor state
func (be *BitExtractor) Reset() {
	be.bitBuffer = be.bitBuffer[:0]
}

// GetRemainingBits returns the number of remaining bits in the buffer
func (be *BitExtractor) GetRemainingBits() int {
	return len(be.bitBuffer)
}

