package modulation

import (
	"testing"
)

// TestBitExtractor_BFSK tests bit extraction for 2-FSK (1 bit per symbol)
func TestBitExtractor_BFSK(t *testing.T) {
	extractor := NewBitExtractor()

	// Test data: 0x48 = 01001000
	// Add symbols: 0, 1, 0, 0, 1, 0, 0, 0
	symbols := []int{0, 1, 0, 0, 1, 0, 0, 0}

	for _, sym := range symbols {
		extractor.AddSymbol(sym, 1) // 1 bit per symbol for 2-FSK
	}

	// Extract bytes
	output := make([]byte, 10)
	bytesWritten := extractor.ExtractBytes(output)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte, got %d", bytesWritten)
	}

	if output[0] != 0x48 {
		t.Errorf("Expected 0x48, got 0x%02X", output[0])
	}

	if extractor.GetRemainingBits() != 0 {
		t.Errorf("Expected 0 remaining bits, got %d", extractor.GetRemainingBits())
	}
}

// TestBitExtractor_MultipleBytes tests extracting multiple bytes
func TestBitExtractor_MultipleBytes(t *testing.T) {
	extractor := NewBitExtractor()

	// Test data: "Hi" = 0x48 0x69 = 01001000 01101001
	// Bits: 0,1,0,0,1,0,0,0, 0,1,1,0,1,0,0,1
	symbols := []int{0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1}

	for _, sym := range symbols {
		extractor.AddSymbol(sym, 1)
	}

	output := make([]byte, 10)
	bytesWritten := extractor.ExtractBytes(output)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes, got %d", bytesWritten)
	}

	if output[0] != 0x48 {
		t.Errorf("Expected first byte 0x48, got 0x%02X", output[0])
	}

	if output[1] != 0x69 {
		t.Errorf("Expected second byte 0x69, got 0x%02X", output[1])
	}
}

// TestBitExtractor_PartialByte tests partial byte handling
func TestBitExtractor_PartialByte(t *testing.T) {
	extractor := NewBitExtractor()

	// Add only 6 bits (not enough for a complete byte)
	symbols := []int{0, 1, 0, 0, 1, 0}

	for _, sym := range symbols {
		extractor.AddSymbol(sym, 1)
	}

	output := make([]byte, 10)
	bytesWritten := extractor.ExtractBytes(output)

	if bytesWritten != 0 {
		t.Errorf("Expected 0 bytes (partial), got %d", bytesWritten)
	}

	if extractor.GetRemainingBits() != 6 {
		t.Errorf("Expected 6 remaining bits, got %d", extractor.GetRemainingBits())
	}

	if extractor.HasCompleteByte() {
		t.Error("Should not have complete byte with only 6 bits")
	}

	// Add 2 more bits to complete the byte
	extractor.AddSymbol(0, 1)
	extractor.AddSymbol(0, 1)

	bytesWritten = extractor.ExtractBytes(output)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte after adding 2 more bits, got %d", bytesWritten)
	}

	if output[0] != 0x48 {
		t.Errorf("Expected 0x48, got 0x%02X", output[0])
	}
}

// TestBitExtractor_4FSK tests bit extraction for 4-FSK (2 bits per symbol)
func TestBitExtractor_4FSK(t *testing.T) {
	extractor := NewBitExtractor()

	// Test data: 0x48 = 01001000
	// Symbols: 0x01, 0x00, 0x10, 0x00 (2 bits each)
	// 0x01 = 01, 0x00 = 00, 0x10 = 10, 0x00 = 00
	// Combined: 01 00 10 00 = 01001000 = 0x48
	symbols := []int{1, 0, 2, 0} // 2 bits per symbol

	for _, sym := range symbols {
		extractor.AddSymbol(sym, 2) // 2 bits per symbol for 4-FSK
	}

	output := make([]byte, 10)
	bytesWritten := extractor.ExtractBytes(output)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte, got %d", bytesWritten)
	}

	if output[0] != 0x48 {
		t.Errorf("Expected 0x48, got 0x%02X", output[0])
	}
}

// TestBitExtractor_PreambleIncluded tests that preamble bits are included in output
func TestBitExtractor_PreambleIncluded(t *testing.T) {
	extractor := NewBitExtractor()

	// Preamble: 16 symbols of all 1s = 16 bits = 2 bytes of 0xFF
	preambleSymbols := make([]int, 16)
	for i := range preambleSymbols {
		preambleSymbols[i] = 1 // All 1s
	}

	// Data: 0x48 = 01001000
	dataSymbols := []int{0, 1, 0, 0, 1, 0, 0, 0}

	// Add preamble + data
	allSymbols := append(preambleSymbols, dataSymbols...)

	for _, sym := range allSymbols {
		extractor.AddSymbol(sym, 1)
	}

	output := make([]byte, 10)
	bytesWritten := extractor.ExtractBytes(output)

	if bytesWritten != 3 {
		t.Errorf("Expected 3 bytes (2 preamble + 1 data), got %d", bytesWritten)
	}

	if output[0] != 0xFF {
		t.Errorf("Expected first byte 0xFF (preamble), got 0x%02X", output[0])
	}

	if output[1] != 0xFF {
		t.Errorf("Expected second byte 0xFF (preamble), got 0x%02X", output[1])
	}

	if output[2] != 0x48 {
		t.Errorf("Expected third byte 0x48 (data), got 0x%02X", output[2])
	}
}

// TestBitExtractor_Reset tests reset functionality
func TestBitExtractor_Reset(t *testing.T) {
	extractor := NewBitExtractor()

	// Add some bits
	symbols := []int{0, 1, 0, 0, 1, 0}
	for _, sym := range symbols {
		extractor.AddSymbol(sym, 1)
	}

	if extractor.GetRemainingBits() != 6 {
		t.Errorf("Expected 6 remaining bits, got %d", extractor.GetRemainingBits())
	}

	extractor.Reset()

	if extractor.GetRemainingBits() != 0 {
		t.Errorf("Expected 0 remaining bits after reset, got %d", extractor.GetRemainingBits())
	}

	if extractor.HasCompleteByte() {
		t.Error("Should not have complete byte after reset")
	}
}
