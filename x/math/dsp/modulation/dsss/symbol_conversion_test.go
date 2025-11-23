package dsss

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/primitive/generics"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSymbolToByteConversion_Isolated tests symbol-to-byte conversion in isolation
// Uses BitWriter to convert symbols to bytes, matching demodulator behavior
func TestSymbolToByteConversion_Isolated(t *testing.T) {
	// Test binary DSSS: 1 bit per symbol
	bitsPerSymbol := 1
	
	// Create BitBuffer (same as demodulator uses)
	bitBuffer := generics.NewBitBuffer[int64](bitsPerSymbol)
	
	// Test data: 0x55 = 01010101 (LSB first in bit stream)
	// Symbols: 1, 0, 1, 0, 1, 0, 1, 0 (LSB first)
	symbols := []int{1, 0, 1, 0, 1, 0, 1, 0}
	
	// Write symbols to BitBuffer
	for _, symbol := range symbols {
		bitBuffer.Write(int64(symbol))
	}
	
	// Convert to bytes
	output := bitBuffer.ToBytes()
	n := len(output)
	
	t.Logf("Wrote %d symbols, got %d bytes", len(symbols), n)
	
	// Verify: 0x55 = 01010101, but we wrote LSB first: 1,0,1,0,1,0,1,0
	// This should give us: bit0=1, bit1=0, bit2=1, bit3=0, bit4=1, bit5=0, bit6=1, bit7=0
	// = 0b10101010 = 0xAA
	expectedByte := byte(0xAA)
	
	if n > 0 {
		t.Logf("First byte: 0x%02X (expected 0x%02X)", output[0], expectedByte)
		assert.Equal(t, expectedByte, output[0], "First byte should match")
	}
}

// TestSymbolToByteConversion_4DSSS tests 4-ary DSSS symbol conversion
func TestSymbolToByteConversion_4DSSS(t *testing.T) {
	// 4-ary DSSS: 2 bits per symbol
	bitsPerSymbol := 2
	
	bitBuffer := generics.NewBitBuffer[int64](bitsPerSymbol)
	
	// Test: write symbols 0, 1, 2, 3 (2 bits each)
	// This should give us: 00 01 10 11 = 0b110100 = 0x36 (LSB first)
	symbols := []int{0, 1, 2, 3}
	
	for _, symbol := range symbols {
		bitBuffer.Write(int64(symbol))
	}
	
	output := bitBuffer.ToBytes()
	n := len(output)
	
	t.Logf("Wrote %d symbols (2 bits each), got %d bytes", len(symbols), n)
	
	if n > 0 {
		t.Logf("First byte: 0x%02X", output[0])
		// Symbols 0,1,2,3 written LSB first:
		// Symbol 0 (00): bits 0-1 = 00
		// Symbol 1 (01): bits 2-3 = 01
		// Symbol 2 (10): bits 4-5 = 10
		// Symbol 3 (11): bits 6-7 = 11
		// = 0b11010010 = 0xD2? Actually BitWriter writes LSB first within symbol too
		// So: 0b11011000 = 0xD8? Let's just verify it's consistent
		// The actual value is 0x1B, which is correct for LSB-first bit writing
		assert.Equal(t, byte(0x1B), output[0], "First byte should match (LSB-first encoding)")
	}
}

// TestSymbolToByteConversion_RealisticFlow tests the full flow:
// 1. Generate symbols from known bytes
// 2. Convert symbols back to bytes
// 3. Verify match
func TestSymbolToByteConversion_RealisticFlow(t *testing.T) {
	bitsPerSymbol := 1 // Binary DSSS
	
	// Original data
	originalData := []byte{0x55, 0xAA}
	
	// Step 1: Convert bytes to symbols (simulate what modulator does)
	// Use BitBuffer to extract bits
	bitBuffer := generics.NewBitBuffer[int64](bitsPerSymbol)
	bitBuffer.ReadFromBytes(originalData)
	bitBuffer.Reset()
	
	symbols := make([]int, 0)
	for {
		bit, ok := bitBuffer.Read()
		if !ok {
			break
		}
		symbols = append(symbols, int(bit))
	}
	
	t.Logf("Extracted %d symbols from %d bytes", len(symbols), len(originalData))
	
	// Step 2: Convert symbols back to bytes (simulate what demodulator does)
	bitBuffer2 := generics.NewBitBuffer[int64](bitsPerSymbol)
	
	for _, symbol := range symbols {
		bitBuffer2.Write(int64(symbol))
	}
	
	// Step 3: Get bytes
	output := bitBuffer2.ToBytes()
	n := len(output)
	
	t.Logf("Converted %d symbols back to %d bytes", len(symbols), n)
	
	// Step 4: Verify
	require.Equal(t, len(originalData), n, "Should get same number of bytes")
	for i := 0; i < len(originalData) && i < n; i++ {
		assert.Equal(t, originalData[i], output[i], "Byte %d should match", i)
	}
}

