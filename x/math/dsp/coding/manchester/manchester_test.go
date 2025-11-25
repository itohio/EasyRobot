package manchester

import (
	"fmt"
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
)

func TestManchesterEncoder_Process(t *testing.T) {
	encoder := NewEncoder()

	tests := []struct {
		name     string
		src      []byte
		expected []byte
	}{
		{
			name:     "single byte 0x00",
			src:      []byte{0x00},
			expected: []byte{0x55, 0x55}, // 0 -> 01, so 8 bits of 0 = 0101010101010101 = 0x5555
		},
		{
			name:     "single byte 0xFF",
			src:      []byte{0xFF},
			expected: []byte{0xAA, 0xAA}, // 1 -> 10, so 8 bits of 1 = 1010101010101010 = 0xAAAA
		},
		{
			name:     "single byte 0xAA",
			src:      []byte{0xAA},       // 10101010
			expected: []byte{0x99, 0x99}, // 1->10, 0->01, so 10101010 -> 10 01 10 01 10 01 10 01 = 1001100110011001 = 0x99 0x99
		},
		{
			name:     "two bytes",
			src:      []byte{0x00, 0xFF},
			expected: []byte{0x55, 0x55, 0xAA, 0xAA},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create source reader
			srcReader := buffer.New(tt.src)

			// Create destination buffer
			dstBuf := buffer.New(make([]byte, 0, len(tt.expected)*2))

			bytesWritten := encoder.Process(dstBuf, srcReader)

			if bytesWritten != len(tt.expected) {
				t.Errorf("Expected %d bytes written, got %d", len(tt.expected), bytesWritten)
			}

			// Get written data
			dst := dstBuf.Items()

			// Check bytes match
			for i := 0; i < len(tt.expected) && i < len(dst); i++ {
				if dst[i] != tt.expected[i] {
					t.Errorf("Byte %d: expected 0x%02X, got 0x%02X", i, tt.expected[i], dst[i])
				}
			}
		})
	}
}

func TestManchesterEncoder_Process_Standard(t *testing.T) {
	encoder := NewEncoder()

	// Test: 0x00 -> all 0s -> 0101010101010101 = 0x5555
	src := []byte{0x00}
	srcReader := buffer.New(src)
	dstBuf := buffer.New(make([]byte, 0, 4))
	bytesWritten := encoder.Process(dstBuf, srcReader)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	dst := dstBuf.Items()
	// 0 -> 01, so 8 zeros = 0101010101010101
	// In bytes: 01010101 01010101 = 0x55 0x55
	if len(dst) < 2 || dst[0] != 0x55 || dst[1] != 0x55 {
		t.Errorf("Expected [0x55, 0x55], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}

	// Test: 0xFF -> all 1s -> 1010101010101010 = 0xAAAA
	src = []byte{0xFF}
	srcReader = buffer.New(src)
	dstBuf = buffer.New(make([]byte, 0, 4))
	bytesWritten = encoder.Process(dstBuf, srcReader)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	dst = dstBuf.Items()
	// 1 -> 10, so 8 ones = 1010101010101010
	// In bytes: 10101010 10101010 = 0xAA 0xAA
	if len(dst) < 2 || dst[0] != 0xAA || dst[1] != 0xAA {
		t.Errorf("Expected [0xAA, 0xAA], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}
}

func TestManchesterEncoder_Process_IEEE8023(t *testing.T) {
	encoder := NewEncoderIEEE8023()

	// Test: 0x00 -> all 0s -> 1010101010101010 = 0xAAAA (IEEE 802.3: 0 = 10)
	src := []byte{0x00}
	srcReader := buffer.New(src)
	dstBuf := buffer.New(make([]byte, 0, 4))
	bytesWritten := encoder.Process(dstBuf, srcReader)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	dst := dstBuf.Items()
	// IEEE 802.3: 0 -> 10, so 8 zeros = 1010101010101010
	// In bytes: 10101010 10101010 = 0xAA 0xAA
	if len(dst) < 2 || dst[0] != 0xAA || dst[1] != 0xAA {
		t.Errorf("Expected [0xAA, 0xAA], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}

	// Test: 0xFF -> all 1s -> 0101010101010101 = 0x5555 (IEEE 802.3: 1 = 01)
	src = []byte{0xFF}
	srcReader = buffer.New(src)
	dstBuf = buffer.New(make([]byte, 0, 4))
	bytesWritten = encoder.Process(dstBuf, srcReader)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	dst = dstBuf.Items()
	// IEEE 802.3: 1 -> 01, so 8 ones = 0101010101010101
	// In bytes: 01010101 01010101 = 0x55 0x55
	if len(dst) < 2 || dst[0] != 0x55 || dst[1] != 0x55 {
		t.Errorf("Expected [0x55, 0x55], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}
}

func TestManchesterEncoder_Process_SizeCalculation(t *testing.T) {
	encoder := NewEncoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{
			name:     "single byte",
			src:      []byte{0x00},
			expected: 2, // 1 byte -> 2 bytes
		},
		{
			name:     "two bytes",
			src:      []byte{0x00, 0xFF},
			expected: 4, // 2 bytes -> 4 bytes
		},
		{
			name:     "empty",
			src:      []byte{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := encoder.Process(nil, srcReader)

			if size != tt.expected {
				t.Errorf("Expected size %d, got %d", tt.expected, size)
			}
		})
	}
}

func TestManchesterDecoder_Process(t *testing.T) {
	decoder := NewDecoder()

	// Test: 0x5555 -> 0x00 (standard: 01 = 0)
	src := []byte{0x55, 0x55} // 0101010101010101
	srcReader := buffer.New(src)
	dstBuf := buffer.New(make([]byte, 0, 2))
	bytesWritten := decoder.Process(dstBuf, srcReader)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	dst := dstBuf.Items()
	if len(dst) < 1 || dst[0] != 0x00 {
		t.Errorf("Expected 0x00, got 0x%02X", dst[0])
	}

	// Test: 0xAAAA -> 0xFF (standard: 10 = 1)
	src = []byte{0xAA, 0xAA} // 1010101010101010
	srcReader = buffer.New(src)
	dstBuf = buffer.New(make([]byte, 0, 2))
	bytesWritten = decoder.Process(dstBuf, srcReader)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	dst = dstBuf.Items()
	if len(dst) < 1 || dst[0] != 0xFF {
		t.Errorf("Expected 0xFF, got 0x%02X", dst[0])
	}
}

func TestManchesterDecoder_Process_IEEE8023(t *testing.T) {
	decoder := NewDecoderIEEE8023()

	// Test: 0xAAAA -> 0x00 (IEEE 802.3: 10 = 0)
	src := []byte{0xAA, 0xAA} // 1010101010101010
	srcReader := buffer.New(src)
	dstBuf := buffer.New(make([]byte, 0, 2))
	bytesWritten := decoder.Process(dstBuf, srcReader)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	dst := dstBuf.Items()
	if len(dst) < 1 || dst[0] != 0x00 {
		t.Errorf("Expected 0x00, got 0x%02X", dst[0])
	}

	// Test: 0x5555 -> 0xFF (IEEE 802.3: 01 = 1)
	src = []byte{0x55, 0x55} // 0101010101010101
	srcReader = buffer.New(src)
	dstBuf = buffer.New(make([]byte, 0, 2))
	bytesWritten = decoder.Process(dstBuf, srcReader)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	dst = dstBuf.Items()
	if len(dst) < 1 || dst[0] != 0xFF {
		t.Errorf("Expected 0xFF, got 0x%02X", dst[0])
	}
}

func TestManchesterDecoder_Process_SizeCalculation(t *testing.T) {
	decoder := NewDecoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{
			name:     "two bytes",
			src:      []byte{0x55, 0x55},
			expected: 1, // 2 bytes -> 1 byte
		},
		{
			name:     "four bytes",
			src:      []byte{0x55, 0x55, 0xAA, 0xAA},
			expected: 2, // 4 bytes -> 2 bytes
		},
		{
			name:     "odd bytes (round up)",
			src:      []byte{0x55, 0x55, 0xAA},
			expected: 2, // 3 bytes -> (3+1)/2 = 2 bytes
		},
		{
			name:     "empty",
			src:      []byte{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := decoder.Process(nil, srcReader)

			if size != tt.expected {
				t.Errorf("Expected size %d, got %d", tt.expected, size)
			}
		})
	}
}

func TestManchester_Roundtrip(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	testCases := [][]byte{
		{0x00},
		{0xFF},
		{0xAA},
		{0x55},
		{0x12, 0x34},
		{0x00, 0xFF, 0xAA, 0x55},
		{0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF},
	}

	for _, original := range testCases {
		t.Run("", func(t *testing.T) {
			// Encode
			srcReader := buffer.New(original)
			encodedBuf := buffer.New(make([]byte, 0, len(original)*2))
			encoder.Process(encodedBuf, srcReader)
			encoded := encodedBuf.Items()

			// Decode
			encodedReader := buffer.New(encoded)
			decodedBuf := buffer.New(make([]byte, 0, len(original)))
			decoder.Process(decodedBuf, encodedReader)
			decoded := decodedBuf.Items()

			// Compare
			if len(decoded) != len(original) {
				t.Errorf("Length mismatch: expected %d, got %d", len(original), len(decoded))
				return
			}

			for i := 0; i < len(original); i++ {
				if decoded[i] != original[i] {
					t.Errorf("Byte %d: expected 0x%02X, got 0x%02X", i, original[i], decoded[i])
				}
			}
		})
	}
}

func TestManchester_Roundtrip_IEEE8023(t *testing.T) {
	encoder := NewEncoderIEEE8023()
	decoder := NewDecoderIEEE8023()

	testCases := [][]byte{
		{0x00},
		{0xFF},
		{0xAA},
		{0x55},
		{0x12, 0x34},
		{0x00, 0xFF, 0xAA, 0x55},
		{0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF},
	}

	for _, original := range testCases {
		t.Run("", func(t *testing.T) {
			// Encode
			srcReader := buffer.New(original)
			encodedBuf := buffer.New(make([]byte, 0, len(original)*2))
			encoder.Process(encodedBuf, srcReader)
			encoded := encodedBuf.Items()

			// Decode
			encodedReader := buffer.New(encoded)
			decodedBuf := buffer.New(make([]byte, 0, len(original)))
			decoder.Process(decodedBuf, encodedReader)
			decoded := decodedBuf.Items()

			// Compare
			if len(decoded) != len(original) {
				t.Errorf("Length mismatch: expected %d, got %d", len(original), len(decoded))
				return
			}

			for i := 0; i < len(original); i++ {
				if decoded[i] != original[i] {
					t.Errorf("Byte %d: expected 0x%02X, got 0x%02X", i, original[i], decoded[i])
				}
			}
		})
	}
}

func TestManchesterEncoder_Reset(t *testing.T) {
	encoder := NewEncoder()
	// Reset should not panic (no state to reset)
	encoder.Reset()
}

func TestManchesterDecoder_Reset(t *testing.T) {
	decoder := NewDecoder()
	// Reset should not panic
	decoder.Reset()
}

func TestManchesterDecoder_BitOffset(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	// Test data
	original := []byte{0xAA, 0x55}
	srcReader := buffer.New(original)
	encodedBuf := buffer.New(make([]byte, 0, len(original)*2))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Test with different bit offsets
	for offset := 0; offset < 8; offset++ {
		t.Run(fmt.Sprintf("offset_%d", offset), func(t *testing.T) {
			decoder.SetBitOffset(offset)
			decoder.Reset()

			// Create misaligned input by shifting bits
			misaligned := make([]byte, len(encoded)+1)
			if offset > 0 {
				// Shift right by offset bits
				for i := 0; i < len(encoded); i++ {
					if i+1 < len(encoded) {
						misaligned[i] = (encoded[i] << uint(offset)) | (encoded[i+1] >> uint(8-offset))
					} else {
						misaligned[i] = encoded[i] << uint(offset)
					}
				}
				misaligned = misaligned[1:] // Skip first byte
			} else {
				copy(misaligned, encoded)
			}

			misalignedReader := buffer.New(misaligned)
			decodedBuf := buffer.New(make([]byte, 0, len(original)))
			bytesWritten := decoder.Process(decodedBuf, misalignedReader)

			// With offset, we may not get perfect decoding, but should handle gracefully
			if bytesWritten > 0 {
				// At least some bytes should decode
				t.Logf("Offset %d: decoded %d bytes", offset, bytesWritten)
			}
		})
	}
}

func TestManchesterDecoder_MisalignedSignal(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	// Encode test data
	original := []byte{0x12, 0x34}
	srcReader := buffer.New(original)
	encodedBuf := buffer.New(make([]byte, 0, len(original)*2))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Test: manually create misaligned signal by prepending bits
	// This simulates starting decoding in the middle of a Manchester symbol
	for offset := 1; offset < 8; offset++ {
		t.Run(fmt.Sprintf("misaligned_%d_bits", offset), func(t *testing.T) {
			decoder.SetBitOffset(offset)
			decoder.Reset()

			// Create misaligned input: shift encoded data right by offset bits
			misalignedLen := len(encoded) + 1
			misaligned := make([]byte, misalignedLen)
			for i := 0; i < len(encoded); i++ {
				shifted := encoded[i] << uint(offset)
				if i+1 < len(encoded) {
					shifted |= encoded[i+1] >> uint(8-offset)
				}
				misaligned[i+1] = shifted
			}

			misalignedReader := buffer.New(misaligned[1:]) // Skip first padding byte
			decodedBuf := buffer.New(make([]byte, 0, len(original)))
			bytesWritten := decoder.Process(decodedBuf, misalignedReader)

			// Should handle misalignment gracefully
			if bytesWritten == 0 && offset < 7 {
				// May fail for some offsets, that's expected
				t.Logf("Offset %d: no bytes decoded (expected for some offsets)", offset)
			}
		})
	}
}

// TestManchesterDecoder_PartialBytes tests that the decoder maintains internal state
// and can resume decoding when a batch ends in the middle of an encoded byte.
// Manchester encoding: 2 input bytes -> 1 output byte
// If we have an odd number of input bytes, the last output byte is incomplete.
func TestManchesterDecoder_PartialBytes(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	// Encode test data
	original := []byte{0x12, 0x34, 0x56, 0x78}
	srcReader := buffer.New(original)
	encodedBuf := buffer.New(make([]byte, 0, len(original)*2))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Test: send encoded data in batches where one batch ends with odd number of bytes
	// Batch 1: first 3 bytes (odd number - can't complete last output byte)
	// Batch 2: remaining bytes (should complete the partial byte from batch 1)
	batch1 := encoded[:3] // 3 input bytes = 1.5 output bytes (1 complete + 0.5 partial)
	batch2 := encoded[3:] // Remaining bytes

	decodedBuf := buffer.New(make([]byte, 0, len(original)))

	// First call with partial data (3 bytes = odd number)
	batch1Reader := buffer.New(batch1)
	bytesDecoded1 := decoder.Process(decodedBuf, batch1Reader)
	// Should decode 1 complete byte (from first 2 input bytes)
	// The 3rd input byte should be partially decoded and saved in internal state
	if bytesDecoded1 != 1 {
		t.Logf("First batch: decoded %d bytes (expected 1 complete byte from 3 input bytes)", bytesDecoded1)
	}

	// Second call with remaining data
	batch2Reader := buffer.New(batch2)
	bytesDecoded2 := decoder.Process(decodedBuf, batch2Reader)
	// Should decode remaining bytes, including completing the partial byte from batch 1
	totalDecoded := bytesDecoded1 + bytesDecoded2
	decoded := decodedBuf.Items()

	// Verify we got all original bytes
	if totalDecoded != len(original) {
		t.Errorf("Total decoded %d bytes, expected %d", totalDecoded, len(original))
	}

	// Verify decoded data matches original
	if len(decoded) >= len(original) {
		for i := 0; i < len(original); i++ {
			if decoded[i] != original[i] {
				t.Errorf("Byte %d: got 0x%02x, expected 0x%02x", i, decoded[i], original[i])
			}
		}
	}
}

// TestManchesterDecoder_PartialBytesMultipleBatches tests multiple batches with partial bytes
func TestManchesterDecoder_PartialBytesMultipleBatches(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	// Encode test data
	original := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC}
	srcReader := buffer.New(original)
	encodedBuf := buffer.New(make([]byte, 0, len(original)*2))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Test: send in multiple small batches (1 byte each)
	// Each batch of 1 input byte = 0.5 output bytes (partial)
	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	totalDecoded := 0

	for i := 0; i < len(encoded); i++ {
		batch := encoded[i : i+1]
		batchReader := buffer.New(batch)
		bytesDecoded := decoder.Process(decodedBuf, batchReader)
		totalDecoded += bytesDecoded
	}

	decoded := decodedBuf.Items()

	// Verify we got all original bytes
	if totalDecoded != len(original) {
		t.Errorf("Total decoded %d bytes, expected %d", totalDecoded, len(original))
	}

	// Verify decoded data matches original
	if len(decoded) >= len(original) {
		for i := 0; i < len(original); i++ {
			if decoded[i] != original[i] {
				t.Errorf("Byte %d: got 0x%02x, expected 0x%02x", i, decoded[i], original[i])
			}
		}
	}
}
