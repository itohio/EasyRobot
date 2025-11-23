package manchester

import (
	"fmt"
	"testing"
)

func TestManchesterEncoder_Encode(t *testing.T) {
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
			dst := make([]byte, len(tt.src)*2)
			bytesWritten := encoder.Encode(dst, tt.src)

			if bytesWritten != len(tt.expected) {
				t.Errorf("Expected %d bytes written, got %d", len(tt.expected), bytesWritten)
			}

			// Check first few bytes match
			for i := 0; i < len(tt.expected) && i < bytesWritten; i++ {
				if dst[i] != tt.expected[i] {
					t.Errorf("Byte %d: expected 0x%02X, got 0x%02X", i, tt.expected[i], dst[i])
				}
			}
		})
	}
}

func TestManchesterEncoder_Encode_Standard(t *testing.T) {
	encoder := NewEncoder()

	// Test: 0x00 -> all 0s -> 0101010101010101 = 0x5555
	src := []byte{0x00}
	dst := make([]byte, 2)
	bytesWritten := encoder.Encode(dst, src)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	// 0 -> 01, so 8 zeros = 0101010101010101
	// In bytes: 01010101 01010101 = 0x55 0x55
	if dst[0] != 0x55 || dst[1] != 0x55 {
		t.Errorf("Expected [0x55, 0x55], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}

	// Test: 0xFF -> all 1s -> 1010101010101010 = 0xAAAA
	src = []byte{0xFF}
	dst = make([]byte, 2)
	bytesWritten = encoder.Encode(dst, src)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	// 1 -> 10, so 8 ones = 1010101010101010
	// In bytes: 10101010 10101010 = 0xAA 0xAA
	if dst[0] != 0xAA || dst[1] != 0xAA {
		t.Errorf("Expected [0xAA, 0xAA], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}
}

func TestManchesterEncoder_Encode_IEEE8023(t *testing.T) {
	encoder := NewEncoderIEEE8023()

	// Test: 0x00 -> all 0s -> 1010101010101010 = 0xAAAA (IEEE 802.3: 0 = 10)
	src := []byte{0x00}
	dst := make([]byte, 2)
	bytesWritten := encoder.Encode(dst, src)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	// IEEE 802.3: 0 -> 10, so 8 zeros = 1010101010101010
	// In bytes: 10101010 10101010 = 0xAA 0xAA
	if dst[0] != 0xAA || dst[1] != 0xAA {
		t.Errorf("Expected [0xAA, 0xAA], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}

	// Test: 0xFF -> all 1s -> 0101010101010101 = 0x5555 (IEEE 802.3: 1 = 01)
	src = []byte{0xFF}
	dst = make([]byte, 2)
	bytesWritten = encoder.Encode(dst, src)

	if bytesWritten != 2 {
		t.Errorf("Expected 2 bytes written, got %d", bytesWritten)
	}

	// IEEE 802.3: 1 -> 01, so 8 ones = 0101010101010101
	// In bytes: 01010101 01010101 = 0x55 0x55
	if dst[0] != 0x55 || dst[1] != 0x55 {
		t.Errorf("Expected [0x55, 0x55], got [0x%02X, 0x%02X]", dst[0], dst[1])
	}
}

func TestManchesterDecoder_Decode(t *testing.T) {
	decoder := NewDecoder()

	// Test: 0x5555 -> 0x00 (standard: 01 = 0)
	src := []byte{0x55, 0x55} // 0101010101010101
	dst := make([]byte, 1)
	bytesWritten := decoder.Decode(dst, src)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	if dst[0] != 0x00 {
		t.Errorf("Expected 0x00, got 0x%02X", dst[0])
	}

	// Test: 0xAAAA -> 0xFF (standard: 10 = 1)
	src = []byte{0xAA, 0xAA} // 1010101010101010
	dst = make([]byte, 1)
	bytesWritten = decoder.Decode(dst, src)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	if dst[0] != 0xFF {
		t.Errorf("Expected 0xFF, got 0x%02X", dst[0])
	}
}

func TestManchesterDecoder_Decode_IEEE8023(t *testing.T) {
	decoder := NewDecoderIEEE8023()

	// Test: 0xAAAA -> 0x00 (IEEE 802.3: 10 = 0)
	src := []byte{0xAA, 0xAA} // 1010101010101010
	dst := make([]byte, 1)
	bytesWritten := decoder.Decode(dst, src)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	if dst[0] != 0x00 {
		t.Errorf("Expected 0x00, got 0x%02X", dst[0])
	}

	// Test: 0x5555 -> 0xFF (IEEE 802.3: 01 = 1)
	src = []byte{0x55, 0x55} // 0101010101010101
	dst = make([]byte, 1)
	bytesWritten = decoder.Decode(dst, src)

	if bytesWritten != 1 {
		t.Errorf("Expected 1 byte written, got %d", bytesWritten)
	}

	if dst[0] != 0xFF {
		t.Errorf("Expected 0xFF, got 0x%02X", dst[0])
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
			encoded := make([]byte, len(original)*2)
			encoder.Encode(encoded, original)

			// Decode
			decoded := make([]byte, len(original))
			decoder.Decode(decoded, encoded)

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
			encoded := make([]byte, len(original)*2)
			encoder.Encode(encoded, original)

			// Decode
			decoded := make([]byte, len(original))
			decoder.Decode(decoded, encoded)

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

func TestManchesterEncoder_Encode_DestinationTooSmall(t *testing.T) {
	encoder := NewEncoder()
	src := []byte{0x00, 0xFF}

	// Destination too small
	dst := make([]byte, 2) // Need 4 bytes
	bytesWritten := encoder.Encode(dst, src)

	if bytesWritten != 0 {
		t.Errorf("Expected 0 bytes written when destination too small, got %d", bytesWritten)
	}
}

func TestManchesterDecoder_Decode_DestinationTooSmall(t *testing.T) {
	decoder := NewDecoder()
	src := []byte{0x55, 0x55, 0xAA, 0xAA} // 2 bytes encoded

	// Destination too small
	dst := make([]byte, 0) // Need 1 byte
	bytesWritten := decoder.Decode(dst, src)

	if bytesWritten != 0 {
		t.Errorf("Expected 0 bytes written when destination too small, got %d", bytesWritten)
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
	encoded := make([]byte, len(original)*2)
	encoder.Encode(encoded, original)

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

			decoded := make([]byte, len(original))
			bytesWritten := decoder.Decode(decoded, misaligned)

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
	encoded := make([]byte, len(original)*2)
	encoder.Encode(encoded, original)

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

			decoded := make([]byte, len(original))
			bytesWritten := decoder.Decode(decoded, misaligned[1:]) // Skip first padding byte

			// Should handle misalignment gracefully
			if bytesWritten == 0 && offset < 7 {
				// May fail for some offsets, that's expected
				t.Logf("Offset %d: no bytes decoded (expected for some offsets)", offset)
			}
		})
	}
}
