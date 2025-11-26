package gray

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGray_Roundtrip_8Bit(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	testCases := [][]byte{
		{0x12},
		{0x12, 0x34},
		{0x12, 0x34, 0x56, 0x78},
		make([]byte, 100),
	}

	// Initialize test data
	for i := range testCases {
		if len(testCases[i]) > 0 {
			for j := range testCases[i] {
				testCases[i][j] = byte(i + j)
			}
		}
	}

	for _, original := range testCases {
		t.Run("", func(t *testing.T) {
			if len(original) == 0 {
				return
			}

			encoder.Reset()
			decoder.Reset()

			// Encode
			srcReader := buffer.New(original)
			expectedSize := encoder.Process(nil, srcReader)
			encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
			srcReader = buffer.New(original)
			bytesWritten := encoder.Process(encodedBuf, srcReader)
			require.Equal(t, expectedSize, bytesWritten)
			encoded := encodedBuf.Items()

			// Decode
			encodedReader := buffer.New(encoded)
			expectedDecodedSize := decoder.Process(nil, encodedReader)
			decodedBuf := buffer.New(make([]byte, 0, expectedDecodedSize*2))
			encodedReader = buffer.New(encoded)
			bytesDecoded := decoder.Process(decodedBuf, encodedReader)
			decoded := decodedBuf.Items()

			// Verify
			require.Equal(t, len(original), bytesDecoded, "Should decode all bytes")
			assert.Equal(t, original, decoded, "Decoded data should match original")
		})
	}
}

func TestGray_Roundtrip_4Bit(t *testing.T) {
	encoder := NewEncoderWithBits(4)
	decoder := NewDecoderWithBits(4)

	testCases := [][]byte{
		{0x12},
		{0x12, 0x34},
		{0x12, 0x34, 0x56, 0x78},
		make([]byte, 100),
	}

	// Initialize test data
	for i := range testCases {
		if len(testCases[i]) > 0 {
			for j := range testCases[i] {
				testCases[i][j] = byte(i + j)
			}
		}
	}

	for _, original := range testCases {
		t.Run("", func(t *testing.T) {
			if len(original) == 0 {
				return
			}

			encoder.Reset()
			decoder.Reset()

			// Encode
			srcReader := buffer.New(original)
			expectedSize := encoder.Process(nil, srcReader)
			encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
			srcReader = buffer.New(original)
			bytesWritten := encoder.Process(encodedBuf, srcReader)
			require.Equal(t, expectedSize, bytesWritten)
			encoded := encodedBuf.Items()

			// Decode
			encodedReader := buffer.New(encoded)
			expectedDecodedSize := decoder.Process(nil, encodedReader)
			decodedBuf := buffer.New(make([]byte, 0, expectedDecodedSize*2))
			encodedReader = buffer.New(encoded)
			bytesDecoded := decoder.Process(decodedBuf, encodedReader)
			decoded := decodedBuf.Items()

			// Verify
			require.Equal(t, len(original), bytesDecoded, "Should decode all bytes")
			assert.Equal(t, original, decoded, "Decoded data should match original")
		})
	}
}

func TestGray_Roundtrip_EdgeCases(t *testing.T) {
	encoder := NewEncoder()
	decoder := NewDecoder()

	testCases := []struct {
		name string
		data []byte
	}{
		{"all zeros", []byte{0x00, 0x00}},
		{"all ones", []byte{0xFF, 0xFF}},
		{"alternating", []byte{0xAA, 0x55}},
		{"single byte", []byte{0x12}},
		{"large data", make([]byte, 1000)},
		{"sequential", []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07}},
	}

	// Initialize test data
	for i := range testCases {
		if len(testCases[i].data) > 0 {
			for j := range testCases[i].data {
				if testCases[i].name == "large data" {
					testCases[i].data[j] = byte(j % 256)
				} else if testCases[i].name == "sequential" {
					// Use predefined pattern
				} else {
					// Use predefined patterns
				}
			}
		}
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.data) == 0 {
				return
			}

			encoder.Reset()
			decoder.Reset()

			// Encode
			srcReader := buffer.New(tt.data)
			expectedSize := encoder.Process(nil, srcReader)
			encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
			srcReader = buffer.New(tt.data)
			bytesWritten := encoder.Process(encodedBuf, srcReader)
			require.Equal(t, expectedSize, bytesWritten)
			encoded := encodedBuf.Items()

			// Decode
			encodedReader := buffer.New(encoded)
			decodedBuf := buffer.New(make([]byte, 0, len(tt.data)))
			bytesDecoded := decoder.Process(decodedBuf, encodedReader)
			decoded := decodedBuf.Items()

			// Verify
			require.Equal(t, len(tt.data), bytesDecoded)
			assert.Equal(t, tt.data, decoded)
		})
	}
}

func TestGray_AdjacentValues(t *testing.T) {
	// Test that adjacent values in Gray code differ by only one bit
	encoder := NewEncoder()

	// Test for 8-bit values
	for i := 0; i < 255; i++ {
		val1 := byte(i)
		val2 := byte(i + 1)

		gray1 := encoder.encodeToGray(val1)
		gray2 := encoder.encodeToGray(val2)

		// Calculate Hamming distance (number of differing bits)
		xor := gray1 ^ gray2
		bits := 0
		for xor != 0 {
			bits++
			xor &= xor - 1 // Clear least significant bit
		}

		// Adjacent values in Gray code should differ by exactly 1 bit
		assert.Equal(t, 1, bits, "Adjacent values %d and %d should differ by 1 bit in Gray code", val1, val2)
	}
}

func TestGray_AllBitWidths(t *testing.T) {
	// Test all supported bit widths
	for bits := 1; bits <= 8; bits++ {
		t.Run("", func(t *testing.T) {
		encoder := NewEncoderWithBits(bits)
		decoder := NewDecoderWithBits(bits)

			// Test all possible values for this bit width
			maxVal := (1 << bits) - 1
			for val := 0; val <= maxVal; val++ {
				binary := byte(val)
				gray := encoder.encodeToGray(binary)
				decoded := decoder.decodeFromGray(gray)

				assert.Equal(t, binary, decoded, "Bit width %d, value %d: binary->gray->binary should be identity", bits, val)
			}
		})
	}
}

