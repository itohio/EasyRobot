package ecc

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHamming_Roundtrip(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

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

func TestHamming_Roundtrip_EdgeCases(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	testCases := []struct {
		name string
		data []byte
	}{
		{"all zeros", []byte{0x00, 0x00}},
		{"all ones", []byte{0xFF, 0xFF}},
		{"alternating", []byte{0xAA, 0x55}},
		{"single byte", []byte{0x12}},
		{"large data", make([]byte, 1000)},
	}

	// Initialize test data
	for i := range testCases {
		if len(testCases[i].data) > 0 {
			for j := range testCases[i].data {
				if testCases[i].name == "large data" {
					testCases[i].data[j] = byte(j % 256)
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

