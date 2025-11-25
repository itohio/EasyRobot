package ecc

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNRZI_Roundtrip(t *testing.T) {
	encoder := NewNRZIEncoder()
	decoder := NewNRZIDecoder()

	testCases := [][]byte{
		{0x12},
		{0x12, 0x34},
		{0x12, 0x34, 0x56, 0x78},
		make([]byte, 100),
		{0x00, 0xFF, 0xAA, 0x55},
	}

	// Initialize test data
	for i := range testCases {
		if len(testCases[i]) > 0 {
			for j := range testCases[i] {
				if i < 4 {
					testCases[i][j] = byte(i + j)
				}
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

func TestNRZI_Transitions(t *testing.T) {
	encoder := NewNRZIEncoder()
	decoder := NewNRZIDecoder()

	// Test that '1' causes transition, '0' maintains state
	testData := []byte{0x01} // Binary: 00000001 (only LSB is 1)
	
	encoder.Reset()
	srcReader := buffer.New(testData)
	encodedBuf := buffer.New(make([]byte, 0, 10))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()
	
	// The encoded byte should have transition only at LSB
	// Since we start with low state, and LSB=1 causes transition,
	// the LSB should be high (1), rest should be low (0)
	assert.Equal(t, byte(0x01), encoded[0], "LSB=1 should cause transition to high")
	
	// Decode and verify
	decoder.Reset()
	encodedReader := buffer.New(encoded)
	decodedBuf := buffer.New(make([]byte, 0, 10))
	decoder.Process(decodedBuf, encodedReader)
	decoded := decodedBuf.Items()
	
	assert.Equal(t, testData, decoded, "Should decode back to original")
}

