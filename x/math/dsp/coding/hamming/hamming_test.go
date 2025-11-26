package hamming

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHammingEncoder_NewEncoder(t *testing.T) {
	encoder := NewEncoder()
	require.NotNil(t, encoder)
}

func TestHammingEncoder_Process(t *testing.T) {
	tests := []struct {
		name string
		src  []byte
	}{
		{"empty", []byte{}},
		{"single byte", []byte{0x12}},
		{"multiple bytes", []byte{0x12, 0x34, 0x56}},
		{"all zeros", []byte{0x00, 0x00}},
		{"all ones", []byte{0xFF, 0xFF}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder := NewEncoder()

			// Test nil dst (size query)
			srcReader := buffer.New(tt.src)
			expectedSize := encoder.Process(nil, srcReader)
			expected := len(tt.src) * 2 // Each input byte → 2 output bytes
			assert.Equal(t, expected, expectedSize)

			// Test actual encoding
			srcReader = buffer.New(tt.src)
			dstBuf := buffer.New(make([]byte, 0, expectedSize*2))
			written := encoder.Process(dstBuf, srcReader)
			assert.Equal(t, expectedSize, written)

			// Verify output size
			dst := dstBuf.Items()
			assert.Equal(t, expectedSize, len(dst))
		})
	}
}

func TestHammingEncoder_Process_SizeCalculation(t *testing.T) {
	encoder := NewEncoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"empty", []byte{}, 0},
		{"single byte", []byte{0x12}, 2}, // 1 byte → 2 bytes
		{"multiple bytes", []byte{0x12, 0x34, 0x56}, 6}, // 3 bytes → 6 bytes
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := encoder.Process(nil, srcReader)
			assert.Equal(t, tt.expected, size)
		})
	}
}

func TestHammingDecoder_NewDecoder(t *testing.T) {
	decoder := NewDecoder()
	require.NotNil(t, decoder)
}

func TestHammingDecoder_Process(t *testing.T) {
	// First encode some data to get valid Hamming-encoded data
	encoder := NewEncoder()
	testData := []byte{0x12, 0x34, 0x56}
	srcReader := buffer.New(testData)
	encodedBuf := buffer.New(make([]byte, 0, len(testData)*2))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Now decode it
	decoder := NewDecoder()
	encodedReader := buffer.New(encoded)
	decodedBuf := buffer.New(make([]byte, 0, len(testData)))
	written := decoder.Process(decodedBuf, encodedReader)
	assert.Equal(t, len(testData), written)

	decoded := decodedBuf.Items()
	assert.Equal(t, testData, decoded)
}

func TestHammingDecoder_Process_SizeCalculation(t *testing.T) {
	decoder := NewDecoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"too short", []byte{}, 0},
		{"incomplete", []byte{0x12}, 0}, // Need at least 2 bytes (2 codewords)
		{"valid", []byte{0x12, 0x34}, 1}, // 2 bytes → 1 byte
		{"multiple", []byte{0x12, 0x34, 0x56, 0x78}, 2}, // 4 bytes → 2 bytes
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := decoder.Process(nil, srcReader)
			assert.Equal(t, tt.expected, size)
		})
	}
}

func TestHammingDecoder_Process_SingleBitError(t *testing.T) {
	// Encode data
	encoder := NewEncoder()
	testData := []byte{0x12}
	srcReader := buffer.New(testData)
	encodedBuf := buffer.New(make([]byte, 0, 10))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Introduce a single-bit error in the first codeword
	// Flip one bit
	encoded[0] ^= 0x01

	// Decode should correct the error
	decoder := NewDecoder()
	encodedReader := buffer.New(encoded)
	decodedBuf := buffer.New(make([]byte, 0, 10))
	written := decoder.Process(decodedBuf, encodedReader)
	assert.Equal(t, len(testData), written)

	decoded := decodedBuf.Items()
	assert.Equal(t, testData, decoded, "Should correct single-bit error")
}

func TestHammingEncoder_Reset(t *testing.T) {
	encoder := NewEncoder()
	// Reset should not panic
	encoder.Reset()
}

func TestHammingDecoder_Reset(t *testing.T) {
	decoder := NewDecoder()
	// Reset should not panic
	decoder.Reset()
}

