package crc

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCRCEncoder_NewEncoderCRC8(t *testing.T) {
	encoder := NewEncoder()
	require.NotNil(t, encoder)
	assert.Equal(t, 8, encoder.Width())
}

func TestCRCEncoder_NewEncoderCRC16(t *testing.T) {
	encoder := NewEncoder()
	require.NotNil(t, encoder)
	assert.Equal(t, 16, encoder.Width())
}

func TestCRCEncoder_NewEncoderCRC32(t *testing.T) {
	encoder := NewEncoder()
	require.NotNil(t, encoder)
	assert.Equal(t, 32, encoder.Width())
}

func TestCRCEncoder_Process(t *testing.T) {
	tests := []struct {
		name    string
		encoder *Encoder
		src     []byte
	}{
		{"CRC-8 empty", NewEncoder(), []byte{}},
		{"CRC-8 single byte", NewEncoder(), []byte{0x12}},
		{"CRC-8 multiple bytes", NewEncoder(), []byte{0x12, 0x34, 0x56}},
		{"CRC-16 empty", NewEncoder(), []byte{}},
		{"CRC-16 single byte", NewEncoder(), []byte{0x12}},
		{"CRC-32 empty", NewEncoder(), []byte{}},
		{"CRC-32 multiple bytes", NewEncoder(), []byte{0x12, 0x34, 0x56, 0x78}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test nil dst (size query)
			srcReader := buffer.New(tt.src)
			expectedSize := tt.encoder.Process(nil, srcReader)
			crcBytes := tt.encoder.Width() / 8
			expected := len(tt.src) + crcBytes
			assert.Equal(t, expected, expectedSize)

			// Test actual encoding
			srcReader = buffer.New(tt.src)
			dstBuf := buffer.New(make([]byte, 0, expectedSize*2))
			written := tt.encoder.Process(dstBuf, srcReader)
			assert.Equal(t, expectedSize, written)
			if len(tt.src) > 0 {
				assert.Greater(t, written, len(tt.src), "Should include CRC bytes")
			}

			// Verify data is preserved
			dst := dstBuf.Items()
			if len(tt.src) > 0 {
				assert.Equal(t, tt.src, dst[:len(tt.src)], "Data should be preserved")
			}
		})
	}
}

func TestCRCEncoder_Process_SizeCalculation(t *testing.T) {
	encoder := NewEncoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"empty", []byte{}, 1},                          // CRC-8: 1 byte
		{"single byte", []byte{0x12}, 2},                // 1 data + 1 CRC
		{"multiple bytes", []byte{0x12, 0x34, 0x56}, 4}, // 3 data + 1 CRC
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := encoder.Process(nil, srcReader)
			assert.Equal(t, tt.expected, size)
		})
	}
}

func TestCRCDecoder_NewCRCDecoder(t *testing.T) {
	decoder8 := NewDecoder()
	require.NotNil(t, decoder8)
	assert.Equal(t, 8, decoder8.Width())

	decoder16 := NewDecoder()
	require.NotNil(t, decoder16)
	assert.Equal(t, 16, decoder16.Width())

	decoder32 := NewDecoder()
	require.NotNil(t, decoder32)
	assert.Equal(t, 32, decoder32.Width())
}

func TestCRCDecoder_Process(t *testing.T) {
	// First encode some data to get valid CRC-encoded data
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

func TestCRCDecoder_Process_SizeCalculation(t *testing.T) {
	decoder := NewDecoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"too short", []byte{}, 0},
		{"incomplete", []byte{0x12}, 0},  // Need at least 1 CRC byte
		{"valid", []byte{0x12, 0x34}, 1}, // 2 bytes - 1 CRC = 1 data
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := decoder.Process(nil, srcReader)
			assert.Equal(t, tt.expected, size)
		})
	}
}

func TestCRCDecoder_Process_InvalidCRC(t *testing.T) {
	decoder := NewDecoder()
	// Create invalid CRC data (corrupted)
	invalidData := []byte{0x12, 0x34, 0x56, 0x99} // Last byte is wrong CRC
	srcReader := buffer.New(invalidData)
	decodedBuf := buffer.New(make([]byte, 0, 10))
	written := decoder.Process(decodedBuf, srcReader)
	// Should return 0 because CRC validation failed
	assert.Equal(t, 0, written)
}

func TestCRCEncoder_Reset(t *testing.T) {
	encoder := NewEncoder()
	// Reset should not panic
	encoder.Reset()
}

func TestCRCDecoder_Reset(t *testing.T) {
	decoder := NewDecoder()
	// Reset should not panic
	decoder.Reset()
}
