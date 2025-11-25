package ecc

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCRCEncoder_NewCRC8Encoder(t *testing.T) {
	encoder := NewCRC8Encoder()
	require.NotNil(t, encoder)
	assert.Equal(t, 8, encoder.Width())
}

func TestCRCEncoder_NewCRC16Encoder(t *testing.T) {
	encoder := NewCRC16Encoder()
	require.NotNil(t, encoder)
	assert.Equal(t, 16, encoder.Width())
}

func TestCRCEncoder_NewCRC32Encoder(t *testing.T) {
	encoder := NewCRC32Encoder()
	require.NotNil(t, encoder)
	assert.Equal(t, 32, encoder.Width())
}

func TestCRCEncoder_Process(t *testing.T) {
	tests := []struct {
		name    string
		encoder *CRCEncoder
		src     []byte
	}{
		{"CRC-8 empty", NewCRC8Encoder(), []byte{}},
		{"CRC-8 single byte", NewCRC8Encoder(), []byte{0x12}},
		{"CRC-8 multiple bytes", NewCRC8Encoder(), []byte{0x12, 0x34, 0x56}},
		{"CRC-16 empty", NewCRC16Encoder(), []byte{}},
		{"CRC-16 single byte", NewCRC16Encoder(), []byte{0x12}},
		{"CRC-32 empty", NewCRC32Encoder(), []byte{}},
		{"CRC-32 multiple bytes", NewCRC32Encoder(), []byte{0x12, 0x34, 0x56, 0x78}},
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
	encoder := NewCRC8Encoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"empty", []byte{}, 1}, // CRC-8: 1 byte
		{"single byte", []byte{0x12}, 2}, // 1 data + 1 CRC
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
	decoder8 := NewCRC8Decoder()
	require.NotNil(t, decoder8)
	assert.Equal(t, 8, decoder8.Width())

	decoder16 := NewCRC16Decoder()
	require.NotNil(t, decoder16)
	assert.Equal(t, 16, decoder16.Width())

	decoder32 := NewCRC32Decoder()
	require.NotNil(t, decoder32)
	assert.Equal(t, 32, decoder32.Width())
}

func TestCRCDecoder_Process(t *testing.T) {
	// First encode some data to get valid CRC-encoded data
	encoder := NewCRC8Encoder()
	testData := []byte{0x12, 0x34, 0x56}
	srcReader := buffer.New(testData)
	encodedBuf := buffer.New(make([]byte, 0, len(testData)*2))
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Now decode it
	decoder := NewCRC8Decoder()
	encodedReader := buffer.New(encoded)
	decodedBuf := buffer.New(make([]byte, 0, len(testData)))
	written := decoder.Process(decodedBuf, encodedReader)
	assert.Equal(t, len(testData), written)

	decoded := decodedBuf.Items()
	assert.Equal(t, testData, decoded)
}

func TestCRCDecoder_Process_SizeCalculation(t *testing.T) {
	decoder := NewCRC8Decoder()

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"too short", []byte{}, 0},
		{"incomplete", []byte{0x12}, 0}, // Need at least 1 CRC byte
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
	decoder := NewCRC8Decoder()
	// Create invalid CRC data (corrupted)
	invalidData := []byte{0x12, 0x34, 0x56, 0x99} // Last byte is wrong CRC
	srcReader := buffer.New(invalidData)
	decodedBuf := buffer.New(make([]byte, 0, 10))
	written := decoder.Process(decodedBuf, srcReader)
	// Should return 0 because CRC validation failed
	assert.Equal(t, 0, written)
}

func TestCRCEncoder_Reset(t *testing.T) {
	encoder := NewCRC8Encoder()
	// Reset should not panic
	encoder.Reset()
}

func TestCRCDecoder_Reset(t *testing.T) {
	decoder := NewCRC8Decoder()
	// Reset should not panic
	decoder.Reset()
}

