package ecc

import (
	"testing"

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

func TestCRCEncoder_Encode(t *testing.T) {
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
			expectedSize := tt.encoder.Encode(nil, tt.src)
			crcBytes := tt.encoder.Width() / 8
			expected := len(tt.src) + crcBytes
			assert.Equal(t, expected, expectedSize)

			// Test actual encoding
			dst := make([]byte, expectedSize)
			written := tt.encoder.Encode(dst, tt.src)
			assert.Equal(t, expectedSize, written)
			if len(tt.src) > 0 {
				assert.Greater(t, written, len(tt.src), "Should include CRC bytes")
			}
		})
	}
}

func TestCRCEncoder_Encode_DestinationTooSmall(t *testing.T) {
	encoder := NewCRC8Encoder()
	src := []byte{0x12, 0x34}

	// Destination too small
	dst := make([]byte, 1) // Need at least len(src) + 1
	written := encoder.Encode(dst, src)
	assert.Equal(t, 0, written)
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

func TestCRCDecoder_Decode(t *testing.T) {
	tests := []struct {
		name    string
		decoder *CRCDecoder
		src     []byte
	}{
		{"CRC-8 too short", NewCRC8Decoder(), []byte{}},
		{"CRC-8 incomplete", NewCRC8Decoder(), []byte{0x12}},
		{"CRC-16 too short", NewCRC16Decoder(), []byte{}},
		{"CRC-16 incomplete", NewCRC16Decoder(), []byte{0x12}},
		{"CRC-32 too short", NewCRC32Decoder(), []byte{}},
		{"CRC-32 incomplete", NewCRC32Decoder(), []byte{0x12}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test nil dst (size query)
			expectedSize := tt.decoder.Decode(nil, tt.src)
			crcBytes := tt.decoder.Width() / 8
			if len(tt.src) < crcBytes {
				assert.Equal(t, 0, expectedSize)
			} else {
				expected := len(tt.src) - crcBytes
				assert.Equal(t, expected, expectedSize)
			}
		})
	}
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
