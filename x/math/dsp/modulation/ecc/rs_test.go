package ecc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRSEncoder_NewRSEncoder(t *testing.T) {
	tests := []struct {
		name    string
		n, k    int
		wantErr bool
	}{
		{"valid RS(255, 223)", 255, 223, false},
		{"valid RS(255, 239)", 255, 239, false},
		{"valid RS(255, 200)", 255, 200, false},
		{"invalid n <= k", 10, 10, true},
		{"invalid n < k", 10, 15, true},
		{"invalid k = 0", 255, 0, true},
		{"invalid n > 255", 256, 200, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantErr {
				assert.Panics(t, func() {
					NewRSEncoder(tt.n, tt.k)
				})
			} else {
				encoder := NewRSEncoder(tt.n, tt.k)
				require.NotNil(t, encoder)
				assert.Equal(t, tt.n, encoder.N())
				assert.Equal(t, tt.k, encoder.K())
				assert.Equal(t, (tt.n-tt.k)/2, encoder.T())
			}
		})
	}
}

func TestRSEncoder_Encode(t *testing.T) {
	encoder := NewRSEncoder(255, 223)

	tests := []struct {
		name string
		src  []byte
	}{
		{"empty", []byte{}},
		{"single byte", []byte{0x12}},
		{"small data", []byte{0x12, 0x34, 0x56}},
		{"exact block", make([]byte, 223)},
		{"multiple blocks", make([]byte, 500)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test nil dst (size query)
			expectedSize := encoder.Encode(nil, tt.src)
			if len(tt.src) == 0 {
				assert.Equal(t, 0, expectedSize)
			} else {
				// Only complete blocks are encoded (no padding - encoder doesn't add padding)
				blocks := len(tt.src) / encoder.K()
				expected := blocks * encoder.N()
				assert.Equal(t, expected, expectedSize)
			}

			// Test actual encoding
			if len(tt.src) > 0 {
				dst := make([]byte, expectedSize)
				written := encoder.Encode(dst, tt.src)
				assert.Equal(t, expectedSize, written)
				assert.Greater(t, written, 0)
			}
		})
	}
}

func TestRSEncoder_Encode_DestinationTooSmall(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	src := make([]byte, 100)

	// Destination too small
	dst := make([]byte, 10)
	written := encoder.Encode(dst, src)
	assert.Equal(t, 0, written)
}

func TestRSDecoder_NewRSDecoder(t *testing.T) {
	tests := []struct {
		name    string
		n, k    int
		wantErr bool
	}{
		{"valid RS(255, 223)", 255, 223, false},
		{"valid RS(255, 239)", 255, 239, false},
		{"invalid n <= k", 10, 10, true},
		{"invalid n < k", 10, 15, true},
		{"invalid k = 0", 255, 0, true},
		{"invalid n > 255", 256, 200, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantErr {
				assert.Panics(t, func() {
					NewRSDecoder(tt.n, tt.k)
				})
			} else {
				decoder := NewRSDecoder(tt.n, tt.k)
				require.NotNil(t, decoder)
				assert.Equal(t, tt.n, decoder.N())
				assert.Equal(t, tt.k, decoder.K())
				assert.Equal(t, (tt.n-tt.k)/2, decoder.T())
			}
		})
	}
}

func TestRSDecoder_Decode(t *testing.T) {
	decoder := NewRSDecoder(255, 223)

	tests := []struct {
		name string
		src  []byte
	}{
		{"empty", []byte{}},
		{"incomplete block", make([]byte, 100)},
		{"single block", make([]byte, 255)},
		{"multiple blocks", make([]byte, 510)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test nil dst (size query)
			expectedSize := decoder.Decode(nil, tt.src)
			blocks := len(tt.src) / decoder.N()
			expected := blocks * decoder.K()
			assert.Equal(t, expected, expectedSize)

			// Test actual decoding
			if len(tt.src) >= decoder.N() {
				dst := make([]byte, expectedSize)
				written := decoder.Decode(dst, tt.src)
				assert.GreaterOrEqual(t, written, 0)
			}
		})
	}
}

func TestRSDecoder_Decode_DestinationTooSmall(t *testing.T) {
	decoder := NewRSDecoder(255, 223)
	src := make([]byte, 255) // One complete block

	// Destination too small
	dst := make([]byte, 0)
	written := decoder.Decode(dst, src)
	assert.Equal(t, 0, written)
}

func TestRSEncoder_Reset(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	// Reset should not panic
	encoder.Reset()
}

func TestRSDecoder_Reset(t *testing.T) {
	decoder := NewRSDecoder(255, 223)
	// Reset should not panic
	decoder.Reset()
}

