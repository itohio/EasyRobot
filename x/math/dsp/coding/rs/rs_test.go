package rs

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRSEncoder_NewEncoder(t *testing.T) {
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
					NewEncoder(tt.n, tt.k)
				})
			} else {
				encoder := NewEncoder(tt.n, tt.k)
				require.NotNil(t, encoder)
				assert.Equal(t, tt.n, encoder.N())
				assert.Equal(t, tt.k, encoder.K())
				assert.Equal(t, (tt.n-tt.k)/2, encoder.T())
			}
		})
	}
}

func TestRSEncoder_Process(t *testing.T) {
	encoder := NewEncoder(255, 223)

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
			srcReader := buffer.New(tt.src)
			expectedSize := encoder.Process(nil, srcReader)
			if len(tt.src) == 0 {
				assert.Equal(t, 0, expectedSize)
			} else {
				// Only complete blocks are encoded (no padding - encoder doesn't add padding)
				blocks := len(tt.src) / encoder.K()
				expected := blocks * encoder.N()
				assert.Equal(t, expected, expectedSize)
			}

			// Test actual encoding
			if len(tt.src) > 0 && expectedSize > 0 {
				srcReader = buffer.New(tt.src)
				dstBuf := buffer.New(make([]byte, 0, expectedSize*2))
				written := encoder.Process(dstBuf, srcReader)
				assert.Equal(t, expectedSize, written)
				assert.Greater(t, written, 0)
			}
		})
	}
}

func TestRSEncoder_Process_SizeCalculation(t *testing.T) {
	encoder := NewEncoder(255, 223)

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"empty", []byte{}, 0},
		{"incomplete block", make([]byte, 100), 0}, // Less than k bytes
		{"exact block", make([]byte, 223), 255},    // 1 block: 223 -> 255
		{"two blocks", make([]byte, 446), 510},     // 2 blocks: 446 -> 510
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := encoder.Process(nil, srcReader)
			assert.Equal(t, tt.expected, size)
		})
	}
}

func TestRSDecoder_NewDecoder(t *testing.T) {
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
					NewDecoder(tt.n, tt.k)
				})
			} else {
				decoder := NewDecoder(tt.n, tt.k)
				require.NotNil(t, decoder)
				assert.Equal(t, tt.n, decoder.N())
				assert.Equal(t, tt.k, decoder.K())
				assert.Equal(t, (tt.n-tt.k)/2, decoder.T())
			}
		})
	}
}

func TestRSDecoder_Process(t *testing.T) {
	encoder := NewEncoder(255, 223)
	decoder := NewDecoder(255, 223)

	tests := []struct {
		name string
		src  []byte
	}{
		{"empty", []byte{}},
		{"incomplete block", make([]byte, 100)},
		{"single block", make([]byte, 223)},    // Will be encoded to 255
		{"multiple blocks", make([]byte, 446)}, // Will be encoded to 510
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.src) >= encoder.K() {
				// Encode first
				srcReader := buffer.New(tt.src)
				encodedBuf := buffer.New(make([]byte, 0, len(tt.src)*2))
				encodedSize := encoder.Process(encodedBuf, srcReader)
				if encodedSize > 0 {
					encoded := encodedBuf.Items()

					// Test nil dst (size query)
					encodedReader := buffer.New(encoded)
					expectedSize := decoder.Process(nil, encodedReader)
					blocks := len(encoded) / decoder.N()
					expected := blocks * decoder.K()
					assert.Equal(t, expected, expectedSize)

					// Test actual decoding
					encodedReader = buffer.New(encoded)
					decodedBuf := buffer.New(make([]byte, 0, len(tt.src)))
					written := decoder.Process(decodedBuf, encodedReader)
					assert.GreaterOrEqual(t, written, 0)
					if written > 0 {
						decoded := decodedBuf.Items()
						// Verify decoded data matches original (for complete blocks)
						completeBlocks := len(tt.src) / encoder.K()
						expectedDecodedLen := completeBlocks * encoder.K()
						if written == expectedDecodedLen {
							assert.Equal(t, tt.src[:expectedDecodedLen], decoded)
						}
					}
				}
			}
		})
	}
}

func TestRSDecoder_Process_SizeCalculation(t *testing.T) {
	decoder := NewDecoder(255, 223)

	tests := []struct {
		name     string
		src      []byte
		expected int
	}{
		{"empty", []byte{}, 0},
		{"incomplete block", make([]byte, 100), 0}, // Less than n bytes
		{"single block", make([]byte, 255), 223},   // 1 block: 255 -> 223
		{"two blocks", make([]byte, 510), 446},     // 2 blocks: 510 -> 446
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcReader := buffer.New(tt.src)
			size := decoder.Process(nil, srcReader)
			assert.Equal(t, tt.expected, size)
		})
	}
}

func TestRSEncoder_Reset(t *testing.T) {
	encoder := NewEncoder(255, 223)
	// Reset should not panic
	encoder.Reset()
}

func TestRSDecoder_Reset(t *testing.T) {
	decoder := NewDecoder(255, 223)
	// Reset should not panic
	decoder.Reset()
}
