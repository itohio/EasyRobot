package ecc

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestRS_Streaming_Roundtrip tests roundtrip with streaming data
// that may have padding and unknown boundaries.
func TestRS_Streaming_Roundtrip(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	decoder := NewRSDecoder(255, 223)

	testCases := []struct {
		name    string
		data    []byte
		padding []byte
	}{
		{
			name:    "small data with padding",
			data:    []byte{0x12, 0x34, 0x56},
			padding: []byte{0xFF, 0xAA, 0x55, 0x00},
		},
		{
			name:    "exact block with padding",
			data:    make([]byte, 223),
			padding: make([]byte, 100),
		},
		{
			name:    "multiple blocks with padding",
			data:    make([]byte, 500),
			padding: make([]byte, 200),
		},
	}

	// Initialize test data
	for i := range testCases {
		for j := range testCases[i].data {
			testCases[i].data[j] = byte(i + j)
		}
		for j := range testCases[i].padding {
			testCases[i].padding[j] = byte(0xAA + j)
		}
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			// Encode
			expectedSize := encoder.Encode(nil, tt.data)
			encoded := make([]byte, expectedSize)
			encoder.Encode(encoded, tt.data)

			// Create stream with padding before and after
			stream := make([]byte, len(tt.padding)+len(encoded)+len(tt.padding))
			copy(stream, tt.padding)
			copy(stream[len(tt.padding):], encoded)
			copy(stream[len(tt.padding)+len(encoded):], tt.padding)

			// Decode stream
			decoded := make([]byte, len(tt.data)*2)
			bytesDecoded := decoder.Decode(decoded, stream)

			// Should decode the original data (may have some extra from padding if it looks valid)
			assert.Greater(t, bytesDecoded, 0, "Should decode some bytes from stream")

			// Verify original data is present in decoded output
			found := false
			for i := 0; i <= bytesDecoded-len(tt.data); i++ {
				match := true
				for j := 0; j < len(tt.data); j++ {
					if decoded[i+j] != tt.data[j] {
						match = false
						break
					}
				}
				if match {
					found = true
					break
				}
			}
			assert.True(t, found, "Original data should be found in decoded output")
		})
	}
}

// TestRS_Streaming_Chunked tests decoding in chunks (simulating real streaming).
func TestRS_Streaming_Chunked(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	decoder := NewRSDecoder(255, 223)

	original := make([]byte, 446) // 2 blocks
	for i := range original {
		original[i] = byte(i % 256)
	}

	// Encode
	expectedSize := encoder.Encode(nil, original)
	encoded := make([]byte, expectedSize)
	encoder.Encode(encoded, original)

	// Add padding
	padding := make([]byte, 50)
	stream := make([]byte, len(padding)+len(encoded)+len(padding))
	copy(stream, padding)
	copy(stream[len(padding):], encoded)
	copy(stream[len(padding)+len(encoded):], padding)

	// Decode in chunks
	decoded := make([]byte, len(original)*2)
	totalDecoded := 0
	chunkSize := 100

	for i := 0; i < len(stream); i += chunkSize {
		end := i + chunkSize
		if end > len(stream) {
			end = len(stream)
		}
		chunk := stream[i:end]

		remaining := decoded[totalDecoded:]
		bytesDecoded := decoder.Decode(remaining, chunk)
		totalDecoded += bytesDecoded
	}

	// Should decode the original data
	assert.Greater(t, totalDecoded, 0, "Should decode bytes from chunked stream")

	// Verify original data is present
	found := false
	for i := 0; i <= totalDecoded-len(original); i++ {
		match := true
		for j := 0; j < len(original); j++ {
			if decoded[i+j] != original[j] {
				match = false
				break
			}
		}
		if match {
			found = true
			break
		}
	}
	assert.True(t, found, "Original data should be found in chunked decoded output")
}

// TestRS_Streaming_ArbitraryOffset tests decoding when data starts at arbitrary offset.
func TestRS_Streaming_ArbitraryOffset(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	decoder := NewRSDecoder(255, 223)

	original := make([]byte, 223)
	for i := range original {
		original[i] = byte(i)
	}

	// Encode
	expectedSize := encoder.Encode(nil, original)
	encoded := make([]byte, expectedSize)
	encoder.Encode(encoded, original)

	// Test with different offsets
	for offset := 1; offset < 50; offset += 5 {
		t.Run("", func(t *testing.T) {
			decoder.Reset()

			// Create stream with offset
			prefix := make([]byte, offset)
			for i := range prefix {
				prefix[i] = byte(0xFF)
			}
			stream := make([]byte, len(prefix)+len(encoded))
			copy(stream, prefix)
			copy(stream[len(prefix):], encoded)

			// Decode
			decoded := make([]byte, len(original)*2)
			bytesDecoded := decoder.Decode(decoded, stream)

			// Should decode the original data despite offset
			assert.Greater(t, bytesDecoded, 0, "Should decode bytes with offset %d", offset)

			// Verify original data is present
			found := false
			for i := 0; i <= bytesDecoded-len(original); i++ {
				match := true
				for j := 0; j < len(original); j++ {
					if decoded[i+j] != original[j] {
						match = false
						break
					}
				}
				if match {
					found = true
					break
				}
			}
			assert.True(t, found, "Original data should be found with offset %d", offset)
		})
	}
}

// TestRS_Streaming_PartialBlocks tests handling of partial blocks at boundaries.
func TestRS_Streaming_PartialBlocks(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	decoder := NewRSDecoder(255, 223)

	original := make([]byte, 223)
	for i := range original {
		original[i] = byte(i)
	}

	// Encode
	expectedSize := encoder.Encode(nil, original)
	encoded := make([]byte, expectedSize)
	encoder.Encode(encoded, original)

	// Send partial block first, then complete block
	partialSize := 100
	partial := encoded[:partialSize]
	complete := encoded[partialSize:]

	decoded := make([]byte, len(original)*2)

	// First call with partial data
	bytesDecoded1 := decoder.Decode(decoded, partial)
	// Should buffer partial data, not decode yet
	assert.Equal(t, 0, bytesDecoded1, "Should not decode partial block")

	// Second call with complete data
	bytesDecoded2 := decoder.Decode(decoded, complete)
	// Should now decode the complete block
	assert.Greater(t, bytesDecoded2, 0, "Should decode after receiving complete block")
	assert.Equal(t, len(original), bytesDecoded2, "Should decode full original data")
	assert.Equal(t, original, decoded[:bytesDecoded2], "Decoded data should match original")
}

// TestRS_Streaming_MultiplePackets tests handling multiple encoded packets in stream.
func TestRS_Streaming_MultiplePackets(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	decoder := NewRSDecoder(255, 223)

	packet1 := make([]byte, 223)
	packet2 := make([]byte, 223)
	for i := range packet1 {
		packet1[i] = byte(i)
		packet2[i] = byte(i + 100)
	}

	// Encode both packets
	encoded1 := make([]byte, encoder.Encode(nil, packet1))
	encoder.Encode(encoded1, packet1)

	encoded2 := make([]byte, encoder.Encode(nil, packet2))
	encoder.Encode(encoded2, packet2)

	// Create stream with padding between packets
	padding := make([]byte, 50)
	stream := make([]byte, len(encoded1)+len(padding)+len(encoded2))
	copy(stream, encoded1)
	copy(stream[len(encoded1):], padding)
	copy(stream[len(encoded1)+len(padding):], encoded2)

	// Decode
	decoded := make([]byte, (len(packet1)+len(packet2))*2)
	bytesDecoded := decoder.Decode(decoded, stream)

	// Should decode both packets
	assert.Greater(t, bytesDecoded, len(packet1), "Should decode at least first packet")

	// Verify both packets are present
	found1 := false
	found2 := false
	for i := 0; i <= bytesDecoded-len(packet1); i++ {
		match := true
		for j := 0; j < len(packet1); j++ {
			if decoded[i+j] != packet1[j] {
				match = false
				break
			}
		}
		if match {
			found1 = true
			break
		}
	}

	for i := 0; i <= bytesDecoded-len(packet2); i++ {
		match := true
		for j := 0; j < len(packet2); j++ {
			if decoded[i+j] != packet2[j] {
				match = false
				break
			}
		}
		if match {
			found2 = true
			break
		}
	}

	assert.True(t, found1, "First packet should be found")
	assert.True(t, found2, "Second packet should be found")
}

// TestRS_Streaming_RandomPadding tests with random padding data.
func TestRS_Streaming_RandomPadding(t *testing.T) {
	encoder := NewRSEncoder(255, 223)
	decoder := NewRSDecoder(255, 223)

	original := make([]byte, 223)
	rng := rand.New(rand.NewSource(42))
	for i := range original {
		original[i] = byte(rng.Intn(256))
	}

	// Encode
	expectedSize := encoder.Encode(nil, original)
	encoded := make([]byte, expectedSize)
	encoder.Encode(encoded, original)

	// Create stream with random padding
	paddingSize := 200
	padding := make([]byte, paddingSize)
	for i := range padding {
		padding[i] = byte(rng.Intn(256))
	}

	stream := make([]byte, len(padding)+len(encoded)+len(padding))
	copy(stream, padding)
	copy(stream[len(padding):], encoded)
	copy(stream[len(padding)+len(encoded):], padding)

	// Decode
	decoded := make([]byte, len(original)*2)
	bytesDecoded := decoder.Decode(decoded, stream)

	// Should decode the original data despite random padding
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes with random padding")

	// Verify original data is present
	found := false
	for i := 0; i <= bytesDecoded-len(original); i++ {
		match := true
		for j := 0; j < len(original); j++ {
			if decoded[i+j] != original[j] {
				match = false
				break
			}
		}
		if match {
			found = true
			break
		}
	}
	assert.True(t, found, "Original data should be found despite random padding")
}

// TestRS_Streaming_Reset tests that Reset clears internal buffer.
func TestRS_Streaming_Reset(t *testing.T) {
	decoder := NewRSDecoder(255, 223)

	// Add some data to buffer
	partial := make([]byte, 100)
	decoder.Decode(make([]byte, 100), partial)

	// Reset should clear buffer
	decoder.Reset()

	// Buffer should be empty
	assert.Equal(t, 0, len(decoder.buffer), "Buffer should be empty after Reset")
}

