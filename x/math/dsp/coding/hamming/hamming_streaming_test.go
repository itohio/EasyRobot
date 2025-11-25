package ecc

import (
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHamming_Streaming_EncodeDecode tests that encoder and decoder
// can handle streaming data by encoding into a buffer, then decoding
// by splitting the encoded data into multiple parts.
func TestHamming_Streaming_EncodeDecode(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	testCases := []struct {
		name string
		data []byte
	}{
		{"small", []byte{0x12, 0x34}},
		{"medium", []byte{0x12, 0x34, 0x56, 0x78, 0x9A}},
		{"large", make([]byte, 100)},
		{"odd_length", []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE}},
	}

	// Initialize test data
	for i := range testCases {
		if len(testCases[i].data) > 0 {
			for j := range testCases[i].data {
				if testCases[i].name == "large" {
					testCases[i].data[j] = byte(j % 256)
				} else {
					testCases[i].data[j] = byte(i + j)
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

			// Step 1: Encode data into a buffer
			srcReader := buffer.New(tt.data)
			expectedSize := encoder.Process(nil, srcReader)
			encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
			srcReader = buffer.New(tt.data)
			bytesWritten := encoder.Process(encodedBuf, srcReader)
			require.Equal(t, expectedSize, bytesWritten)
			encoded := encodedBuf.Items()

			// Step 2: Split encoded data into 3 parts
			part1Size := len(encoded) / 3
			part2Size := len(encoded) / 3
			part3Start := part1Size + part2Size

			part1 := encoded[:part1Size]
			part2 := encoded[part1Size:part3Start]
			part3 := encoded[part3Start:]

			// Step 3: Decode each part separately (streaming fashion)
			decodedBuf := buffer.New(make([]byte, 0, len(tt.data)*2))
			totalDecoded := 0

			// Decode part 1
			part1Reader := buffer.New(part1)
			bytesDecoded1 := decoder.Process(decodedBuf, part1Reader)
			totalDecoded += bytesDecoded1

			// Decode part 2
			part2Reader := buffer.New(part2)
			bytesDecoded2 := decoder.Process(decodedBuf, part2Reader)
			totalDecoded += bytesDecoded2

			// Decode part 3
			part3Reader := buffer.New(part3)
			bytesDecoded3 := decoder.Process(decodedBuf, part3Reader)
			totalDecoded += bytesDecoded3

			// Step 4: Verify decoded result matches original
			decoded := decodedBuf.Items()
			require.Equal(t, len(tt.data), totalDecoded, "Should decode all bytes")
			assert.Equal(t, tt.data, decoded, "Decoded data should match original")
		})
	}
}

// TestHamming_Streaming_MultipleChunks tests decoding with multiple chunks
// Uses chunk sizes that ensure proper alignment (multiples of 2 bytes)
func TestHamming_Streaming_MultipleChunks(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	original := make([]byte, 50)
	for i := range original {
		original[i] = byte(i)
	}

	// Encode
	srcReader := buffer.New(original)
	expectedSize := encoder.Process(nil, srcReader)
	encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
	srcReader = buffer.New(original)
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Split into multiple chunks (varying sizes, but ensuring we can decode)
	// Use chunk sizes that are multiples of 2 to ensure complete pairs
	decodedBuf := buffer.New(make([]byte, 0, len(original)*2))
	totalDecoded := 0

	chunkSizes := []int{4, 6, 8, 10, 12, 14, 16, 18, 12} // Vary but keep even
	chunkIdx := 0
	for i := 0; i < len(encoded); {
		chunkSize := chunkSizes[chunkIdx%len(chunkSizes)]
		end := i + chunkSize
		if end > len(encoded) {
			end = len(encoded)
		}
		chunk := encoded[i:end]
		chunkReader := buffer.New(chunk)
		bytesDecoded := decoder.Process(decodedBuf, chunkReader)
		totalDecoded += bytesDecoded
		i = end
		chunkIdx++
	}

	// Verify all bytes decoded
	decoded := decodedBuf.Items()
	require.Equal(t, len(original), totalDecoded, "Should decode all bytes")
	assert.Equal(t, original, decoded)
}

// TestHamming_Streaming_EncoderMultipleChunks tests encoding with multiple chunks
func TestHamming_Streaming_EncoderMultipleChunks(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	original := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0}

	// Encode in multiple chunks
	encodedBuf := buffer.New(make([]byte, 0, len(original)*3))
	totalEncoded := 0

	// Split original into chunks and encode each
	chunkSize := 2
	for i := 0; i < len(original); i += chunkSize {
		end := i + chunkSize
		if end > len(original) {
			end = len(original)
		}
		chunk := original[i:end]
		chunkReader := buffer.New(chunk)
		bytesEncoded := encoder.Process(encodedBuf, chunkReader)
		totalEncoded += bytesEncoded
	}

	encoded := encodedBuf.Items()
	require.Equal(t, len(original)*2, totalEncoded) // Each byte → 2 bytes

	// Decode the entire encoded stream
	encodedReader := buffer.New(encoded)
	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	bytesDecoded := decoder.Process(decodedBuf, encodedReader)
	decoded := decodedBuf.Items()

	// Verify
	require.Equal(t, len(original), bytesDecoded)
	assert.Equal(t, original, decoded)
}

// TestHamming_Streaming_EncoderDecoderInterleaved tests interleaved encoding and decoding
func TestHamming_Streaming_EncoderDecoderInterleaved(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	original := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC}

	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	totalDecoded := 0

	// Encode and decode first chunk (2 bytes → 4 encoded bytes)
	chunk1 := original[:2]
	encodedBuf1 := buffer.New(make([]byte, 0, 10))
	chunk1Reader := buffer.New(chunk1)
	encoder.Process(encodedBuf1, chunk1Reader)
	encoded1 := encodedBuf1.Items()
	encoded1Reader := buffer.New(encoded1)
	bytesDecoded1 := decoder.Process(decodedBuf, encoded1Reader)
	totalDecoded += bytesDecoded1

	// Encode and decode second chunk (2 bytes → 4 encoded bytes)
	chunk2 := original[2:4]
	encodedBuf2 := buffer.New(make([]byte, 0, 10))
	chunk2Reader := buffer.New(chunk2)
	encoder.Process(encodedBuf2, chunk2Reader)
	encoded2 := encodedBuf2.Items()
	encoded2Reader := buffer.New(encoded2)
	bytesDecoded2 := decoder.Process(decodedBuf, encoded2Reader)
	totalDecoded += bytesDecoded2

	// Encode and decode third chunk (2 bytes → 4 encoded bytes)
	chunk3 := original[4:]
	encodedBuf3 := buffer.New(make([]byte, 0, 10))
	chunk3Reader := buffer.New(chunk3)
	encoder.Process(encodedBuf3, chunk3Reader)
	encoded3 := encodedBuf3.Items()
	encoded3Reader := buffer.New(encoded3)
	bytesDecoded3 := decoder.Process(decodedBuf, encoded3Reader)
	totalDecoded += bytesDecoded3

	// Verify
	decoded := decodedBuf.Items()
	require.Equal(t, len(original), totalDecoded)
	assert.Equal(t, original, decoded)
}

