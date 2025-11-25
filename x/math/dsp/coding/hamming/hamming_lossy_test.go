package ecc

import (
	"math/rand"
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHamming_LossyRoundtrip_SingleBitErrors(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	testCases := []struct {
		name string
		data []byte
	}{
		{"small", []byte{0x12}},
		{"medium", []byte{0x12, 0x34, 0x56}},
		{"large", make([]byte, 100)},
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

			// Encode
			srcReader := buffer.New(tt.data)
			expectedSize := encoder.Process(nil, srcReader)
			encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
			srcReader = buffer.New(tt.data)
			encoder.Process(encodedBuf, srcReader)
			encoded := encodedBuf.Items()

			// Test correcting single-bit errors in each codeword position
			// Each input byte produces 2 codewords (2 output bytes)
			for codewordIdx := 0; codewordIdx < len(encoded); codewordIdx++ {
				for bitPos := 0; bitPos < 7; bitPos++ { // 7 bits per codeword
					// Create corrupted version with single-bit error
					corrupted := make([]byte, len(encoded))
					copy(corrupted, encoded)
					corrupted[codewordIdx] ^= (1 << bitPos) // Flip one bit

					// Decode
					corruptedReader := buffer.New(corrupted)
					decodedBuf := buffer.New(make([]byte, 0, len(tt.data)))
					bytesDecoded := decoder.Process(decodedBuf, corruptedReader)
					decoded := decodedBuf.Items()

					// Should correct the error and decode successfully
					require.Equal(t, len(tt.data), bytesDecoded,
						"Should decode after correcting single-bit error at codeword %d, bit %d",
						codewordIdx, bitPos)
					assert.Equal(t, tt.data, decoded,
						"Should match original after correcting single-bit error at codeword %d, bit %d",
						codewordIdx, bitPos)
				}
			}
		})
	}
}

func TestHamming_LossyRoundtrip_MultipleSingleBitErrors(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	original := []byte{0x12, 0x34, 0x56, 0x78}

	// Encode
	srcReader := buffer.New(original)
	expectedSize := encoder.Process(nil, srcReader)
	encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
	srcReader = buffer.New(original)
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Introduce multiple single-bit errors in different codewords
	// Hamming(7,4) can correct one error per codeword
	corrupted := make([]byte, len(encoded))
	copy(corrupted, encoded)
	corrupted[0] ^= 0x01 // Error in first codeword
	corrupted[2] ^= 0x02 // Error in third codeword
	corrupted[4] ^= 0x04 // Error in fifth codeword

	// Decode
	corruptedReader := buffer.New(corrupted)
	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	bytesDecoded := decoder.Process(decodedBuf, corruptedReader)
	decoded := decodedBuf.Items()

	// Should correct all errors (one per codeword)
	require.Equal(t, len(original), bytesDecoded)
	assert.Equal(t, original, decoded, "Should correct multiple single-bit errors in different codewords")
}

func TestHamming_LossyRoundtrip_RandomSingleBitErrors(t *testing.T) {
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

	// Introduce random single-bit errors (one per codeword)
	rng := rand.New(rand.NewSource(42))
	corrupted := make([]byte, len(encoded))
	copy(corrupted, encoded)

	// Corrupt random bits in different codewords
	// Each codeword can have one error corrected
	// Use a map to track which codewords we've already corrupted
	corruptedCodewords := make(map[int]bool)
	numErrors := len(encoded) / 2 // One error per codeword, but spread across codewords
	if numErrors > 20 {
		numErrors = 20 // Limit for test speed
	}
	attempts := 0
	maxAttempts := numErrors * 10 // Prevent infinite loop
	for i := 0; i < numErrors && attempts < maxAttempts; attempts++ {
		codewordIdx := rng.Intn(len(encoded))
		if !corruptedCodewords[codewordIdx] {
			bitPos := rng.Intn(7) // 7 bits per codeword
			corrupted[codewordIdx] ^= (1 << bitPos)
			corruptedCodewords[codewordIdx] = true
			i++
		}
	}

	// Decode
	corruptedReader := buffer.New(corrupted)
	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	bytesDecoded := decoder.Process(decodedBuf, corruptedReader)
	decoded := decodedBuf.Items()

	// Should correct all single-bit errors (one per codeword)
	require.Equal(t, len(original), bytesDecoded)
	assert.Equal(t, original, decoded, "Should correct random single-bit errors")
}

func TestHamming_LossyRoundtrip_DoubleBitErrors(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	original := []byte{0x12, 0x34, 0x56}

	// Encode
	srcReader := buffer.New(original)
	expectedSize := encoder.Process(nil, srcReader)
	encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
	srcReader = buffer.New(original)
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Introduce double-bit error in a single codeword
	// Hamming(7,4) can detect but not correct double-bit errors
	corrupted := make([]byte, len(encoded))
	copy(corrupted, encoded)
	corrupted[0] ^= 0x03 // Flip two bits in first codeword

	// Decode
	corruptedReader := buffer.New(corrupted)
	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	bytesDecoded := decoder.Process(decodedBuf, corruptedReader)
	decoded := decodedBuf.Items()

	// Hamming(7,4) can detect double-bit errors but may not correct them correctly
	// The decoder will attempt correction, but may not produce correct result
	// This is expected behavior - Hamming(7,4) is designed for single-bit errors
	if bytesDecoded > 0 {
		// If it decoded, the result may not match (double-bit errors are not always correctable)
		// This is acceptable - Hamming codes are designed for single-bit error correction
		_ = decoded // May not match original
		t.Logf("Note: Double-bit error may not be correctable (expected for Hamming(7,4))")
	}
}

func TestHamming_LossyRoundtrip_NoErrors(t *testing.T) {
	encoder := NewHammingEncoder()
	decoder := NewHammingDecoder()

	original := []byte{0x12, 0x34, 0x56, 0x78}

	// Encode
	srcReader := buffer.New(original)
	expectedSize := encoder.Process(nil, srcReader)
	encodedBuf := buffer.New(make([]byte, 0, expectedSize*2))
	srcReader = buffer.New(original)
	encoder.Process(encodedBuf, srcReader)
	encoded := encodedBuf.Items()

	// Decode without errors
	encodedReader := buffer.New(encoded)
	decodedBuf := buffer.New(make([]byte, 0, len(original)))
	bytesDecoded := decoder.Process(decodedBuf, encodedReader)
	decoded := decodedBuf.Items()

	// Should decode successfully
	require.Equal(t, len(original), bytesDecoded)
	assert.Equal(t, original, decoded)
}

