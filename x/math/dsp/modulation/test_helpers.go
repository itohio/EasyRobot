package modulation

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ModulateWithPadding tests a modulator with padding and chunked encoding.
// It generates padding at the beginning, calls Encode with chunks of data,
// concatenates the signal, and adds padding at the end.
//
// Parameters:
//   - t: testing.T for assertions
//   - modulator: the modulator to test (must implement Modulator interface)
//   - data: the input data to encode
//   - paddingSamples: number of zero samples to add at beginning and end
//   - chunkSize: size of data chunks to encode (0 means encode all at once)
//
// Returns the complete signal with padding (beginning + encoded signal + end).
func ModulateWithPadding(
	t *testing.T,
	modulator Modulator,
	data []byte,
	paddingSamples int,
	chunkSize int,
) vecTypes.Vector {
	require.NotNil(t, modulator, "Modulator must not be nil")
	require.NotNil(t, data, "Data must not be nil")
	require.GreaterOrEqual(t, paddingSamples, 0, "Padding samples must be non-negative")
	if chunkSize == 0 {
		chunkSize = len(data) // Encode all at once if chunkSize is 0
	}
	require.Greater(t, chunkSize, 0, "Chunk size must be positive")

	// Calculate total signal size (without padding)
	expectedSize := modulator.Encode(nil, data)
	require.Greater(t, expectedSize, 0, "Expected signal size must be positive")

	// Create output buffer for complete signal
	totalSize := paddingSamples*2 + expectedSize
	signal := vec.New(totalSize)
	signalData := []float32(signal)

	// Encode data in chunks
	var encodedChunks []vecTypes.Vector
	if chunkSize > 0 && chunkSize < len(data) {
		// Encode in chunks
		offset := 0
		for offset < len(data) {
			end := offset + chunkSize
			if end > len(data) {
				end = len(data)
			}
			chunk := data[offset:end]

			// Calculate size for this chunk
			chunkSizeExpected := modulator.Encode(nil, chunk)
			if chunkSizeExpected > 0 {
				chunkSignal := vec.New(chunkSizeExpected)
				samplesWritten := modulator.Encode(chunkSignal, chunk)
				require.Greater(t, samplesWritten, 0, "Should write samples for chunk")
				encodedChunks = append(encodedChunks, chunkSignal)
			}
			offset = end
		}
	} else {
		// Encode all at once
		chunkSignal := vec.New(expectedSize)
		samplesWritten := modulator.Encode(chunkSignal, data)
		require.Greater(t, samplesWritten, 0, "Should write samples")
		require.Equal(t, expectedSize, samplesWritten, "Should write all expected samples")
		encodedChunks = append(encodedChunks, chunkSignal)
	}

	// Concatenate all encoded chunks
	signalOffset := paddingSamples
	for _, chunk := range encodedChunks {
		chunkData := []float32(chunk.(vec.Vector))
		copy(signalData[signalOffset:], chunkData)
		signalOffset += len(chunkData)
	}

	// Verify we used the expected amount of space
	assert.Equal(t, paddingSamples+expectedSize, signalOffset, "Should have used correct amount of space for signal")

	return signal
}

// DemodulateWithChunks tests a demodulator with chunked decoding.
// It calls Decode with chunks of the signal and returns concatenated bytes.
//
// Parameters:
//   - t: testing.T for assertions
//   - demodulator: the demodulator to test (must implement Demodulator interface)
//   - signal: the complete signal to decode (including any padding)
//   - chunkSize: size of signal chunks to decode
//   - expectedDataLen: expected length of decoded data (0 means estimate from signal)
//
// Returns the decoded bytes.
func DemodulateWithChunks(
	t *testing.T,
	demodulator Demodulator,
	signal vecTypes.Vector,
	chunkSize int,
	expectedDataLen int,
) []byte {
	require.NotNil(t, demodulator, "Demodulator must not be nil")
	require.NotNil(t, signal, "Signal must not be nil")
	require.Greater(t, chunkSize, 0, "Chunk size must be positive")

	signalData := []float32(signal.(vec.Vector))
	require.Greater(t, len(signalData), 0, "Signal must not be empty")

	// Estimate output size if not provided
	if expectedDataLen == 0 {
		expectedDataLen = demodulator.Decode(nil, signal)
		if expectedDataLen <= 0 {
			// Fallback: use a reasonable default based on signal length
			expectedDataLen = len(signalData) / 100 // Rough estimate
		}
	}

	// Allocate output buffer (with some headroom)
	decoded := make([]byte, expectedDataLen*2)
	bytesDecoded := 0

	// Decode in chunks
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}

		// Create chunk vector
		chunkSizeActual := end - i
		chunk := vec.New(chunkSizeActual)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])

		// Decode chunk
		remaining := decoded[bytesDecoded:]
		chunkDecoded := demodulator.Decode(remaining, chunk)
		bytesDecoded += chunkDecoded

		// Stop if we've decoded enough or run out of space
		if bytesDecoded >= len(decoded) {
			break
		}
	}

	// Trim to actual decoded length
	return decoded[:bytesDecoded]
}

// VerifyExactMatch verifies that the original data appears exactly in the decoded output.
// It searches for the original data in the decoded bytes, allowing for extra bytes from padding
// before or after the signal data. Returns true if the original data is found exactly.
//
// Parameters:
//   - t: testing.T for assertions
//   - original: the original data that was encoded
//   - decoded: the decoded bytes from demodulation
//
// This function will fail the test if the original data is not found exactly in the decoded output.
func VerifyExactMatch(t *testing.T, original, decoded []byte) {
	require.NotNil(t, original, "Original data must not be nil")
	require.NotNil(t, decoded, "Decoded data must not be nil")

	if len(original) == 0 {
		// If original is empty, decoded should also be empty (or we don't care)
		return
	}

	if len(decoded) == 0 {
		assert.Fail(t, "No bytes decoded, but original data was non-empty", "original length: %d", len(original))
		return
	}

	// Try to find original data exactly in decoded output
	found := false
	for i := 0; i <= len(decoded)-len(original); i++ {
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

	if !found {
		// Also check if first len(original) bytes match (for cases where demodulator outputs data first)
		if len(decoded) >= len(original) {
			prefixMatch := true
			for i := 0; i < len(original); i++ {
				if decoded[i] != original[i] {
					prefixMatch = false
					break
				}
			}
			if prefixMatch {
				found = true
			}
		}
	}

	if !found {
		// Show helpful error message
		assert.Fail(t,
			"Original data not found exactly in decoded output",
			"original: %v\n decoded: %v (first %d bytes: %v)",
			original,
			decoded,
			len(decoded),
			decoded,
		)
	}
}
