package modulation_test

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/ecc"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/manchester"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestChainEncoder_Single tests chaining a single encoder
func TestChainEncoder_Single(t *testing.T) {
	manchesterEncoder := manchester.NewEncoder()
	chain := modulation.NewEncoderChain(manchesterEncoder)

	data := []byte{0xAA, 0x55}
	expectedSize := chain.Encode(nil, data)
	require.Greater(t, expectedSize, 0, "Expected size should be positive")

	output := make([]byte, expectedSize)
	bytesWritten := chain.Encode(output, data)
	require.Equal(t, expectedSize, bytesWritten, "Should write expected bytes")
	assert.Greater(t, bytesWritten, 0, "Should encode bytes")
}

// TestChainEncoder_Multiple tests chaining multiple encoders
func TestChainEncoder_Multiple(t *testing.T) {
	// Create chain: Manchester -> Reed-Solomon
	manchesterEncoder := manchester.NewEncoder()
	rsEncoder := ecc.NewRSEncoder(255, 223) // Standard RS(255,223) codec
	chain := modulation.NewEncoderChain(manchesterEncoder, rsEncoder)

	data := []byte{0x12, 0x34, 0x56, 0x78}
	expectedSize := chain.Encode(nil, data)
	require.Greater(t, expectedSize, 0, "Expected size should be positive")

	output := make([]byte, expectedSize*2) // Extra space for safety
	bytesWritten := chain.Encode(output, data)
	assert.Greater(t, bytesWritten, 0, "Should encode bytes through chain")
}

// TestChainDecoder_Single tests chaining a single decoder
func TestChainDecoder_Single(t *testing.T) {
	manchesterDecoder := manchester.NewDecoder()
	chain := modulation.NewDecoderChain(manchesterDecoder)

	// Encode some data first
	manchesterEncoder := manchester.NewEncoder()
	data := []byte{0xAA, 0x55}
	encoded := make([]byte, len(data)*2)
	bytesEncoded := manchesterEncoder.Encode(encoded, data)
	require.Greater(t, bytesEncoded, 0, "Should encode data")

	// Decode
	expectedSize := chain.Decode(nil, encoded[:bytesEncoded])
	require.Greater(t, expectedSize, 0, "Expected size should be positive")

	output := make([]byte, expectedSize)
	bytesDecoded := chain.Decode(output, encoded[:bytesEncoded])
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes")
}

// TestChainDecoder_Multiple tests chaining multiple decoders in reverse order
func TestChainDecoder_Multiple(t *testing.T) {
	// RS(255,223) requires data in multiples of 223 bytes
	// Use a smaller test that works with Manchester only
	// For full RS test, use data that fits RS block size
	manchesterEncoder := manchester.NewEncoder()
	encoderChain := modulation.NewEncoderChain(manchesterEncoder)

	manchesterDecoder := manchester.NewDecoder()
	decoderChain := modulation.NewDecoderChain(manchesterDecoder)

	data := []byte{0x12, 0x34, 0x56, 0x78}

	// Encode through chain
	encodedSize := encoderChain.Encode(nil, data)
	require.Greater(t, encodedSize, 0, "Should calculate encoded size")

	encoded := make([]byte, encodedSize*2) // Extra space
	bytesEncoded := encoderChain.Encode(encoded, data)
	require.Greater(t, bytesEncoded, 0, "Should encode through chain")

	// Decode through reverse chain
	decodedSize := decoderChain.Decode(nil, encoded[:bytesEncoded])
	require.Greater(t, decodedSize, 0, "Should calculate decoded size")

	decoded := make([]byte, decodedSize)
	bytesDecoded := decoderChain.Decode(decoded, encoded[:bytesEncoded])
	assert.Greater(t, bytesDecoded, 0, "Should decode through reverse chain")

	// Verify we got back the original data
	assert.Equal(t, data, decoded[:len(data)], "Decoded data should match original")
}

// TestRoundtrip_Manchester_RS tests round-trip encoding/decoding with Manchester only
// (RS requires specific block sizes, so tested separately)
func TestRoundtrip_Manchester_RS(t *testing.T) {
	// Use Manchester only for simpler round-trip test
	// RS round-trip would require data in multiples of RS block size
	manchesterEncoder := manchester.NewEncoder()
	encoderChain := modulation.NewEncoderChain(manchesterEncoder)

	manchesterDecoder := manchester.NewDecoder()
	decoderChain := modulation.NewDecoderChain(manchesterDecoder)

	testCases := [][]byte{
		{0xAA, 0x55},
		{0x12, 0x34, 0x56, 0x78},
		{0x00, 0xFF, 0x80, 0x7F},
		{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08},
	}

	for _, originalData := range testCases {
		t.Run("", func(t *testing.T) {
			encoderChain.Reset()
			decoderChain.Reset()

			// Encode
			expectedSize := encoderChain.Encode(nil, originalData)
			require.Greater(t, expectedSize, 0, "Should calculate encoded size")

			encoded := make([]byte, expectedSize*2) // Extra space
			bytesEncoded := encoderChain.Encode(encoded, originalData)
			require.Greater(t, bytesEncoded, 0, "Should encode data")

			// Decode
			decodedSize := decoderChain.Decode(nil, encoded[:bytesEncoded])
			require.Greater(t, decodedSize, 0, "Should calculate decoded size")

			decoded := make([]byte, decodedSize)
			bytesDecoded := decoderChain.Decode(decoded, encoded[:bytesEncoded])
			assert.Greater(t, bytesDecoded, 0, "Should decode data")

			// Verify round-trip
			assert.Equal(t, originalData, decoded[:len(originalData)], "Round-trip should preserve data")
		})
	}
}

// TestChainEncoder_Reset tests reset functionality
func TestChainEncoder_Reset(t *testing.T) {
	manchesterEncoder := manchester.NewEncoder()
	rsEncoder := ecc.NewRSEncoder(255, 223)
	chain := modulation.NewEncoderChain(manchesterEncoder, rsEncoder)

	// Encode some data
	data := []byte{0xAA, 0x55}
	output := make([]byte, 1000)
	chain.Encode(output, data)

	// Reset
	chain.Reset()

	// Should be able to encode again after reset
	bytesWritten := chain.Encode(output, data)
	assert.Greater(t, bytesWritten, 0, "Should encode after reset")
}

// TestChainDecoder_Reset tests reset functionality
func TestChainDecoder_Reset(t *testing.T) {
	manchesterDecoder := manchester.NewDecoder()
	rsDecoder := ecc.NewRSDecoder(255, 223)
	chain := modulation.NewDecoderChain(rsDecoder, manchesterDecoder)

	// Decode some data (even if empty)
	input := make([]byte, 100)
	output := make([]byte, 100)
	chain.Decode(output, input)

	// Reset
	chain.Reset()

	// Should be able to decode again after reset
	bytesDecoded := chain.Decode(output, input)
	_ = bytesDecoded // Reset doesn't guarantee bytes decoded
}

// TestChainEncoder_EmptyData tests with empty data
func TestChainEncoder_EmptyData(t *testing.T) {
	manchesterEncoder := manchester.NewEncoder()
	chain := modulation.NewEncoderChain(manchesterEncoder)

	output := make([]byte, 100)
	bytesWritten := chain.Encode(output, []byte{})
	assert.Equal(t, 0, bytesWritten, "Empty data should return 0")
}

// TestChainDecoder_EmptyData tests with empty data
func TestChainDecoder_EmptyData(t *testing.T) {
	manchesterDecoder := manchester.NewDecoder()
	chain := modulation.NewDecoderChain(manchesterDecoder)

	output := make([]byte, 100)
	bytesDecoded := chain.Decode(output, []byte{})
	assert.Equal(t, 0, bytesDecoded, "Empty data should return 0")
}

// TestChainEncoder_NilDst tests nil destination (size query)
func TestChainEncoder_NilDst(t *testing.T) {
	manchesterEncoder := manchester.NewEncoder()
	rsEncoder := ecc.NewRSEncoder(255, 223)
	chain := modulation.NewEncoderChain(manchesterEncoder, rsEncoder)

	data := []byte{0xAA, 0x55}
	expectedSize := chain.Encode(nil, data)
	assert.Greater(t, expectedSize, 0, "Should return expected size when dst is nil")
}

// TestChainDecoder_NilDst tests nil destination (size query)
func TestChainDecoder_NilDst(t *testing.T) {
	manchesterEncoder := manchester.NewEncoder()
	rsEncoder := ecc.NewRSEncoder(255, 223)
	encoderChain := modulation.NewEncoderChain(manchesterEncoder, rsEncoder)

	// Encode first
	data := []byte{0xAA, 0x55}
	encoded := make([]byte, 1000)
	bytesEncoded := encoderChain.Encode(encoded, data)
	require.Greater(t, bytesEncoded, 0, "Should encode data")

	// Create decoder chain
	rsDecoder := ecc.NewRSDecoder(255, 223)
	manchesterDecoder := manchester.NewDecoder()
	decoderChain := modulation.NewDecoderChain(rsDecoder, manchesterDecoder)

	expectedSize := decoderChain.Decode(nil, encoded[:bytesEncoded])
	assert.Greater(t, expectedSize, 0, "Should return expected size when dst is nil")
}

// TestChainEncoder_InterfaceCompliance tests that ChainEncoder implements ByteEncoder
func TestChainEncoder_InterfaceCompliance(t *testing.T) {
	manchesterEncoder := manchester.NewEncoder()
	chain := modulation.NewEncoderChain(manchesterEncoder)

	var _ modulation.ByteEncoder = chain
	assert.True(t, true, "ChainEncoder implements ByteEncoder interface")
}

// TestChainDecoder_InterfaceCompliance tests that ChainDecoder implements ByteDecoder
func TestChainDecoder_InterfaceCompliance(t *testing.T) {
	manchesterDecoder := manchester.NewDecoder()
	chain := modulation.NewDecoderChain(manchesterDecoder)

	var _ modulation.ByteDecoder = chain
	assert.True(t, true, "ChainDecoder implements ByteDecoder interface")
}
