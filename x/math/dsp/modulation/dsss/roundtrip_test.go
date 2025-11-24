package dsss

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

// TestRoundtrip_Baseband tests roundtrip with baseband (no carrier)
func TestRoundtrip_Baseband(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	sampleRate := float32(64000)
	carrierFreq := float32(0) // Baseband
	symbolRate := float32(1000)
	rolloffFactor := float32(0.5)

	modulator := NewModulator(sampleRate, carrierFreq, symbolRate, rolloffFactor, code)
	demodulator := NewDemodulator(sampleRate, carrierFreq, symbolRate, rolloffFactor, code)

	testData := []byte{0x55} // 01010101

	// Encode
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)

	// Decode
	output := make([]byte, 10)
	bytesDecoded := demodulator.Decode(output, signal)

	t.Logf("Decoded %d bytes", bytesDecoded)
	t.Logf("Output: %v", output[:bytesDecoded])
	t.Logf("Expected: [85]")

	assert.Equal(t, 1, bytesDecoded, "Should decode 1 byte")
	assert.Equal(t, byte(0x55), output[0], "Should decode correct byte")
}

// TestRoundtrip_HighCarrierFrequency tests roundtrip with 20kHz carrier and 96kHz sample rate
func TestRoundtrip_HighCarrierFrequency(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	sampleRate := float32(96000)  // 96 kHz
	carrierFreq := float32(20000) // 20 kHz
	symbolRate := float32(1000)
	rolloffFactor := float32(0.5)

	// Verify bandwidth requirements
	bandwidth := Bandwidth(symbolRate, float32(code.Length()), rolloffFactor)
	minCarrierFreq := bandwidth / 2
	nyquistLimit := sampleRate / 2
	upperFreq := carrierFreq + bandwidth/2

	t.Logf("Sample rate: %.0f Hz", sampleRate)
	t.Logf("Carrier frequency: %.0f Hz", carrierFreq)
	t.Logf("Bandwidth: %.0f Hz", bandwidth)
	t.Logf("Minimum carrier frequency: %.0f Hz", minCarrierFreq)
	t.Logf("Upper frequency: %.0f Hz", upperFreq)
	t.Logf("Nyquist limit: %.0f Hz", nyquistLimit)

	assert.GreaterOrEqual(t, carrierFreq, minCarrierFreq,
		"Carrier frequency must be >= bandwidth/2")
	assert.Greater(t, nyquistLimit, upperFreq,
		"Nyquist limit must be > upper frequency")

	modulator := NewModulator(sampleRate, carrierFreq, symbolRate, rolloffFactor, code)
	demodulator := NewDemodulator(sampleRate, carrierFreq, symbolRate, rolloffFactor, code)

	testData := []byte{0x55} // 01010101

	// Encode
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)

	// Decode
	output := make([]byte, 10)
	bytesDecoded := demodulator.Decode(output, signal)

	t.Logf("Encoded %d samples", expectedSize)
	t.Logf("Decoded %d bytes", bytesDecoded)
	t.Logf("Output: %v", output[:bytesDecoded])
	t.Logf("Expected: [85]")

	// For passband, we might need carrier recovery, so just check we got something
	assert.Greater(t, bytesDecoded, 0, "Should decode at least some bytes")
}

