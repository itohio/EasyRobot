package dsss

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	// Test parameters
	testCarrierFreq = float32(1000)  // 1 kHz carrier frequency
	testSymbolRate  = float32(1000)  // 1 kbaud symbol rate
	testAmplitude   = float32(1.0)
)

var (
	// Test spreading factor: m-sequence with n=3 -> length = 2^3 - 1 = 7 chips
	testSpreadingFactor = 7
	
	// Test sample rate: calculated to exceed recommended rate for reliable testing
	// Recommended rate calculation:
	//   chipRate = symbolRate * spreadingFactor = 1000 * 7 = 7000 chips/sec
	//   recommendedRate = max(4*carrierFreq, chipRate*8) = max(4000, 56000) = 56000 Hz
	// Using 1.15x recommended rate (~64000 Hz) for better test reliability
	testSampleRate = CalculateRecommendedSampleRate(testSymbolRate, float32(testSpreadingFactor), testCarrierFreq) * 1.15
)

func init() {
	// Verify test sample rate exceeds recommended rate
	recommendedRate := CalculateRecommendedSampleRate(testSymbolRate, float32(testSpreadingFactor), testCarrierFreq)
	if testSampleRate < recommendedRate {
		panic("testSampleRate must exceed recommended rate for reliable tests")
	}
}

func TestModulator_NewModulator(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)

	require.NotNil(t, modulator)
	assert.Equal(t, testSampleRate, modulator.sampleRate)
	assert.Equal(t, testCarrierFreq, modulator.carrierFreq)
	assert.Equal(t, testSymbolRate, modulator.symbolRate)
	assert.Equal(t, code.Length(), modulator.spreadingFactor)
	assert.Equal(t, 2, modulator.modulationOrder)
}

func TestModulator_NewMModulator(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, codeTable...)

	require.NotNil(t, modulator)
	assert.Equal(t, 4, modulator.modulationOrder)
	assert.Equal(t, codeTable[0].Length(), modulator.spreadingFactor)
}

func TestModulator_Encode_NilAmplitudes(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)
	data := []byte{0x55, 0xAA}

	expectedSize := modulator.Encode(nil, data)
	assert.Greater(t, expectedSize, 0, "Expected size should be positive")
	// Encoder doesn't add padding - size is just for the signal
}

func TestModulator_Encode_BinaryDSSS(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)
	data := []byte{0x55} // 01010101

	// Calculate required size (no padding - encoder doesn't add padding)
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")
	assert.Equal(t, expectedSize, samplesWritten, "Should write all expected samples")

	ampData := []float32(amplitudes)

	// Check that signal is present (non-zero values)
	hasNonZero := false
	for i := 0; i < len(ampData); i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Signal should be present")
}

func TestModulator_Encode_MaryDSSS(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, codeTable...)
	data := []byte{0x12} // 00010010

	// Calculate required size (no padding - encoder doesn't add padding)
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")

	ampData := []float32(amplitudes)
	// Check that we have signal (non-zero values)
	hasNonZero := false
	for i := 0; i < len(ampData); i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "M-ary DSSS should have non-zero amplitudes")
}

func TestModulator_Reset(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)

	// Reset should not panic
	modulator.Reset()

	// Verify we can still encode after reset
	data := []byte{0x55}
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)
	assert.Greater(t, samplesWritten, 0, "Should still encode after reset")
}

func TestModulator_SetAmplitude(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)
	newAmplitude := float32(0.5)

	modulator.SetAmplitude(newAmplitude)
	assert.Equal(t, newAmplitude, modulator.amplitude)
}

func TestModulator_ModulationOrder(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator2 := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)
	codeTable := GenerateWalshCodeTable(4)
	modulator4 := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, codeTable...)

	assert.Equal(t, 2, modulator2.ModulationOrder())
	assert.Equal(t, 4, modulator4.ModulationOrder())
}

func TestModulator_SpreadingFactor(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)

	assert.Equal(t, code.Length(), modulator.SpreadingFactor())
}

func TestModulator_InterfaceCompliance(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, code)

	// Verify it implements the Modulator interface
	var _ interface {
		Encode(amplitudes vecTypes.Vector, data []byte) int
		Reset()
	} = modulator
}

func TestModulator_TestSampleRate_ExceedsRecommended(t *testing.T) {
	// Verify that test sample rate exceeds recommended rate
	recommendedRate := CalculateRecommendedSampleRate(testSymbolRate, float32(testSpreadingFactor), testCarrierFreq)
	
	assert.GreaterOrEqual(t, testSampleRate, recommendedRate,
		"Test sample rate must be >= recommended rate for reliable tests")
	
	// Verify it's at least 1.1x recommended (should be 1.15x based on our calculation)
	assert.GreaterOrEqual(t, testSampleRate, recommendedRate*1.1,
		"Test sample rate should be at least 1.1x recommended rate")
	
	t.Logf("Test sample rate: %.0f Hz", testSampleRate)
	t.Logf("Recommended rate: %.0f Hz", recommendedRate)
	t.Logf("Minimum rate: %.0f Hz", CalculateMinimumSampleRate(testSymbolRate, float32(testSpreadingFactor), testCarrierFreq, MinimumSamplesPerChip))
}
