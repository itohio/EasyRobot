package modulation

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

// TestGoertzelFrequencyDetector_PureSine tests Goertzel with pure sine waves
// This verifies the algorithm itself is correct, independent of encoder
func TestGoertzelFrequencyDetector_PureSine(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	samplesPerSymbol := 48

	detector := NewGoertzelFrequencyDetector(sampleRate, freqLow, freqHigh)

	// Test 1: Pure sine wave at freqLow
	signalLow := vec.New(samplesPerSymbol)
	signalDataLow := []float32(signalLow)
	amplitude := float32(0.5)
	for i := 0; i < samplesPerSymbol; i++ {
		phase := 2 * math.Pi * freqLow * float32(i) / sampleRate
		signalDataLow[i] = amplitude * float32(math.Cos(float64(phase)))
	}

	freqIdx, magnitude := detector.DetectFrequency(signalDataLow)
	assert.Equal(t, 0, freqIdx, "Should detect freqLow (index 0) for pure sine")
	assert.Greater(t, magnitude, float32(0.1), "Magnitude should be significant")

	// Test 2: Pure sine wave at freqHigh
	signalHigh := vec.New(samplesPerSymbol)
	signalDataHigh := []float32(signalHigh)
	for i := 0; i < samplesPerSymbol; i++ {
		phase := 2 * math.Pi * freqHigh * float32(i) / sampleRate
		signalDataHigh[i] = amplitude * float32(math.Cos(float64(phase)))
	}

	freqIdx, magnitude = detector.DetectFrequency(signalDataHigh)
	assert.Equal(t, 1, freqIdx, "Should detect freqHigh (index 1) for pure sine")
	assert.Greater(t, magnitude, float32(0.1), "Magnitude should be significant")

	// Test 3: Magnitudes should be similar for same amplitude
	_, magLow := detector.DetectFrequency(signalDataLow)
	_, magHigh := detector.DetectFrequency(signalDataHigh)
	ratio := magLow / magHigh
	assert.InDelta(t, 1.0, ratio, 0.2, "Magnitudes should be similar for same amplitude (ratio should be close to 1.0)")
}

// TestGoertzelFrequencyDetector tests frequency detection with Goertzel
func TestGoertzelFrequencyDetector(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	samplesPerSymbol := 48

	detector := NewGoertzelFrequencyDetector(sampleRate, freqLow, freqHigh)

	// Test 1: Generate pure sine wave at freqLow and detect it
	signalLow := vec.New(samplesPerSymbol)
	signalDataLow := []float32(signalLow)
	amplitude := float32(0.5)
	phase := float32(0)
	for i := 0; i < samplesPerSymbol; i++ {
		phase += 2 * math.Pi * freqLow / sampleRate
		signalDataLow[i] = amplitude * float32(math.Cos(float64(phase)))
	}

	freqIdx, magnitude := detector.DetectFrequency(signalDataLow)
	assert.Equal(t, 0, freqIdx, "Should detect freqLow (index 0) for bit 0")
	assert.Greater(t, magnitude, float32(0.1), "Magnitude should be significant")

	// Test 2: Generate pure sine wave at freqHigh and detect it
	signalHigh := vec.New(samplesPerSymbol)
	signalDataHigh := []float32(signalHigh)
	phase = float32(0)
	for i := 0; i < samplesPerSymbol; i++ {
		phase += 2 * math.Pi * freqHigh / sampleRate
		signalDataHigh[i] = amplitude * float32(math.Cos(float64(phase)))
	}

	freqIdx, magnitude = detector.DetectFrequency(signalDataHigh)
	assert.Equal(t, 1, freqIdx, "Should detect freqHigh (index 1) for bit 1")
	assert.Greater(t, magnitude, float32(0.1), "Magnitude should be significant")
}

// TestCorrelationFrequencyDetector tests frequency detection with correlation
func TestCorrelationFrequencyDetector(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	samplesPerSymbol := 48

	detector := NewCorrelationFrequencyDetector(sampleRate, freqLow, freqHigh)

	// Test 1: Generate pure sine wave at freqLow and detect it
	signalLow := vec.New(samplesPerSymbol)
	signalDataLow := []float32(signalLow)
	amplitude := float32(0.5)
	phase := float32(0)
	for i := 0; i < samplesPerSymbol; i++ {
		phase += 2 * math.Pi * freqLow / sampleRate
		signalDataLow[i] = amplitude * float32(math.Cos(float64(phase)))
	}

	freqIdx, magnitude := detector.DetectFrequency(signalDataLow)
	assert.Equal(t, 0, freqIdx, "Should detect freqLow (index 0)")
	assert.Greater(t, magnitude, float32(0.1), "Magnitude should be significant")

	// Test 2: Generate pure sine wave at freqHigh and detect it
	signalHigh := vec.New(samplesPerSymbol)
	signalDataHigh := []float32(signalHigh)
	phase = float32(0)
	for i := 0; i < samplesPerSymbol; i++ {
		phase += 2 * math.Pi * freqHigh / sampleRate
		signalDataHigh[i] = amplitude * float32(math.Cos(float64(phase)))
	}

	freqIdx, magnitude = detector.DetectFrequency(signalDataHigh)
	assert.Equal(t, 1, freqIdx, "Should detect freqHigh (index 1)")
	assert.Greater(t, magnitude, float32(0.1), "Magnitude should be significant")
}

// TestFrequencyDetector_PhaseIndependence tests that detection works regardless of phase
func TestFrequencyDetector_PhaseIndependence(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	samplesPerSymbol := 48

	detector := NewGoertzelFrequencyDetector(sampleRate, freqLow, freqHigh)

	// Generate signals with different starting phases
	testCases := []struct {
		name      string
		frequency  float32
		phase     float32
		expected  int
	}{
		{"freqLow phase 0", freqLow, 0, 0},
		{"freqLow phase Pi/4", freqLow, math.Pi / 4, 0},
		{"freqLow phase Pi/2", freqLow, math.Pi / 2, 0},
		{"freqLow phase Pi", freqLow, math.Pi, 0},
		{"freqHigh phase 0", freqHigh, 0, 1},
		{"freqHigh phase Pi/4", freqHigh, math.Pi / 4, 1},
		{"freqHigh phase Pi/2", freqHigh, math.Pi / 2, 1},
		{"freqHigh phase Pi", freqHigh, math.Pi, 1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			signal := vec.New(samplesPerSymbol)
			signalData := []float32(signal)
			amplitude := float32(0.5)
			phase := tc.phase
			for i := 0; i < samplesPerSymbol; i++ {
				phase += 2 * math.Pi * tc.frequency / sampleRate
				signalData[i] = amplitude * float32(math.Cos(float64(phase)))
			}

			freqIdx, magnitude := detector.DetectFrequency(signalData)

			assert.Equal(t, tc.expected, freqIdx,
				"Should detect correct frequency regardless of starting phase")
			assert.Greater(t, magnitude, float32(0.05),
				"Magnitude should be significant")
		})
	}
}
