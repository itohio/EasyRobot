package iq

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
)

func TestRoundtrip_Baseband_Direct(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 0 // Baseband
	cfg.UseInterpolation = false
	require.NoError(t, cfg.Validate())

	testRoundtrip(t, cfg)
}

func TestRoundtrip_Baseband_WithLUT(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 0 // Baseband
	cfg.UseInterpolation = true
	cfg.LUTSize = 4096
	require.NoError(t, cfg.Validate())

	testRoundtrip(t, cfg)
}

func TestRoundtrip_WithCarrier_Direct(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 10_000 // 10 kHz carrier
	cfg.UseInterpolation = false
	require.NoError(t, cfg.Validate())

	testRoundtrip(t, cfg)
}

func TestRoundtrip_WithCarrier_WithLUT(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 10_000 // 10 kHz carrier
	cfg.UseInterpolation = true
	cfg.LUTSize = 4096
	require.NoError(t, cfg.Validate())

	testRoundtrip(t, cfg)
}

func TestRoundtrip_VariousSizes(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 0 // Baseband for simplicity
	require.NoError(t, cfg.Validate())

	sizes := []int{1, 2, 10, 100, 1000}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			testRoundtripSize(t, cfg, size)
		})
	}
}

func testRoundtrip(t *testing.T, cfg Config) {
	testRoundtripSize(t, cfg, 100)
}

func testRoundtripSize(t *testing.T, cfg Config, numSamples int) {
	// Create input: real signal samples
	inputReal := make([]float32, numSamples)
	for i := range numSamples {
		// Generate a simple sine wave for testing
		angle := float32(i) * 0.1
		inputReal[i] = math32.Sin(angle)
	}

	// Create modulator and demodulator
	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// Step 1: Convert real signal -> IQ
	srcBuf := buffer.New(inputReal)
	
	iqBuf := buffer.New(make([]complex64, 0, numSamples))
	writtenIQ := mod.Process(iqBuf, srcBuf)
	require.Equal(t, numSamples, writtenIQ)

	// Step 2: Convert IQ -> real signal
	iqReader := buffer.New(iqBuf.Items())
	
	outputReal := buffer.New(make([]float32, 0, numSamples))
	writtenReal := demod.Process(outputReal, iqReader)
	require.Equal(t, numSamples, writtenReal)

	// Step 3: Verify roundtrip accuracy
	output := outputReal.Items()
	require.Equal(t, len(inputReal), len(output))

	// For baseband, expect exact match (within float32 precision)
	// For carrier mixing, expect close match (mixing introduces larger errors)
	tolerance := float64(1e-5)
	if cfg.CarrierFrequency > 0 {
		// Carrier mixing introduces more errors due to phase accumulation and mixing process
		// The signal goes through modulation and demodulation, so errors accumulate
		tolerance = float64(1e-2) // Higher tolerance for carrier mixing roundtrip
	}
	if cfg.UseInterpolation && cfg.LUTSize <= 256 {
		tolerance = float64(1e-3) // Small LUT needs more tolerance
		if cfg.CarrierFrequency > 0 {
			tolerance = float64(1e-2) // Even higher with small LUT and carrier
		}
	}

	for i := range inputReal {
		require.InDelta(t, float64(inputReal[i]), float64(output[i]), tolerance,
			"mismatch at index %d: input=%.6f, output=%.6f", i, inputReal[i], output[i])
	}
}

func TestModulator_EstimateMatchesActual(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	inputReal := make([]float32, 100)
	for i := range inputReal {
		inputReal[i] = float32(i) * 0.1
	}

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	srcBuf := buffer.New(inputReal)

	estimate := mod.Process(nil, srcBuf)
	require.Greater(t, estimate, 0)

	srcBuf2 := buffer.New(inputReal)
	iqBuf := buffer.New(make([]complex64, 0, estimate))
	written := mod.Process(iqBuf, srcBuf2)
	require.Equal(t, estimate, written)
}

func TestDemodulator_EstimateMatchesActual(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	iqSamples := make([]complex64, 50)
	for i := range iqSamples {
		iqSamples[i] = complex64(complex(float32(i)*0.1, float32(i)*0.1))
	}

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	srcBuf := buffer.New(iqSamples)

	estimate := demod.Process(nil, srcBuf)
	require.Greater(t, estimate, 0)

	srcBuf2 := buffer.New(iqSamples)
	realBuf := buffer.New(make([]float32, 0, estimate))
	written := demod.Process(realBuf, srcBuf2)
	require.Equal(t, estimate, written)
}

func TestModulator_EmptyInput(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	srcBuf := buffer.New([]float32{})

	iqBuf := buffer.New(make([]complex64, 10))
	written := mod.Process(iqBuf, srcBuf)
	require.Equal(t, 0, written)
}

func TestDemodulator_EmptyInput(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	srcBuf := buffer.New([]complex64{})

	realBuf := buffer.New(make([]float32, 10))
	written := demod.Process(realBuf, srcBuf)
	require.Equal(t, 0, written)
}

func TestModulator_Reset(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	inputReal := []float32{1.0, 2.0, 3.0}
	srcBuf := buffer.New(inputReal)

	iqBuf := buffer.New(make([]complex64, 0, 10))
	mod.Process(iqBuf, srcBuf)

	// Reset should clear state
	mod.Reset()
	require.Equal(t, 0, mod.bufferedCount)
	require.Equal(t, float32(0), mod.phaseAccumulator)
}

func TestDemodulator_Reset(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	iqSamples := []complex64{1 + 2i, 3 + 4i}
	srcBuf := buffer.New(iqSamples)

	realBuf := buffer.New(make([]float32, 0, 10))
	demod.Process(realBuf, srcBuf)

	// Reset should clear state
	demod.Reset()
	require.Equal(t, 0, demod.bufferedCount)
	require.Equal(t, float32(0), demod.phaseAccumulator)
}

func TestModulator_Baseband_IQ_Output(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 0 // Baseband
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	// Simple test: real signal at baseband should produce I = signal, Q = 0
	inputReal := []float32{1.0, 2.0, 3.0}
	srcBuf := buffer.New(inputReal)

	iqBuf := buffer.New(make([]complex64, 0, 10))
	written := mod.Process(iqBuf, srcBuf)
	require.Equal(t, len(inputReal), written)

	output := iqBuf.Items()
	require.Equal(t, len(inputReal), len(output))

	// Verify Q component is 0 for baseband
	for i := range output {
		require.Equal(t, inputReal[i], float32(real(output[i])), "I component should equal input")
		require.Equal(t, float32(0), float32(imag(output[i])), "Q component should be 0 at baseband")
	}
}

func TestDemodulator_Baseband_Real_Output(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierFrequency = 0 // Baseband
	require.NoError(t, cfg.Validate())

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// Simple test: IQ at baseband should produce real = I
	iqSamples := []complex64{1 + 2i, 3 + 4i, 5 + 6i}
	srcBuf := buffer.New(iqSamples)

	realBuf := buffer.New(make([]float32, 0, 10))
	written := demod.Process(realBuf, srcBuf)
	require.Equal(t, len(iqSamples), written)

	output := realBuf.Items()
	require.Equal(t, len(iqSamples), len(output))

	// Verify output is real part of input
	for i := range output {
		require.Equal(t, float32(real(iqSamples[i])), output[i], "Output should equal real part of input")
	}
}
