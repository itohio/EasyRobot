package mfsk

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// Test8FSK_Noise tests 8-FSK specifically with various noise levels
func Test8FSK_Noise(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 3 // 8-FSK: 3 bits = 8 frequencies
	require.NoError(t, cfg.Validate())
	
	// Verify it's actually 8-FSK
	require.Equal(t, 8, cfg.toneCount(), "8-FSK should have 8 frequencies")
	
	payload := makePayload(5)
	targetBER := float32(1e-5)
	marginDb := float32(2.5)
	
	minSNR := cfg.MinimumSNR(targetBER, marginDb)
	t.Logf("8-FSK: theoretical minimum SNR: %.1f dB", minSNR)
	
	// Test at various SNR levels
	testSNRs := []float32{25, 22, 20}
	for _, snr := range testSNRs {
		if snr < minSNR {
			t.Skipf("skipping SNR=%.1fdB (below theoretical minimum %.1fdB)", snr, minSNR)
			continue
		}
		t.Run(fmt.Sprintf("SNR_%.1fdB", snr), func(t *testing.T) {
			runRoundtrip(t, cfg, payload, snr)
		})
	}
}

// Test16FSK_Noise tests 16-FSK specifically with various noise levels
func Test16FSK_Noise(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 4 // 16-FSK: 4 bits = 16 frequencies
	require.NoError(t, cfg.Validate())
	
	// Verify it's actually 16-FSK
	require.Equal(t, 16, cfg.toneCount(), "16-FSK should have 16 frequencies")
	
	payload := makePayload(5)
	targetBER := float32(1e-5)
	marginDb := float32(2.5)
	
	minSNR := cfg.MinimumSNR(targetBER, marginDb)
	t.Logf("16-FSK: theoretical minimum SNR: %.1f dB", minSNR)
	t.Logf("16-FSK: bandwidth info - tone BW=%.1fHz, total BW=%.1fHz, separation=%.1fHz",
		cfg.ToneBandwidth(), cfg.RequiredBandwidth(), cfg.SymbolSeparationHz())
	
	// Test at various SNR levels (16-FSK needs higher SNR)
	testSNRs := []float32{28, 25, 22}
	for _, snr := range testSNRs {
		if snr < minSNR {
			t.Skipf("skipping SNR=%.1fdB (below theoretical minimum %.1fdB)", snr, minSNR)
			continue
		}
		t.Run(fmt.Sprintf("SNR_%.1fdB", snr), func(t *testing.T) {
			runRoundtrip(t, cfg, payload, snr)
		})
	}
}

// Test8FSK_Bandwidth verifies 8-FSK bandwidth calculations
func Test8FSK_Bandwidth(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 3 // 8-FSK
	require.NoError(t, cfg.Validate())
	
	require.Equal(t, 8, cfg.toneCount())
	
	toneBW := cfg.ToneBandwidth()
	totalBW := cfg.RequiredBandwidth()
	separation := cfg.SymbolSeparationHz()
	
	t.Logf("8-FSK bandwidth: tone BW=%.1fHz, total BW=%.1fHz, separation=%.1fHz",
		toneBW, totalBW, separation)
	
	// For 8-FSK with orthogonal spacing, separation should equal symbol rate
	symbolRate := cfg.SymbolRate()
	require.InDelta(t, symbolRate, separation, 1.0, 
		"symbol separation should equal symbol rate for orthogonal FSK")
	
	// Tone bandwidth should be (M-1) * separation
	expectedToneBW := separation * float32(cfg.toneCount()-1)
	require.InDelta(t, expectedToneBW, toneBW, 1.0,
		"tone bandwidth should be (M-1) * separation")
}

// Test16FSK_Bandwidth verifies 16-FSK bandwidth calculations
func Test16FSK_Bandwidth(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 4 // 16-FSK
	require.NoError(t, cfg.Validate())
	
	require.Equal(t, 16, cfg.toneCount())
	
	toneBW := cfg.ToneBandwidth()
	totalBW := cfg.RequiredBandwidth()
	separation := cfg.SymbolSeparationHz()
	lowestFreq, highestFreq := cfg.CarrierHz, cfg.CarrierHz + toneBW
	
	t.Logf("16-FSK bandwidth: tone BW=%.1fHz, total BW=%.1fHz, separation=%.1fHz",
		toneBW, totalBW, separation)
	t.Logf("16-FSK frequency range: %.1fHz to %.1fHz", lowestFreq, highestFreq)
	
	// Verify it fits within Nyquist
	nyquist := cfg.SampleRate / 2
	require.Less(t, highestFreq, nyquist,
		"highest frequency %.1fHz should be below Nyquist %.1fHz", highestFreq, nyquist)
	
	// For 16-FSK with orthogonal spacing, separation should equal symbol rate
	symbolRate := cfg.SymbolRate()
	require.InDelta(t, symbolRate, separation, 1.0,
		"symbol separation should equal symbol rate for orthogonal FSK")
	
	// Tone bandwidth should be (M-1) * separation
	expectedToneBW := separation * float32(cfg.toneCount()-1)
	require.InDelta(t, expectedToneBW, toneBW, 1.0,
		"tone bandwidth should be (M-1) * separation")
}

