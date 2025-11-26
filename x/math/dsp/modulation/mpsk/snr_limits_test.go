package mpsk

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestSNRLimits tests actual performance at various SNR levels, bypassing theoretical checks.
// This helps determine practical SNR limits for the implementation.
func TestSNRLimits(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 2 // QPSK
	require.NoError(t, cfg.Validate())

	payload := makePayload(7) // 7 bytes

	// Test SNR values from high to low
	// QPSK theoretically should work at ~10dB SNR for coherent detection
	// But practical implementations need more margin due to:
	// - PLL acquisition and tracking
	// - AGC convergence
	// - Phase ambiguity resolution
	// - Noise in preamble detection
	snrValues := []float32{30, 25, 20, 15, 12, 10}

	for _, snr := range snrValues {
		t.Run(fmt.Sprintf("%.0fdB", snr), func(t *testing.T) {
			runRoundtrip(t, cfg, payload, snr)
		})
	}
}


