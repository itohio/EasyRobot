package qam

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	test_utils "github.com/itohio/EasyRobot/x/math/dsp/modulation/test_utils"
)

// TestSNRLimits tests actual performance at various SNR levels, bypassing theoretical checks.
// This helps determine practical SNR limits for the implementation.
func TestSNRLimits(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 4 // 16-QAM
	require.NoError(t, cfg.Validate())

	payload := test_utils.MakeData(7) // 7 bytes

	// Test SNR values from high to low
	// 16-QAM theoretically should work at ~15dB SNR for coherent detection
	// But practical implementations need more margin due to:
	// - PLL acquisition and tracking
	// - AGC convergence
	// - Noise in signal detection
	snrValues := []float32{30, 25, 20, 15, 12, 10}

	for _, snr := range snrValues {
		t.Run(fmt.Sprintf("%.0fdB", snr), func(t *testing.T) {
			runRoundtrip(t, cfg, payload, snr)
		})
	}
}

