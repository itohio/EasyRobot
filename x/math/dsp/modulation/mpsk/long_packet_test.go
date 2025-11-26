package mpsk

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	test_utils "github.com/itohio/EasyRobot/x/math/dsp/modulation/test_utils"
)

// TestLongPackets tests PLL stability for long packet transmissions at 16dB SNR.
// This verifies that the PLL maintains lock over extended periods and handles
// long payloads correctly without phase drift or loss of synchronization.
func TestLongPackets(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 2 // QPSK
	require.NoError(t, cfg.Validate())

	// Test various packet sizes to verify PLL stability
	// 16dB SNR is chosen as a moderate SNR level that requires stable PLL tracking
	packetSizes := []int{512, 1024, 2048, 4096}
	snrDb := float32(16.0)

	for _, size := range packetSizes {
		t.Run(fmt.Sprintf("%d_bytes", size), func(t *testing.T) {
			payload := test_utils.MakeData(size)
			runRoundtrip(t, cfg, payload, snrDb)
		})
	}
}

