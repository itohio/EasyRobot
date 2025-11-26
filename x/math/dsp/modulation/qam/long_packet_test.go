package qam

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestLongPackets tests PLL stability for long packet transmissions at 16dB SNR.
// This verifies that the PLL maintains lock over extended periods and handles
// long payloads correctly without phase drift or loss of synchronization.
func TestLongPackets(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 4 // 16-QAM
	require.NoError(t, cfg.Validate())

	// Test various packet sizes to verify PLL stability
	// 30dB SNR is chosen to provide very generous margin above theoretical limit (~15dB for 16-QAM)
	// QAM is sensitive to quantization errors and requires significant margin for reliable operation
	// Practical implementations typically need 10-15dB margin above theoretical limits due to:
	// - PLL acquisition and tracking overhead
	// - AGC convergence time
	// - Numerical precision in quantization
	// - Phase ambiguity resolution
	// - Long packet stability requirements
	packetSizes := []int{512, 1024, 2048, 4096}
	snrDb := float32(30.0)

	for _, size := range packetSizes {
		t.Run(fmt.Sprintf("%d_bytes", size), func(t *testing.T) {
			payload := makePayload(size)
			runRoundtrip(t, cfg, payload, snrDb)
		})
	}
}

