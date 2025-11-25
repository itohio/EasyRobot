package mfsk

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToneCount_MFSK_Formula(t *testing.T) {
	// M-ary FSK: M = 2^BitsPerSymbol
	// 2-FSK: M=2, BitsPerSymbol=1
	// 4-FSK: M=4, BitsPerSymbol=2
	// 8-FSK: M=8, BitsPerSymbol=3
	testCases := []struct {
		bitsPerSymbol int
		expectedM     int
		description   string
	}{
		{1, 2, "2-FSK"},
		{2, 4, "4-FSK"},
		{3, 8, "8-FSK"},
		{4, 16, "16-FSK"},
	}

	for _, tc := range testCases {
		cfg := DefaultConfig()
		cfg.BitsPerSymbol = tc.bitsPerSymbol
		require.NoError(t, cfg.Validate())
		
		actualM := cfg.toneCount()
		require.Equalf(t, tc.expectedM, actualM,
			"%s: expected M=%d frequencies for BitsPerSymbol=%d, got M=%d",
			tc.description, tc.expectedM, tc.bitsPerSymbol, actualM)
		
		// Verify: M = 2^BitsPerSymbol
		expectedM := 1 << tc.bitsPerSymbol
		require.Equalf(t, expectedM, actualM,
			"Formula check: M should equal 2^BitsPerSymbol = 2^%d = %d, got %d",
			tc.bitsPerSymbol, expectedM, actualM)
	}
}

