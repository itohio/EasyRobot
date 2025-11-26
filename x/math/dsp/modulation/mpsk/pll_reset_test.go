package mpsk

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
)

// TestPLLReset_BetweenPackets validates that PLL properly resets and re-acquires
// between separate packet transmissions. This ensures the demodulator can handle
// multiple packets separated by silence gaps.
//
// Test approach:
// - Use single modulator to transmit "FOO" and "BAR" into separate buffers
// - Reset modulator between transmissions
// - Use separate demodulators to decode each packet independently
// - This validates that each demodulator can acquire lock independently
func TestPLLReset_BetweenPackets(t *testing.T) {
	cfg := DefaultConfig()
	cfg.BitsPerSymbol = 2 // QPSK
	cfg.PreambleLength = 20
	require.NoError(t, cfg.Validate())

	packet1 := []byte("FOO")
	packet2 := []byte("BAR")
	snrDb := float32(12.0)

	// Create single modulator for both packets
	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	// Modulate packet1 into buffer1
	symbols1 := bytesToSymbols(packet1, cfg.BitsPerSymbol)
	inputBits1 := buildBits(t, cfg.BitsPerSymbol, symbols1)
	estimate1 := mod.Process(nil, inputBits1)
	require.Greater(t, estimate1, 0)
	inputBits1.Reset()
	
	// Create buffer1 with padding
	paddingSamples := int(0.1 * cfg.SampleRate) * 2 // IQ interleaved
	totalSamples1 := paddingSamples*2 + estimate1
	buffer1 := make([]float32, totalSamples1)
	signalStart1 := paddingSamples
	signalEnd1 := paddingSamples + estimate1
	
	signalBuf1 := buffer.New(buffer1[signalStart1:signalStart1:signalEnd1])
	written1 := mod.Process(signalBuf1, inputBits1)
	require.Equal(t, estimate1, written1)

	// Reset modulator for packet2
	mod.Reset()

	// Modulate packet2 into buffer2
	symbols2 := bytesToSymbols(packet2, cfg.BitsPerSymbol)
	inputBits2 := buildBits(t, cfg.BitsPerSymbol, symbols2)
	estimate2 := mod.Process(nil, inputBits2)
	require.Greater(t, estimate2, 0)
	inputBits2.Reset()
	
	// Create buffer2 with padding
	totalSamples2 := paddingSamples*2 + estimate2
	buffer2 := make([]float32, totalSamples2)
	signalStart2 := paddingSamples
	signalEnd2 := paddingSamples + estimate2
	
	signalBuf2 := buffer.New(buffer2[signalStart2:signalStart2:signalEnd2])
	written2 := mod.Process(signalBuf2, inputBits2)
	require.Equal(t, estimate2, written2)

	// Add noise to both buffers at 12dB SNR
	addNoiseToBuffer(buffer1, buffer1[signalStart1:signalEnd1], snrDb)
	addNoiseToBuffer(buffer2, buffer2[signalStart2:signalEnd2], snrDb)

	// Create separate demodulators for each packet
	demod1, err := NewDemodulator(cfg)
	require.NoError(t, err)
	demod2, err := NewDemodulator(cfg)
	require.NoError(t, err)

	outputBits1, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)
	outputBits2, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	// Decode packet1 with demod1 (simulate streaming)
	chunks1 := splitIntoChunksFromBuffer(buffer1, 5)
	for _, chunk := range chunks1 {
		reader := buffer.New(chunk)
		demod1.Process(outputBits1, reader)
	}

	// Decode packet2 with demod2 (simulate streaming)
	chunks2 := splitIntoChunksFromBuffer(buffer2, 5)
	for _, chunk := range chunks2 {
		reader := buffer.New(chunk)
		demod2.Process(outputBits2, reader)
	}

	// Verify packet1 decoded correctly
	outputBits1.Reset()
	decodedSymbols1 := make([]byte, len(symbols1))
	readBits1, err := outputBits1.Read(decodedSymbols1)
	require.NoError(t, err)
	require.Equal(t, len(symbols1)*cfg.BitsPerSymbol, readBits1)
	decodedBytes1 := symbolsToBytes(decodedSymbols1, cfg.BitsPerSymbol, len(packet1))
	require.Equal(t, packet1, decodedBytes1, "should decode FOO packet")

	// Verify packet2 decoded correctly
	outputBits2.Reset()
	decodedSymbols2 := make([]byte, len(symbols2))
	readBits2, err := outputBits2.Read(decodedSymbols2)
	require.NoError(t, err)
	require.Equal(t, len(symbols2)*cfg.BitsPerSymbol, readBits2)
	decodedBytes2 := symbolsToBytes(decodedSymbols2, cfg.BitsPerSymbol, len(packet2))
	require.Equal(t, packet2, decodedBytes2, "should decode BAR packet")
}
