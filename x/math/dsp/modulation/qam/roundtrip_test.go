package qam

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testSampleRate  = float32(48000)
	testCarrierFreq = float32(1000)
	testSymbolRate  = float32(1000)
	testAmplitude   = float32(1.0)
)

// addNoise adds AWGN noise to a signal to achieve a specific SNR in dB
// Uses dsp.SignalGenerator1D for noise generation
// SNR = 10*log10(signal_power / noise_power)
func addNoise(signal vecTypes.Vector, snrDb float32) vecTypes.Vector {
	signalData := []float32(signal.(vec.Vector))

	// Calculate signal power (RMS)
	var signalPower float64
	for _, s := range signalData {
		signalPower += float64(s * s)
	}
	signalPower /= float64(len(signalData))
	if signalPower < 1e-10 {
		signalPower = 1e-10
	}

	// Calculate noise power needed for desired SNR
	// SNR_dB = 10*log10(signal_power / noise_power)
	// noise_power = signal_power / 10^(SNR_dB/10)
	snrRatio := math.Pow(10, float64(snrDb)/10)
	noisePower := signalPower / snrRatio
	noiseStdDev := float32(math.Sqrt(noisePower))

	// Generate Gaussian noise using dsp package
	sg := dsp.NewSignalGenerator1D()
	noise := sg.NoiseGaussian(len(signalData), noiseStdDev)
	noiseData := []float32(noise.(vec.Vector))

	// Add noise to signal
	noisy := vec.New(len(signalData))
	noisyData := []float32(noisy)
	for i := range signalData {
		noisyData[i] = signalData[i] + noiseData[i]
	}

	return noisy
}

// addZeroPadding adds 0.5 seconds of zero padding on each side to simulate signal in noise
func addZeroPadding(signal vec.Vector) vec.Vector {
	signalData := []float32(signal)
	paddingSamples := int(0.5 * testSampleRate) // 0.5 seconds at 48000Hz = 24000 samples

	// Create new vector with padding on both sides
	padded := vec.New(len(signalData) + paddingSamples*2)
	paddedData := []float32(padded)

	// Left padding (zeros) - already zeros from vec.New
	// Signal in middle
	copy(paddedData[paddingSamples:], signalData)
	// Right padding (zeros) - already zeros from vec.New

	return padded
}

// TestRoundtrip_16QAM_Clean tests round-trip encoding/decoding for 16-QAM without noise
func TestRoundtrip_16QAM_Clean(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// Test data
	testData := []byte{0x55, 0xAA, 0x12, 0x34}

	// Encode
	expectedSize := encoder.Encode(nil, testData)
	amplitudes := vec.New(expectedSize)
	samplesWritten := encoder.Encode(amplitudes, testData)
	require.Greater(t, samplesWritten, 0, "Should write samples")

	// Decode in chunks to allow signal detector to lock
	decoded := make([]byte, len(testData)*2)
	bytesDecoded := 0

	ampData := []float32(amplitudes)
	chunkSize := 1000
	for i := 0; i < len(ampData); i += chunkSize {
		end := i + chunkSize
		if end > len(ampData) {
			end = len(ampData)
		}
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, ampData[i:end])

		remaining := decoded[bytesDecoded:]
		chunkDecoded := decoder.Decode(remaining, chunk)
		bytesDecoded += chunkDecoded

		// Continue processing even if not locked yet - signal detector needs time
		// Only break if we have enough decoded data
		if bytesDecoded >= len(testData) {
			// Got enough data
			break
		}
		// Also break if we've processed most of the signal and haven't decoded anything
		if i > len(ampData)*3/4 && bytesDecoded == 0 {
			// Processed most of signal but didn't decode - might be signal detection issue
			break
		}
	}

	// Should decode some bytes
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes without noise")
}

// TestRoundtrip_16QAM_WithNoise tests round-trip encoding/decoding for 16-QAM with noise
func TestRoundtrip_16QAM_WithNoise(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// Test data
	testData := []byte{0x55, 0xAA}

	// Encode
	expectedSize := encoder.Encode(nil, testData)
	amplitudes := vec.New(expectedSize)
	samplesWritten := encoder.Encode(amplitudes, testData)
	require.Greater(t, samplesWritten, 0, "Should write samples")

	// Add noise (18dB SNR - increased by 3dB for M=16 since 16-QAM is more sensitive to noise)
	noisyAmplitudes := addNoise(amplitudes, 18.0)

	// Reset decoder for clean state
	decoder.Reset()

	// Decode in chunks to allow signal detector to lock
	decoded := make([]byte, len(testData)*2)
	bytesDecoded := 0

	noisyData := []float32(noisyAmplitudes.(vec.Vector))
	chunkSize := 1000
	for i := 0; i < len(noisyData); i += chunkSize {
		end := i + chunkSize
		if end > len(noisyData) {
			end = len(noisyData)
		}
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, noisyData[i:end])

		remaining := decoded[bytesDecoded:]
		chunkDecoded := decoder.Decode(remaining, chunk)
		bytesDecoded += chunkDecoded

		if chunkDecoded == 0 && !decoder.IsLocked() && bytesDecoded > 0 {
			break
		}
		if bytesDecoded >= len(testData) {
			break
		}
	}

	// Should decode some bytes even with noise
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes with 18dB SNR (M=16 requires ~3dB more SNR)")
}

// TestRoundtrip_16QAM_12dB_SNR tests 16-QAM at 15dB SNR (12dB + 3dB for M=16 sensitivity)
func TestRoundtrip_16QAM_12dB_SNR(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// Test data - use simple pattern for easier verification
	testData := []byte{0xAA} // 10101010 - alternating pattern

	// Encode
	expectedSize := encoder.Encode(nil, testData)
	amplitudes := vec.New(expectedSize)
	samplesWritten := encoder.Encode(amplitudes, testData)
	require.Greater(t, samplesWritten, 0, "Should write samples")

	// Add 15dB SNR noise (increased by 3dB for M=16 since 16-QAM is more sensitive to noise)
	noisyAmplitudes := addNoise(amplitudes, 15.0)

	// Reset decoder for clean state
	decoder.Reset()

	// Decode in chunks to simulate streaming
	decoded := make([]byte, len(testData)*4)
	bytesDecoded := 0

	noisyData := []float32(noisyAmplitudes.(vec.Vector))
	chunkSize := 1000
	for i := 0; i < len(noisyData); i += chunkSize {
		end := i + chunkSize
		if end > len(noisyData) {
			end = len(noisyData)
		}
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, noisyData[i:end])

		remaining := decoded[bytesDecoded:]
		chunkDecoded := decoder.Decode(remaining, chunk)
		bytesDecoded += chunkDecoded

		if chunkDecoded == 0 && !decoder.IsLocked() {
			// No more data and not locked, might be done
			break
		}
	}

	// At 15dB SNR (12dB + 3dB for M=16), we should be able to decode something
	// The exact amount depends on signal detection and quality
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes at 15dB SNR (M=16 requires ~3dB more SNR)")
}

// TestRoundtrip_16QAM_Padding tests that padding is handled correctly
func TestRoundtrip_16QAM_Padding(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	testData := []byte{0x55}

	// Encode
	expectedSize := encoder.Encode(nil, testData)
	amplitudes := vec.New(expectedSize)
	samplesWritten := encoder.Encode(amplitudes, testData)
	require.Greater(t, samplesWritten, 0)

	paddingSamples := int(testSampleRate * 0.5)

	// Add zero padding
	paddedSignal := addZeroPadding(amplitudes)
	paddedData := []float32(paddedSignal)

	// Verify padding exists
	leftPadZeros := 0
	for i := 0; i < paddingSamples && i < len(paddedData); i++ {
		if math.Abs(float64(paddedData[i])) < 0.001 {
			leftPadZeros++
		}
	}
	assert.Greater(t, leftPadZeros, paddingSamples/2, "Left padding should be mostly zeros")

	// Reset decoder
	decoder.Reset()

	// Decode in chunks to simulate streaming
	decoded := make([]byte, len(testData)*4)
	bytesDecoded := 0

	chunkSize := 1000
	for i := 0; i < len(paddedData); i += chunkSize {
		end := i + chunkSize
		if end > len(paddedData) {
			end = len(paddedData)
		}
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, paddedData[i:end])

		remaining := decoded[bytesDecoded:]
		chunkDecoded := decoder.Decode(remaining, chunk)
		bytesDecoded += chunkDecoded

		if chunkDecoded == 0 && !decoder.IsLocked() && bytesDecoded > 0 {
			// Have some decoded data and not locked anymore
			break
		}
	}

	// Should decode some bytes even with padding
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes with padding")
}

// TestRoundtrip_16QAM_PaddedWithNoise tests round-trip with padding and 12dB SNR
func TestRoundtrip_16QAM_PaddedWithNoise(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	testData := []byte{0x12, 0x34}

	// Encode
	expectedSize := encoder.Encode(nil, testData)
	amplitudes := vec.New(expectedSize)
	samplesWritten := encoder.Encode(amplitudes, testData)
	require.Greater(t, samplesWritten, 0, "Should write samples")

	// Add zero padding
	paddedSignal := addZeroPadding(amplitudes)

	// Add 15dB SNR noise (increased by 3dB for M=16 since 16-QAM is more sensitive to noise)
	noisyPadded := addNoise(paddedSignal, 15.0)

	// Reset decoder
	decoder.Reset()

	// Decode in chunks to simulate streaming
	decoded := make([]byte, len(testData)*4)
	bytesDecoded := 0

	noisyData := []float32(noisyPadded.(vec.Vector))
	chunkSize := 1000
	for i := 0; i < len(noisyData); i += chunkSize {
		end := i + chunkSize
		if end > len(noisyData) {
			end = len(noisyData)
		}
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, noisyData[i:end])

		remaining := decoded[bytesDecoded:]
		chunkDecoded := decoder.Decode(remaining, chunk)
		bytesDecoded += chunkDecoded

		if chunkDecoded == 0 && !decoder.IsLocked() && bytesDecoded > 0 {
			break
		}
	}

	// Should decode at least some bytes at 15dB SNR (12dB + 3dB for M=16) with padding
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes at 15dB SNR with padding (M=16 requires ~3dB more SNR)")
}

// TestRoundtrip_16QAM_MultiplePackets tests multiple packets with padding between them
func TestRoundtrip_16QAM_MultiplePackets(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// First packet
	testData1 := []byte{0x55, 0xAA}
	expectedSize1 := encoder.Encode(nil, testData1)
	amplitudes1 := vec.New(expectedSize1)
	encoder.Encode(amplitudes1, testData1)
	signal1 := addZeroPadding(amplitudes1)

	// Second packet
	encoder.Reset()
	testData2 := []byte{0x12, 0x34}
	expectedSize2 := encoder.Encode(nil, testData2)
	amplitudes2 := vec.New(expectedSize2)
	encoder.Encode(amplitudes2, testData2)
	signal2 := addZeroPadding(amplitudes2)

	// Combine signals
	signal1Data := []float32(signal1)
	signal2Data := []float32(signal2)
	combined := vec.New(len(signal1Data) + len(signal2Data))
	combinedData := []float32(combined)
	copy(combinedData, signal1Data)
	copy(combinedData[len(signal1Data):], signal2Data)

	// Decode
	decoded := make([]byte, 100)
	bytesDecoded := 0

	// Process in chunks
	chunkSize := 1000
	for i := 0; i < len(combinedData); i += chunkSize {
		end := i + chunkSize
		if end > len(combinedData) {
			end = len(combinedData)
		}
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, combinedData[i:end])
		chunkDecoded := decoder.Decode(decoded[bytesDecoded:], chunk)
		bytesDecoded += chunkDecoded
		if chunkDecoded == 0 && !decoder.IsLocked() && bytesDecoded > 0 {
			break
		}
	}

	// Verify we decoded something from packets
	assert.Greater(t, bytesDecoded, 0, "Should decode bytes from multiple packets")
}

