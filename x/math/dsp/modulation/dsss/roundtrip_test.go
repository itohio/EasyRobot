package dsss

import (
	"math/rand"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test helper: create standard test configuration
func createTestConfig(bitsPerSymbol int) Config {
	sampleRate := float32(96000)
	carrierFreq := float32(20000)
	symbolRate := float32(1000)
	spreadingFactor := 8

	codes := generateOrthogonalCodes(bitsPerSymbol, spreadingFactor)
	samplesPerChip := int(sampleRate / (symbolRate * float32(spreadingFactor)))

	return Config{
		BitsPerSymbol:       bitsPerSymbol,
		ModulationOrder:     1 << bitsPerSymbol,
		Codes:               codes,
		SpreadingFactor:     spreadingFactor,
		SampleRate:          sampleRate,
		SymbolRate:          symbolRate,
		CarrierFreq:         carrierFreq,
		SamplesPerChip:      samplesPerChip,
		PulseShape:          "raised_cosine",
		Rolloff:             0.35,
		PulseFilterSpan:     4,
		PreambleLength:      4,
		PreamblePattern:     []int{0, 1, 0, 1},
		EnergyThreshold:     0.001,
		DetectionWindow:     samplesPerChip * spreadingFactor,
		TransmissionTimeout: 3,
		Amplitude:           1.0,
	}
}

func generateOrthogonalCodes(numCodes, length int) [][]int8 {
	if numCodes <= 0 {
		numCodes = 1
	}
	codes := make([][]int8, numCodes)
	for i := range codes {
		codes[i] = make([]int8, length)
		for j := range codes[i] {
			if ((i + j) & 1) == 0 {
				codes[i][j] = 1
			} else {
				codes[i][j] = -1
			}
		}
	}
	return codes
}

func TestRoundtrip_1Byte(t *testing.T) {
	config := createTestConfig(1)
	testData := []byte{0x42}
	runRoundtripTest(t, config, testData, false)
}

func TestRoundtrip_4Bytes(t *testing.T) {
	config := createTestConfig(1)
	testData := []byte{0x01, 0x23, 0x45, 0x67}
	runRoundtripTest(t, config, testData, false)
}

func TestRoundtrip_33Bytes(t *testing.T) {
	config := createTestConfig(1)
	testData := make([]byte, 33)
	for i := range testData {
		testData[i] = byte(i)
	}
	runRoundtripTest(t, config, testData, false)
}

func TestRoundtrip_Noise_20dB(t *testing.T) {
	config := createTestConfig(1)
	testData := []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD}
	runRoundtripTest(t, config, testData, true)
}

func runRoundtripTest(t *testing.T, config Config, payload []byte, withNoise bool) {
	inputBits := buildInputBits(t, config, payload)
	mod, demod := buildProcessors(t, config)

	modulated := modulateSamples(t, mod, inputBits)
	if withNoise {
		modulated = addNoise(modulated, 20.0)
	}
	outputBits := demodulateSamples(t, demod, modulated)

	verifyRoundtrip(t, config, payload, outputBits)
}

func buildProcessors(t *testing.T, config Config) (*Modulator, *Demodulator) {
	t.Helper()

	mod, err := NewModulator(config)
	require.NoError(t, err)

	demod, err := NewDemodulator(config)
	require.NoError(t, err)

	return mod, demod
}

func buildInputBits(t *testing.T, config Config, payload []byte) *buffer.Bits {
	t.Helper()

	symbols := bytesToSymbols(payload, config.BitsPerSymbol)
	bitsBuf, err := buffer.NewBits(config.BitsPerSymbol)
	require.NoError(t, err)

	_, err = bitsBuf.Write(symbols)
	require.NoError(t, err)
	bitsBuf.Reset()

	return bitsBuf
}

func modulateSamples(t *testing.T, mod *Modulator, input buffer.Reader[byte]) []float32 {
	t.Helper()

	sampleBuffer := buffer.New(make([]float32, 0, 1<<15))
	written := mod.Process(sampleBuffer, input)
	require.Greater(t, written, 0)

	silence := make([]float32, int(mod.config.SampleRate/2))
	all := make([]float32, 0, len(silence)*2+sampleBuffer.Len())
	all = append(all, silence...)
	all = append(all, sampleBuffer.Items()...)
	all = append(all, silence...)

	return all
}

func demodulateSamples(t *testing.T, demod *Demodulator, samples []float32) *buffer.Bits {
	t.Helper()

	outputBits, err := buffer.NewBits(demod.config.BitsPerSymbol)
	require.NoError(t, err)

	chunkSize := len(samples) / 3
	if chunkSize == 0 {
		chunkSize = len(samples)
	}

	for offset := 0; offset < len(samples); offset += chunkSize {
		end := offset + chunkSize
		if end > len(samples) {
			end = len(samples)
		}
		reader := buffer.New(samples[offset:end])
		demod.Process(outputBits, reader)
	}
	outputBits.Reset()
	return outputBits
}

func verifyRoundtrip(t *testing.T, config Config, expected []byte, decodedBits *buffer.Bits) {
	t.Helper()

	totalSymbols := decodedBits.Bits() / config.BitsPerSymbol
	require.Greaterf(t, totalSymbols, 0, "no symbols decoded")

	symbolBuf := make([]byte, totalSymbols)
	nBits, err := decodedBits.Read(symbolBuf)
	require.NoError(t, err)
	require.Equal(t, totalSymbols*config.BitsPerSymbol, nBits)

	decoded := symbolsToBytes(symbolBuf, config.BitsPerSymbol, len(expected))
	assert.Equal(t, expected, decoded)
}

func bytesToSymbols(data []byte, bitsPerSymbol int) []byte {
	if bitsPerSymbol <= 0 {
		return nil
	}
	totalBits := len(data) * 8
	symbols := (totalBits + bitsPerSymbol - 1) / bitsPerSymbol

	out := make([]byte, symbols)
	bitIdx := 0
	for i := 0; i < symbols; i++ {
		var value byte
		for j := 0; j < bitsPerSymbol; j++ {
			value <<= 1
			if bitIdx < totalBits {
				byteIdx := bitIdx / 8
				bitPos := 7 - (bitIdx % 8)
				value |= (data[byteIdx] >> bitPos) & 0x1
			}
			bitIdx++
		}
		out[i] = value
	}
	return out
}

func symbolsToBytes(symbols []byte, bitsPerSymbol int, expectedBytes int) []byte {
	if bitsPerSymbol <= 0 || expectedBytes <= 0 {
		return nil
	}
	out := make([]byte, expectedBytes)
	totalBits := expectedBytes * 8
	bitIdx := 0

	for _, symbol := range symbols {
		for bit := bitsPerSymbol - 1; bit >= 0; bit-- {
			if bitIdx >= totalBits {
				return out
			}
			byteIdx := bitIdx / 8
			bitPos := 7 - (bitIdx % 8)
			bitVal := (symbol >> uint(bit)) & 0x1
			out[byteIdx] |= bitVal << uint(bitPos)
			bitIdx++
		}
	}
	return out
}

func addNoise(signal []float32, snrDB float32) []float32 {
	if len(signal) == 0 {
		return nil
	}

	signalPower := float32(0)
	for _, sample := range signal {
		signalPower += sample * sample
	}
	signalPower /= float32(len(signal))

	if signalPower == 0 {
		return append([]float32(nil), signal...)
	}

	snrLinear := math32.Pow(10, snrDB/10)
	noisePower := signalPower / snrLinear
	noiseStd := math32.Sqrt(noisePower)

	noisy := make([]float32, len(signal))
	copy(noisy, signal)

	rng := rand.New(rand.NewSource(42))
	for i := range noisy {
		noisy[i] += float32(rng.NormFloat64()) * noiseStd
	}
	return noisy
}
