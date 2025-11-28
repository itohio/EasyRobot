package main

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/gray"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/hamming"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/rs"
	"github.com/itohio/EasyRobot/x/math/shape"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// dbToAmplitude converts dB to linear amplitude
func dbToAmplitude(db float64) float32 {
	return float32(math.Pow(10, db/20))
}

// parseDataType parses data type string to DataType
func parseDataType(s string) (types.DataType, error) {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch s {
	case "INT16", "I16":
		return types.INT16, nil
	case "INT32", "I32":
		return types.INT32, nil
	case "FP32", "F32", "FLOAT32":
		return types.FP32, nil
	case "UINT8", "U8":
		return types.UINT8, nil
	case "INT8", "I8":
		return types.INT8, nil
	default:
		return types.DT_UNKNOWN, fmt.Errorf("unknown data type: %s (supported: INT16, INT32, FP32, UINT8, INT8)", s)
	}
}

// parseFloatSlice parses comma-separated float values
func parseFloatSlice(s string) ([]float64, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	result := make([]float64, len(parts))
	for i, part := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float value '%s': %w", part, err)
		}
		result[i] = val
	}
	return result, nil
}

// parseIntSlice parses comma-separated int values
func parseIntSlice(s string) ([]int, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	result := make([]int, len(parts))
	for i, part := range parts {
		val, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("invalid int value '%s': %w", part, err)
		}
		result[i] = val
	}
	return result, nil
}

// parseData parses data from string or hex string
func parseData(dataStr, hexStr string) ([]byte, error) {
	if hexStr != "" {
		data, err := hex.DecodeString(hexStr)
		if err != nil {
			return nil, fmt.Errorf("invalid hex string: %w", err)
		}
		return data, nil
	}
	if dataStr != "" {
		return []byte(dataStr), nil
	}
	return []byte("Hello"), nil // default
}

// tensorDataType converts types.DataType to tensortypes.DataType
func tensorDataType(dt types.DataType) tensortypes.DataType {
	return tensortypes.DataType(dt)
}

// applyPadding adds padding to the beginning and end of a signal
func applyPadding(signal vec.Vector, padSamples int, padType, padAmplitude string, sampleRate float32) vec.Vector {
	if padSamples <= 0 {
		return signal
	}

	signalData := signal.View().(vec.Vector)
	originalLen := len(signalData)
	newLen := originalLen + 2*padSamples
	result := vec.New(newLen)
	resultData := []float32(result)

	// Generate padding
	var padData []float32
	padType = strings.ToLower(padType)
	if padType == "noise" {
		// Generate noise padding
		padAmpDb, err := strconv.ParseFloat(padAmplitude, 64)
		if err != nil {
			padAmpDb = -20 // Default
		}
		padAmp := dbToAmplitude(padAmpDb)
		gen := dsp.NewNoiseWhiteGenerator(padAmp)
		padData = make([]float32, padSamples)
		gen.Generate(padData)
	} else {
		// Zero padding
		padData = make([]float32, padSamples)
	}

	// Copy: padding + signal + padding
	copy(resultData[0:padSamples], padData)
	copy(resultData[padSamples:padSamples+originalLen], signalData)
	copy(resultData[padSamples+originalLen:], padData)

	return result
}

// applyNoise adds noise on top of the signal
func applyNoise(signal vec.Vector, noiseAmplitude float32, noiseType string, sampleRate float32) vec.Vector {
	signalData := signal.View().(vec.Vector)
	length := len(signalData)

	noiseType = strings.ToLower(noiseType)
	var gen dsp.Generator
	switch noiseType {
	case "pink":
		pinkGen := dsp.NewNoisePinkGenerator(noiseAmplitude)
		gen = &pinkGen
	case "brown":
		brownGen := dsp.NewNoiseBrownGenerator(noiseAmplitude)
		gen = &brownGen
	default: // white
		whiteGen := dsp.NewNoiseWhiteGenerator(noiseAmplitude)
		gen = &whiteGen
	}

	// Generate noise
	noiseData := make([]float32, length)
	gen.Generate(noiseData)

	// Add noise to signal
	result := vec.New(length)
	resultData := []float32(result)
	for i := 0; i < length; i++ {
		resultData[i] = signalData[i] + noiseData[i]
		// Clamp to [-1, 1]
		if resultData[i] > 1.0 {
			resultData[i] = 1.0
		} else if resultData[i] < -1.0 {
			resultData[i] = -1.0
		}
	}

	return result
}

// isASCII checks if all bytes in the data are printable ASCII characters
func isASCII(data []byte) bool {
	for _, b := range data {
		if b < 32 || b > 126 {
			return false
		}
	}
	return true
}

const defaultChunkParts = 8

type demodProcessConfig struct {
	BitsPerSymbol int
	Channels      int
	ChunkParts    int
	OnDecoded     func([]byte) error
	OnFrame       func([]float32) error
}

// demodulateAudio routes samples from the unmarshaller through a streaming demodulator.
func demodulateAudio(unmarshaller types.Unmarshaller, demod chain.Processor[byte, float32], ctx context.Context, cfg demodProcessConfig) error {
	if cfg.BitsPerSymbol <= 0 {
		return fmt.Errorf("bits-per-symbol must be positive")
	}
	if cfg.Channels <= 0 {
		cfg.Channels = 1
	}
	if cfg.ChunkParts <= 0 {
		cfg.ChunkParts = defaultChunkParts
	}
	if cfg.OnDecoded == nil {
		cfg.OnDecoded = func([]byte) error { return nil }
	}

	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	if err != nil {
		return fmt.Errorf("failed to allocate bits buffer: %w", err)
	}
	consumedBits := 0

	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err == nil {
		defer stream.Close()
		return processFrameStream(stream, demod, ctx, cfg, outputBits, &consumedBits)
	}

	var tensor types.Tensor
	if err := unmarshaller.Unmarshal(nil, &tensor); err != nil {
		return fmt.Errorf("failed to read from unmarshaller (stream/tensor): %w", err)
	}
	defer tensor.Release()

	return processTensorInput(tensor, demod, ctx, cfg, outputBits, &consumedBits)
}

func processFrameStream(stream types.FrameStream, demod chain.Processor[byte, float32], ctx context.Context, cfg demodProcessConfig, bits *buffer.Bits, consumedBits *int) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case frame, ok := <-stream.C:
			if !ok {
				return flushDecodedSymbols(bits, cfg.BitsPerSymbol, consumedBits, cfg.OnDecoded)
			}
			if len(frame.Tensors) == 0 {
				continue
			}
			t := frame.Tensors[0]
			shape := t.Shape()
			tensorData := t.Data()
			convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, tensorData)
			t.Release()
			if convertedData == nil {
				continue
			}
			data, ok := convertedData.([]float32)
			if !ok {
				continue
			}
			if err := validateChannelCount(shape, cfg.Channels); err != nil {
				return err
			}
			if cfg.OnFrame != nil {
				if err := cfg.OnFrame(data); err != nil {
					return err
				}
			}
			if err := processSampleBuffer(data, demod, ctx, cfg, bits, consumedBits); err != nil {
				return err
			}
		}
	}
}

func processTensorInput(tensor types.Tensor, demod chain.Processor[byte, float32], ctx context.Context, cfg demodProcessConfig, bits *buffer.Bits, consumedBits *int) error {
	shape := tensor.Shape()
	tensorData := tensor.Data()
	convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, tensorData)
	if convertedData == nil {
		return fmt.Errorf("failed to convert tensor data to float32")
	}
	data, ok := convertedData.([]float32)
	if !ok {
		return fmt.Errorf("converted data is not []float32: %T", convertedData)
	}
	if err := validateChannelCount(shape, cfg.Channels); err != nil {
		return err
	}
	if cfg.OnFrame != nil {
		if err := cfg.OnFrame(data); err != nil {
			return err
		}
	}
	if err := processSampleBuffer(data, demod, ctx, cfg, bits, consumedBits); err != nil {
		return err
	}
	return flushDecodedSymbols(bits, cfg.BitsPerSymbol, consumedBits, cfg.OnDecoded)
}

func processSampleBuffer(samples []float32, demod chain.Processor[byte, float32], ctx context.Context, cfg demodProcessConfig, bits *buffer.Bits, consumedBits *int) error {
	if len(samples)%cfg.Channels != 0 {
		return fmt.Errorf("sample buffer of %d values is not divisible by channel count %d", len(samples), cfg.Channels)
	}
	chunks := splitIntoChunks(samples, cfg.Channels, cfg.ChunkParts)
	for _, chunk := range chunks {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if len(chunk) == 0 {
			continue
		}
		reader := buffer.New(chunk)
		demod.Process(bits, reader)
		if err := flushDecodedSymbols(bits, cfg.BitsPerSymbol, consumedBits, cfg.OnDecoded); err != nil {
			return err
		}
	}
	return nil
}

func validateChannelCount(shape shape.Shape, expected int) error {
	actual := 1
	if len(shape) >= 2 {
		actual = shape[len(shape)-1]
	}
	if actual != expected {
		return fmt.Errorf("expected %d channels but input provides %d", expected, actual)
	}
	return nil
}

func flushDecodedSymbols(bits *buffer.Bits, bitsPerSymbol int, consumedBits *int, onDecoded func([]byte) error) error {
	totalBits := bits.Bits()
	if totalBits <= *consumedBits {
		return nil
	}
	newBits := totalBits - *consumedBits
	symbols := newBits / bitsPerSymbol
	if symbols == 0 {
		return nil
	}
	if err := bits.SetPosition(*consumedBits); err != nil {
		return err
	}
	symbolBuf := make([]byte, symbols)
	readBits, err := bits.Read(symbolBuf)
	*consumedBits += readBits
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if err := bits.SetPosition(totalBits); err != nil {
		return err
	}
	payload := symbolsToBytes(symbolBuf, bitsPerSymbol)
	if len(payload) == 0 {
		return nil
	}
	return onDecoded(payload)
}

func splitIntoChunks(samples []float32, channels, parts int) [][]float32 {
	if parts <= 1 || len(samples) == 0 {
		return [][]float32{samples}
	}
	totalFrames := len(samples) / channels
	if totalFrames == 0 {
		return [][]float32{samples}
	}
	chunkFrames := totalFrames / parts
	if chunkFrames == 0 {
		chunkFrames = 1
	}
	chunks := make([][]float32, 0, parts)
	for frame := 0; frame < totalFrames; {
		endFrame := frame + chunkFrames
		if len(chunks) == parts-1 || endFrame > totalFrames {
			endFrame = totalFrames
		}
		start := frame * channels
		end := endFrame * channels
		chunks = append(chunks, samples[start:end])
		frame = endFrame
	}
	return chunks
}

func bytesToSymbols(data []byte, bitsPerSymbol int) []byte {
	if bitsPerSymbol <= 0 {
		return nil
	}
	mask := byte((1 << bitsPerSymbol) - 1)
	acc := uint32(0)
	accBits := 0
	var symbols []byte
	for _, b := range data {
		acc = (acc << 8) | uint32(b)
		accBits += 8
		for accBits >= bitsPerSymbol {
			shift := accBits - bitsPerSymbol
			symbols = append(symbols, byte((acc>>shift)&uint32(mask)))
			accBits -= bitsPerSymbol
			acc &= (1 << shift) - 1
		}
	}
	if accBits > 0 {
		symbols = append(symbols, byte((acc<<(bitsPerSymbol-accBits))&uint32(mask)))
	}
	return symbols
}

func symbolsToBytes(symbols []byte, bitsPerSymbol int) []byte {
	if bitsPerSymbol <= 0 {
		return nil
	}
	acc := uint32(0)
	accBits := 0
	result := make([]byte, 0, (len(symbols)*bitsPerSymbol+7)/8)
	for _, symbol := range symbols {
		acc = (acc << bitsPerSymbol) | uint32(symbol)
		accBits += bitsPerSymbol
		for accBits >= 8 {
			shift := accBits - 8
			result = append(result, byte((acc>>shift)&0xFF))
			accBits -= 8
			acc &= (1 << shift) - 1
		}
	}
	if accBits > 0 {
		result = append(result, byte(acc<<(8-accBits)))
	}
	return result
}

func modulateSamples(mod chain.Processor[float32, byte], bitsPerSymbol int, payload []byte) ([]float32, error) {
	if bitsPerSymbol <= 0 {
		return nil, fmt.Errorf("bits-per-symbol must be positive")
	}
	symbols := bytesToSymbols(payload, bitsPerSymbol)
	inputBits, err := buffer.NewBits(bitsPerSymbol)
	if err != nil {
		return nil, err
	}
	if _, err := inputBits.Write(symbols); err != nil {
		return nil, err
	}

	estimate := mod.Process(nil, inputBits)
	if estimate <= 0 {
		return nil, fmt.Errorf("modulator produced no samples (estimate=%d)", estimate)
	}

	mod.Reset()
	inputBits.Reset()

	sampleBuf := make([]float32, 0, estimate)
	writer := buffer.New(sampleBuf)
	written := mod.Process(writer, inputBits)
	if written <= 0 {
		return nil, fmt.Errorf("modulator produced 0 samples during processing")
	}

	return writer.Items(), nil
}

// createEncodedModulator creates a chain.Adapter that composes an encoder with a modulator.
// Returns the adapter and the encoder type name for logging.
func createEncodedModulator(mod chain.Processor[float32, byte], encoding string, rsN, rsK int) (chain.Processor[float32, byte], string, error) {
	if encoding == "none" {
		return mod, "none", nil
	}

	var encoder chain.Processor[byte, byte]
	encoderName := encoding

	switch encoding {
	case "gray":
		encoder = gray.NewEncoderWithBits(8)
	case "hamming":
		encoder = hamming.NewEncoder()
	case "rs":
		if rsN <= rsK {
			return nil, "", fmt.Errorf("RS encoder: n (%d) must be greater than k (%d)", rsN, rsK)
		}
		if rsK <= 0 {
			return nil, "", fmt.Errorf("RS encoder: k (%d) must be positive", rsK)
		}
		if rsN > 255 {
			return nil, "", fmt.Errorf("RS encoder: n (%d) must be <= 255 for byte-level encoding", rsN)
		}
		encoder = rs.NewEncoder(rsN, rsK)
		encoderName = fmt.Sprintf("rs(n=%d,k=%d)", rsN, rsK)
	default:
		return nil, "", fmt.Errorf("invalid encoding: %s (must be none, gray, hamming, or rs)", encoding)
	}

	// Create adapter: data (byte) -> encoder (byte->byte) -> modulator (byte->float32)
	bufferFactory := func(cap int) *buffer.Values[byte] {
		return buffer.New(make([]byte, 0, cap))
	}
	adapter := chain.NewAdapter[float32, byte, byte, *buffer.Values[byte]](bufferFactory, mod, encoder)

	return &adapter, encoderName, nil
}

// createDecodedDemodulator creates a chain.Adapter that composes a demodulator with a decoder.
// Returns the adapter and the decoder type name for logging.
func createDecodedDemodulator(demod chain.Processor[byte, float32], decoding string, rsN, rsK int) (chain.Processor[byte, float32], string, error) {
	if decoding == "none" {
		return demod, "none", nil
	}

	var decoder chain.Processor[byte, byte]
	decoderName := decoding

	switch decoding {
	case "gray":
		decoder = gray.NewDecoderWithBits(8)
	case "hamming":
		decoder = hamming.NewDecoder()
	case "rs":
		if rsN <= rsK {
			return nil, "", fmt.Errorf("RS decoder: n (%d) must be greater than k (%d)", rsN, rsK)
		}
		if rsK <= 0 {
			return nil, "", fmt.Errorf("RS decoder: k (%d) must be positive", rsK)
		}
		if rsN > 255 {
			return nil, "", fmt.Errorf("RS decoder: n (%d) must be <= 255 for byte-level decoding", rsN)
		}
		decoder = rs.NewDecoder(rsN, rsK)
		decoderName = fmt.Sprintf("rs(n=%d,k=%d)", rsN, rsK)
	default:
		return nil, "", fmt.Errorf("invalid decoding: %s (must be none, gray, hamming, or rs)", decoding)
	}

	// Create adapter: demodulator (float32->byte) -> decoder (byte->byte) -> data (byte)
	bufferFactory := func(cap int) *buffer.Values[byte] {
		return buffer.New(make([]byte, 0, cap))
	}
	adapter := chain.NewAdapter[byte, byte, float32, *buffer.Values[byte]](bufferFactory, decoder, demod)

	return &adapter, decoderName, nil
}
