package main

import (
	"context"
	"encoding/hex"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
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
		sg := dsp.NewSignalGenerator1D()
		noiseVec := sg.NoiseWhite(padSamples, padAmp)
		padData = []float32(noiseVec.View().(vec.Vector))
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
	sg := dsp.NewSignalGenerator1D()

	noiseType = strings.ToLower(noiseType)
	var noiseVec vecTypes.Vector
	switch noiseType {
	case "pink":
		noiseVec = sg.NoisePink(length, noiseAmplitude)
	case "brown":
		noiseVec = sg.NoiseBrown(length, noiseAmplitude)
	default: // white
		noiseVec = sg.NoiseWhite(length, noiseAmplitude)
	}

	noiseData := []float32(noiseVec.View().(vec.Vector))

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

// extractFloat32FromTensor converts tensor data to float32 slice using tensor conversion facilities
// Returns the float32 slice and any error encountered
func extractFloat32FromTensor(tensor types.Tensor) ([]float32, error) {
	tensorData := tensor.Data()
	convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, tensorData)
	if convertedData == nil {
		return nil, fmt.Errorf("failed to convert tensor data to float32")
	}
	floatData, ok := convertedData.([]float32)
	if !ok {
		return nil, fmt.Errorf("converted data is not []float32: %T", convertedData)
	}
	return floatData, nil
}

// extractFloat32FromTensorData converts raw tensor data (from tensor.Data()) to float32 slice
// This is used for frame data from device capture
func extractFloat32FromTensorData(tensorData interface{}) ([]float32, error) {
	convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, tensorData)
	if convertedData == nil {
		return nil, fmt.Errorf("failed to convert tensor data to float32")
	}
	floatData, ok := convertedData.([]float32)
	if !ok {
		return nil, fmt.Errorf("converted data is not []float32: %T", convertedData)
	}
	return floatData, nil
}

// Fixed buffer size: 8k samples (mono)
const fixedBufferSize = 8192

// processAndWrite is a generic function that encodes data using a Modulator
// and writes it to a Marshaller using fixed-size buffers (8k samples).
// Works with both WAV files and audio devices through the marshaller abstraction.
func processAndWrite(marshaller types.Marshaller, modulator modulation.Modulator, data []byte, sampleRate int, dataType types.DataType, ctx context.Context) error {
	// Pre-allocate fixed buffer for encoded samples (8k samples, mono)
	amplitudeBuf := vec.New(fixedBufferSize)
	amplitudeVec := amplitudeBuf.View().(vec.Vector)

	// Reset modulator
	modulator.Reset()

	// Process data in chunks
	dataOffset := 0
	for dataOffset < len(data) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Determine how much data to encode
		remainingData := data[dataOffset:]

		// Encode into amplitude buffer (convert vec.Vector to vecTypes.Vector for interface)
		encodedLen := modulator.Encode(amplitudeVec, remainingData)
		if encodedLen == 0 {
			break // Encoding complete
		}

		// Update data offset - assume all remaining data was consumed
		// TODO: Modulator interface should return bytes consumed
		dataOffset += len(remainingData)

		// Extract float32 data from vector
		amplitudeData := []float32(amplitudeVec)

		// Create tensor from amplitude buffer
		shape := tensor.NewShape(encodedLen, 1) // [samples, channels=1]

		// Create tensor with FP32 (marshaller will handle conversion if needed)
		t := tensor.FromFloat32(shape, amplitudeData[:encodedLen])
		if t.Empty() {
			return fmt.Errorf("failed to create tensor")
		}

		// Write to marshaller
		// Note: sampleRate should be set on marshaller via options during construction
		if err := marshaller.Marshal(nil, t, types.WithContext(ctx)); err != nil {
			return fmt.Errorf("failed to write to marshaller: %w", err)
		}
	}

	return nil
}

// LockedDemodulator is an interface for demodulators that support lock status checking
type LockedDemodulator interface {
	modulation.Demodulator
	IsLocked() bool
}

// readAndProcess is a generic function that reads from an Unmarshaller
// and decodes data using a Demodulator, processing in fixed-size buffers (8k samples).
// Works with both WAV files and audio devices through the unmarshaller abstraction.
// onDecoded is called with decoded bytes. onFrame can optionally be called per frame for RMS statistics.
func readAndProcess(unmarshaller types.Unmarshaller, demodulator modulation.Demodulator, ctx context.Context, onDecoded func([]byte) error, onFrame func([]float32) error) error {
	// Pre-allocate fixed buffer for decoded bytes
	decodedBuf := make([]byte, fixedBufferSize*2) // Large enough for decoded bytes

	// Pre-allocate fixed buffer for float32 samples
	amplitudeBuf := make([]float32, fixedBufferSize)

	// Reset demodulator
	demodulator.Reset()

	// Try to create FrameStream first (for device capture)
	var stream types.FrameStream
	err := unmarshaller.Unmarshal(nil, &stream)
	if err == nil {
		// Successfully created stream - process frames from device
		defer stream.Close()
		return processStreamFrames(stream, demodulator, ctx, decodedBuf, amplitudeBuf, onDecoded, onFrame)
	}

	// If FrameStream failed, try reading as a single tensor (for WAV files)
	var tensor types.Tensor
	if err := unmarshaller.Unmarshal(nil, &tensor); err != nil {
		return fmt.Errorf("failed to read from unmarshaller (tried both stream and tensor): %w", err)
	}
	defer tensor.Release()

	// Extract float32 vector from tensor
	floatData, err := extractFloat32FromTensor(tensor)
	if err != nil {
		return fmt.Errorf("failed to extract float32 from tensor: %w", err)
	}

	// Call onFrame callback for RMS statistics if provided
	if onFrame != nil {
		if err := onFrame(floatData); err != nil {
			return err
		}
	}

	// Process entire tensor in chunks of fixed buffer size
	for offset := 0; offset < len(floatData); offset += fixedBufferSize {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		chunkSize := fixedBufferSize
		if offset+chunkSize > len(floatData) {
			chunkSize = len(floatData) - offset
		}

		// Copy chunk to amplitude buffer
		copy(amplitudeBuf[:chunkSize], floatData[offset:offset+chunkSize])

		// Create vector view of amplitude buffer
		amplitudeVec := vec.NewFrom(amplitudeBuf[:chunkSize]...)

		// Decode chunk
		decodedLen := demodulator.Decode(decodedBuf, amplitudeVec)
		if decodedLen > 0 {
			// Call callback with decoded data
			if err := onDecoded(decodedBuf[:decodedLen]); err != nil {
				return err
			}
		}
	}

	return nil
}

// processStreamFrames processes frames from a FrameStream (for device capture)
func processStreamFrames(stream types.FrameStream, demodulator modulation.Demodulator, ctx context.Context, decodedBuf []byte, amplitudeBuf []float32, onDecoded func([]byte) error, onFrame func([]float32) error) error {
	// Process frames from stream
	for frame := range stream.C {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if len(frame.Tensors) == 0 {
			continue
		}

		// Extract float32 vector from tensor
		t := frame.Tensors[0]
		floatData, err := extractFloat32FromTensor(t)
		if err != nil {
			continue // Skip frame if conversion fails
		}

		// Call onFrame callback for RMS statistics if provided
		if onFrame != nil {
			if err := onFrame(floatData); err != nil {
				return err
			}
		}

		// Process in chunks of fixed buffer size
		for offset := 0; offset < len(floatData); offset += fixedBufferSize {
			chunkSize := fixedBufferSize
			if offset+chunkSize > len(floatData) {
				chunkSize = len(floatData) - offset
			}

			// Copy chunk to amplitude buffer
			copy(amplitudeBuf[:chunkSize], floatData[offset:offset+chunkSize])

			// Create vector view of amplitude buffer
			amplitudeVec := vec.NewFrom(amplitudeBuf[:chunkSize]...)

			// Decode
			decodedLen := demodulator.Decode(decodedBuf, amplitudeVec)
			if decodedLen > 0 {
				// Call callback with decoded data
				if err := onDecoded(decodedBuf[:decodedLen]); err != nil {
					return err
				}
			}
		}

		// Release frame if needed
		if t != nil {
			t.Release()
		}
	}

	return nil
}
