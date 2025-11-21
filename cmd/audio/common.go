package main

import (
	"encoding/hex"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
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
