package audio

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	marshallerTestSampleRate = 48000
	marshallerTestDuration   = 0.1 // seconds (shorter for faster tests)
	marshallerTestFrequency  = 1000 // Hz
)

// generateMarshallerTestSignal creates a test sine wave signal
func generateMarshallerTestSignal(numSamples int, frequency, sampleRate float32) []float32 {
	signal := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		t := float32(i) / sampleRate
		signal[i] = float32(math.Sin(2 * math.Pi * float64(frequency) * float64(t)))
	}
	return signal
}

func TestMarshaller_WriteReadRoundtrip_INT16(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "test_int16.wav")

	numSamples := int(marshallerTestSampleRate * marshallerTestDuration)
	testSignal := generateMarshallerTestSignal(numSamples, marshallerTestFrequency, marshallerTestSampleRate)

	// Create tensor with INT16 data
	shape := tensor.NewShape(numSamples, 1)
	tensorData := tensor.New(tensortypes.INT16, shape)
	data := tensorData.Data().([]int16)
	for i, v := range testSignal {
		val := v * 32767.0
		if val > 32767.0 {
			val = 32767.0
		} else if val < -32768.0 {
			val = -32768.0
		}
		data[i] = int16(val)
	}

	// Write to WAV
	m := NewMarshaller(
		WithWAVPath(wavPath),
		WithSampleRate(marshallerTestSampleRate),
		WithChannels(1),
	)
	err := m.Marshal(nil, tensorData)
	require.NoError(t, err)

	// Read back from WAV
	u := NewUnmarshaller(WithWAVPath(wavPath))
	var result types.Tensor
	err = u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, marshallerTestSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())
	assert.Equal(t, types.INT16, u.DataType())

	// Verify tensor shape
	resultShape := result.Shape()
	assert.Equal(t, 2, len(resultShape))
	assert.Equal(t, numSamples, resultShape[0])
	assert.Equal(t, 1, resultShape[1])

	// Verify tensor dtype
	assert.Equal(t, tensortypes.INT16, result.DataType())

	// Verify data matches (within quantization tolerance)
	resultData := result.Data().([]int16)
	require.Equal(t, numSamples, len(resultData))
	
	// Check that data is not all zeros
	allZeros := true
	for _, v := range resultData {
		if v != 0 {
			allZeros = false
			break
		}
	}
	assert.False(t, allZeros, "Read data should not be all zeros")

	// Verify first few samples match (within quantization tolerance)
	// Due to quantization, exact match may not be possible, but should be close
	matches := 0
	for i := 0; i < numSamples && i < 100; i++ {
		// Allow small difference due to quantization
		diff := int32(resultData[i]) - int32(data[i])
		if diff == 0 || (diff >= -1 && diff <= 1) {
			matches++
		}
	}
	// At least 80% should match exactly or within 1 LSB
	assert.Greater(t, matches, 80, "At least 80%% of samples should match within quantization tolerance")
}

func TestMarshaller_WriteReadRoundtrip_INT32(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "test_int32.wav")

	numSamples := int(marshallerTestSampleRate * marshallerTestDuration)
	testSignal := generateMarshallerTestSignal(numSamples, marshallerTestFrequency, marshallerTestSampleRate)

	// Create tensor with INT32 data
	shape := tensor.NewShape(numSamples, 1)
	tensorData := tensor.New(tensortypes.INT32, shape)
	data := tensorData.Data().([]int32)
	for i, v := range testSignal {
		val := v * 2147483647.0
		if val > 2147483647.0 {
			val = 2147483647.0
		} else if val < -2147483648.0 {
			val = -2147483648.0
		}
		data[i] = int32(val)
	}

	// Write to WAV
	m := NewMarshaller(
		WithWAVPath(wavPath),
		WithSampleRate(marshallerTestSampleRate),
		WithChannels(1),
	)
	err := m.Marshal(nil, tensorData)
	require.NoError(t, err)

	// Read back from WAV
	u := NewUnmarshaller(WithWAVPath(wavPath))
	var result types.Tensor
	err = u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, marshallerTestSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())
	assert.Equal(t, types.INT32, u.DataType())

	// Verify tensor shape
	resultShape := result.Shape()
	assert.Equal(t, 2, len(resultShape))
	assert.Equal(t, numSamples, resultShape[0])
	assert.Equal(t, 1, resultShape[1])

	// Verify tensor dtype
	assert.Equal(t, tensortypes.INT32, result.DataType())

	// Verify data matches
	resultData := result.Data().([]int32)
	require.Equal(t, numSamples, len(resultData))
	
	// Check that data is not all zeros
	allZeros := true
	for _, v := range resultData {
		if v != 0 {
			allZeros = false
			break
		}
	}
	assert.False(t, allZeros, "Read data should not be all zeros")

	// Verify samples match (INT32 has high precision)
	matches := 0
	for i := 0; i < numSamples && i < 100; i++ {
		if resultData[i] == data[i] {
			matches++
		}
	}
	// Most samples should match exactly for INT32
	assert.Greater(t, matches, 90, "At least 90%% of samples should match exactly for INT32")
}

func TestMarshaller_WriteReadRoundtrip_FP32(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "test_fp32.wav")

	numSamples := int(marshallerTestSampleRate * marshallerTestDuration)
	testSignal := generateMarshallerTestSignal(numSamples, marshallerTestFrequency, marshallerTestSampleRate)

	// Create tensor with FP32 data
	shape := tensor.NewShape(numSamples, 1)
	tensorData := tensor.New(tensortypes.FP32, shape)
	data := tensorData.Data().([]float32)
	copy(data, testSignal)

	// Write to WAV
	m := NewMarshaller(
		WithWAVPath(wavPath),
		WithSampleRate(marshallerTestSampleRate),
		WithChannels(1),
	)
	err := m.Marshal(nil, tensorData)
	require.NoError(t, err)

	// Read back from WAV
	u := NewUnmarshaller(WithWAVPath(wavPath))
	var result types.Tensor
	err = u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, marshallerTestSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())

	// Verify tensor shape
	resultShape := result.Shape()
	assert.Equal(t, 2, len(resultShape))
	assert.Equal(t, numSamples, resultShape[0])
	assert.Equal(t, 1, resultShape[1])

	// Verify data matches
	resultData := result.Data()
	require.NotNil(t, resultData)
	
	// Check that data is not all zeros
	var allZeros bool
	var resultFloat []float32
	switch d := resultData.(type) {
	case []float32:
		resultFloat = d
		allZeros = true
		for _, v := range d {
			if math.Abs(float64(v)) > 1e-6 {
				allZeros = false
				break
			}
		}
	case []int32:
		// WAV might convert FP32 to INT32 PCM
		allZeros = true
		for _, v := range d {
			if v != 0 {
				allZeros = false
				break
			}
		}
		resultFloat = make([]float32, len(d))
		for i, v := range d {
			resultFloat[i] = float32(v) / 2147483648.0
		}
	default:
		t.Fatalf("Unexpected data type: %T", resultData)
	}
	
	assert.False(t, allZeros, "Read data should not be all zeros")

	// Verify samples match (within float precision)
	matches := 0
	for i := 0; i < numSamples && i < 100; i++ {
		diff := math.Abs(float64(resultFloat[i] - data[i]))
		if diff < 0.01 { // Allow 1% tolerance
			matches++
		}
	}
	// Most samples should match within tolerance
	assert.Greater(t, matches, 80, "At least 80%% of samples should match within float precision tolerance")
}

func TestMarshaller_WriteReadRoundtrip_UINT8(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "test_uint8.wav")

	numSamples := int(marshallerTestSampleRate * marshallerTestDuration)
	testSignal := generateMarshallerTestSignal(numSamples, marshallerTestFrequency, marshallerTestSampleRate)

	// Create tensor with UINT8 data
	shape := tensor.NewShape(numSamples, 1)
	tensorData := tensor.New(tensortypes.UINT8, shape)
	data := tensorData.Data().([]uint8)
	for i, v := range testSignal {
		val := (v + 1.0) * 127.5
		if val > 255.0 {
			val = 255.0
		} else if val < 0.0 {
			val = 0.0
		}
		data[i] = uint8(val)
	}

	// Write to WAV
	m := NewMarshaller(
		WithWAVPath(wavPath),
		WithSampleRate(marshallerTestSampleRate),
		WithChannels(1),
	)
	err := m.Marshal(nil, tensorData)
	require.NoError(t, err)

	// Read back from WAV
	u := NewUnmarshaller(WithWAVPath(wavPath))
	var result types.Tensor
	err = u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, marshallerTestSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())
	assert.Equal(t, types.UINT8, u.DataType())

	// Verify tensor shape
	resultShape := result.Shape()
	assert.Equal(t, 2, len(resultShape))
	assert.Equal(t, numSamples, resultShape[0])
	assert.Equal(t, 1, resultShape[1])

	// Verify tensor dtype
	assert.Equal(t, tensortypes.UINT8, result.DataType())

	// Verify data matches
	resultData := result.Data().([]uint8)
	require.Equal(t, numSamples, len(resultData))
	
	// Check that data is not all zeros
	allZeros := true
	for _, v := range resultData {
		if v != 0 {
			allZeros = false
			break
		}
	}
	assert.False(t, allZeros, "Read data should not be all zeros")

	// Verify samples are reasonable (UINT8 has lower precision, so exact match is less important)
	// Just verify that the signal shape is preserved (not all zeros, has variation)
	hasVariation := false
	prevVal := resultData[0]
	for i := 1; i < numSamples && i < 100; i++ {
		if resultData[i] != prevVal {
			hasVariation = true
			break
		}
		prevVal = resultData[i]
	}
	assert.True(t, hasVariation, "UINT8 signal should have variation (not constant)")
}

func TestMarshaller_WriteReadRoundtrip_AllFormats(t *testing.T) {
	formats := []struct {
		name     string
		dtype    tensortypes.DataType
		fileType types.DataType
	}{
		{"INT16", tensortypes.INT16, types.INT16},
		{"INT32", tensortypes.INT32, types.INT32},
		{"FP32", tensortypes.FP32, types.FP32},
		{"UINT8", tensortypes.UINT8, types.UINT8},
	}

	for _, fmt := range formats {
		t.Run(fmt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			wavPath := filepath.Join(tmpDir, "test_"+fmt.name+".wav")

			numSamples := int(marshallerTestSampleRate * marshallerTestDuration)
			testSignal := generateMarshallerTestSignal(numSamples, marshallerTestFrequency, marshallerTestSampleRate)

			// Create tensor
			shape := tensor.NewShape(numSamples, 1)
			tensorData := tensor.New(fmt.dtype, shape)

			// Fill tensor based on type
			switch fmt.dtype {
			case tensortypes.INT16:
				data := tensorData.Data().([]int16)
				for i, v := range testSignal {
					val := v * 32767.0
					if val > 32767.0 {
						val = 32767.0
					} else if val < -32768.0 {
						val = -32768.0
					}
					data[i] = int16(val)
				}
			case tensortypes.INT32:
				data := tensorData.Data().([]int32)
				for i, v := range testSignal {
					val := v * 2147483647.0
					if val > 2147483647.0 {
						val = 2147483647.0
					} else if val < -2147483648.0 {
						val = -2147483648.0
					}
					data[i] = int32(val)
				}
			case tensortypes.FP32:
				data := tensorData.Data().([]float32)
				copy(data, testSignal)
			case tensortypes.UINT8:
				data := tensorData.Data().([]uint8)
				for i, v := range testSignal {
					val := (v + 1.0) * 127.5
					if val > 255.0 {
						val = 255.0
					} else if val < 0.0 {
						val = 0.0
					}
					data[i] = uint8(val)
				}
			}

			// Write to WAV
			m := NewMarshaller(
				WithWAVPath(wavPath),
				WithSampleRate(marshallerTestSampleRate),
				WithChannels(1),
			)
			err := m.Marshal(nil, tensorData)
			require.NoError(t, err)

			// Verify file exists
			_, err = os.Stat(wavPath)
			require.NoError(t, err, "WAV file should exist")

			// Read back from WAV
			u := NewUnmarshaller(WithWAVPath(wavPath))
			var result types.Tensor
			err = u.Unmarshal(nil, &result)
			require.NoError(t, err)
			require.NotNil(t, result)

			// Verify format
			assert.Equal(t, marshallerTestSampleRate, u.SampleRate())
			assert.Equal(t, 1, u.Channels())

			// Verify tensor shape
			resultShape := result.Shape()
			assert.Equal(t, 2, len(resultShape))
			assert.Equal(t, numSamples, resultShape[0])
			assert.Equal(t, 1, resultShape[1])

			// Verify data is not all zeros
			resultData := result.Data()
			require.NotNil(t, resultData)
			
			allZeros := true
			switch d := resultData.(type) {
			case []int16:
				for _, v := range d {
					if v != 0 {
						allZeros = false
						break
					}
				}
			case []int32:
				for _, v := range d {
					if v != 0 {
						allZeros = false
						break
					}
				}
			case []float32:
				for _, v := range d {
					if math.Abs(float64(v)) > 1e-6 {
						allZeros = false
						break
					}
				}
			case []uint8:
				for _, v := range d {
					if v != 0 {
						allZeros = false
						break
					}
				}
			}
			assert.False(t, allZeros, "Read data should not be all zeros for format %s", fmt.name)
		})
	}
}

