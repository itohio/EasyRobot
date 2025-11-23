package audio

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testSampleRate = 22000
	testDuration   = 5    // seconds
	testFrequency  = 1000 // Hz
)

func TestUnmarshaller_ReadSingleWAV_Mono_INT16(t *testing.T) {
	path := filepath.Join("testdata", "mono_int16.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	u := NewUnmarshaller(WithWAVPath(path))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, testSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())
	assert.Equal(t, types.INT16, u.DataType())

	// Verify tensor shape
	shape := result.Shape()
	assert.Equal(t, 2, len(shape))
	expectedFrames := testSampleRate * testDuration
	assert.Equal(t, expectedFrames, shape[0])
	assert.Equal(t, 1, shape[1])

	// Verify tensor dtype
	assert.Equal(t, tensortypes.INT16, result.DataType())

	// Verify data is not empty
	data := result.Data()
	assert.NotNil(t, data)
	dataSlice, ok := data.([]int16)
	require.True(t, ok, "data should be []int16")
	assert.Greater(t, len(dataSlice), 0)
}

func TestUnmarshaller_ReadSingleWAV_Mono_INT32(t *testing.T) {
	path := filepath.Join("testdata", "mono_int32.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	u := NewUnmarshaller(WithWAVPath(path))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, testSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())
	assert.Equal(t, types.INT32, u.DataType())

	// Verify tensor dtype
	assert.Equal(t, tensortypes.INT32, result.DataType())

	// Verify data
	data := result.Data()
	assert.NotNil(t, data)
	dataSlice, ok := data.([]int32)
	require.True(t, ok, "data should be []int32")
	assert.Greater(t, len(dataSlice), 0)
}

func TestUnmarshaller_ReadSingleWAV_Mono_FP32(t *testing.T) {
	path := filepath.Join("testdata", "mono_fp32.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	u := NewUnmarshaller(WithWAVPath(path))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, testSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())

	// Verify tensor dtype (may be INT32 if WAV is PCM, or FP32 if float)
	dtype := result.DataType()
	assert.True(t, dtype == tensortypes.INT32 || dtype == tensortypes.FP32, "dtype should be INT32 or FP32")

	// Verify data
	data := result.Data()
	assert.NotNil(t, data)
}

func TestUnmarshaller_ReadSingleWAV_Mono_UINT8(t *testing.T) {
	path := filepath.Join("testdata", "mono_uint8.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	u := NewUnmarshaller(WithWAVPath(path))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, testSampleRate, u.SampleRate())
	assert.Equal(t, 1, u.Channels())

	// Verify tensor dtype
	assert.Equal(t, tensortypes.UINT8, result.DataType())

	// Verify data
	data := result.Data()
	assert.NotNil(t, data)
	dataSlice, ok := data.([]uint8)
	require.True(t, ok, "data should be []uint8")
	assert.Greater(t, len(dataSlice), 0)
}

func TestUnmarshaller_ReadSingleWAV_Stereo_INT16(t *testing.T) {
	path := filepath.Join("testdata", "stereo_int16.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	u := NewUnmarshaller(WithWAVPath(path))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, testSampleRate, u.SampleRate())
	assert.Equal(t, 2, u.Channels())
	assert.Equal(t, types.INT16, u.DataType())

	// Verify tensor shape
	shape := result.Shape()
	assert.Equal(t, 2, len(shape))
	expectedFrames := testSampleRate * testDuration
	assert.Equal(t, expectedFrames, shape[0])
	assert.Equal(t, 2, shape[1])

	// Verify tensor dtype
	assert.Equal(t, tensortypes.INT16, result.DataType())

	// Verify data
	data := result.Data()
	assert.NotNil(t, data)
	dataSlice, ok := data.([]int16)
	require.True(t, ok, "data should be []int16")
	assert.Equal(t, expectedFrames*2, len(dataSlice))
}

func TestUnmarshaller_ReadSingleWAV_Stereo_INT32(t *testing.T) {
	path := filepath.Join("testdata", "stereo_int32.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	u := NewUnmarshaller(WithWAVPath(path))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify format
	assert.Equal(t, testSampleRate, u.SampleRate())
	assert.Equal(t, 2, u.Channels())
	assert.Equal(t, types.INT32, u.DataType())

	// Verify tensor shape
	shape := result.Shape()
	assert.Equal(t, 2, len(shape))
	expectedFrames := testSampleRate * testDuration
	assert.Equal(t, expectedFrames, shape[0])
	assert.Equal(t, 2, shape[1])
}

func TestUnmarshaller_ReadSingleWAV_PhaseShifted(t *testing.T) {
	path0 := filepath.Join("testdata", "mono_int16_phase0.wav")
	path10 := filepath.Join("testdata", "mono_int16_phase10.wav")

	if _, err := os.Stat(path0); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path0)
	}
	if _, err := os.Stat(path10); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path10)
	}

	// Read both files
	u0 := NewUnmarshaller(WithWAVPath(path0))
	var result0 types.Tensor
	err := u0.Unmarshal(nil, &result0)
	require.NoError(t, err)

	u10 := NewUnmarshaller(WithWAVPath(path10))
	var result10 types.Tensor
	err = u10.Unmarshal(nil, &result10)
	require.NoError(t, err)

	// Verify both have same format
	assert.Equal(t, u0.SampleRate(), u10.SampleRate())
	assert.Equal(t, u0.Channels(), u10.Channels())
	assert.Equal(t, u0.DataType(), u10.DataType())

	// Verify shapes match
	assert.Equal(t, result0.Shape(), result10.Shape())

	// Verify data is different (phase shift)
	data0 := result0.Data().([]int16)
	data10 := result10.Data().([]int16)

	// They should be different (not all zeros)
	allSame := true
	for i := 0; i < len(data0) && i < len(data10); i++ {
		if data0[i] != data10[i] {
			allSame = false
			break
		}
	}
	assert.False(t, allSame, "Phase-shifted files should have different data")
}

func TestUnmarshaller_ReadSingleWAV_WithSampleRateValidation(t *testing.T) {
	path := filepath.Join("testdata", "mono_int16.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	// Should fail with wrong sample rate
	u := NewUnmarshaller(
		WithWAVPath(path),
		WithSampleRate(44100), // Different from file's 22000
	)
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sample rate mismatch")
}

func TestUnmarshaller_ReadSingleWAV_WithChannelValidation(t *testing.T) {
	path := filepath.Join("testdata", "stereo_int16.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	// Should fail with wrong channel count
	u := NewUnmarshaller(
		WithWAVPath(path),
		WithChannels(1), // File has 2 channels
	)
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "channel count mismatch")
}

func TestUnmarshaller_ReadSingleWAV_WithDestinationType(t *testing.T) {
	path := filepath.Join("testdata", "mono_int16.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
	}

	// Force conversion to FP32
	u := NewUnmarshaller(
		WithWAVPath(path),
		types.WithDestinationType(types.FP32),
	)
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.NoError(t, err)

	// Verify tensor is FP32 even though file is INT16
	assert.Equal(t, tensortypes.FP32, result.DataType())

	// Verify data is float32
	data := result.Data()
	_, ok := data.([]float32)
	assert.True(t, ok, "data should be []float32 when destination type is FP32")
}

func TestUnmarshaller_ReadSingleWAV_NonExistentFile(t *testing.T) {
	u := NewUnmarshaller(WithWAVPath("nonexistent.wav"))
	var result types.Tensor
	err := u.Unmarshal(nil, &result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to open WAV file")
}

func TestUnmarshaller_ReadSingleWAV_InvalidWAV(t *testing.T) {
	// Create a temporary invalid file
	tmpFile, err := os.CreateTemp("", "invalid_*.wav")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	tmpFile.WriteString("not a wav file")
	tmpFile.Close()

	u := NewUnmarshaller(WithWAVPath(tmpFile.Name()))
	var result types.Tensor
	err = u.Unmarshal(nil, &result)
	require.Error(t, err)
}

func TestUnmarshaller_Format(t *testing.T) {
	u := NewUnmarshaller()
	assert.Equal(t, "audio", u.Format())
}

func TestUnmarshaller_SampleRate_Channels_DataType_BeforeUnmarshal(t *testing.T) {
	u := NewUnmarshaller()

	// Before unmarshalling, should return zero/unknown values
	assert.Equal(t, 0, u.SampleRate())
	assert.Equal(t, 0, u.Channels())
	assert.Equal(t, types.DT_UNKNOWN, u.DataType())
}

// TestUnmarshaller_ReadSingleWAV_FrequencyVerification verifies that the WAV files contain 1kHz signals
func TestUnmarshaller_ReadSingleWAV_FrequencyVerification(t *testing.T) {
	testCases := []struct {
		name     string
		filePath string
		channels int
	}{
		{"mono_int16", "mono_int16.wav", 1},
		{"mono_int32", "mono_int32.wav", 1},
		{"mono_fp32", "mono_fp32.wav", 1},
		{"mono_uint8", "mono_uint8.wav", 1},
		{"stereo_int16", "stereo_int16.wav", 2},
		{"stereo_int32", "stereo_int32.wav", 2},
	}

	targetFreq := float32(testFrequency)  // 1kHz
	sampleRate := float32(testSampleRate) // 22kHz

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join("testdata", tc.filePath)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path)
			}

			// Read WAV file
			u := NewUnmarshaller(WithWAVPath(path))
			var result types.Tensor
			err := u.Unmarshal(nil, &result)
			require.NoError(t, err)
			require.NotNil(t, result)

			// Convert tensor data to float32 vector for frequency analysis
			shape := result.Shape()
			numFrames := shape[0]
			numChannels := shape[1]

			// Extract first channel for analysis
			var signalData []float32
			switch result.DataType() {
			case tensortypes.INT16:
				data := result.Data().([]int16)
				signalData = make([]float32, numFrames)
				for i := 0; i < numFrames; i++ {
					// Normalize to [-1.0, 1.0] range
					signalData[i] = float32(data[i*numChannels]) / 32768.0
				}
			case tensortypes.INT32:
				data := result.Data().([]int32)
				signalData = make([]float32, numFrames)
				for i := 0; i < numFrames; i++ {
					// Normalize to [-1.0, 1.0] range
					signalData[i] = float32(data[i*numChannels]) / 2147483648.0
				}
			case tensortypes.FP32:
				data := result.Data().([]float32)
				signalData = make([]float32, numFrames)
				for i := 0; i < numFrames; i++ {
					signalData[i] = data[i*numChannels]
				}
			case tensortypes.UINT8:
				data := result.Data().([]uint8)
				signalData = make([]float32, numFrames)
				for i := 0; i < numFrames; i++ {
					// Convert from [0, 255] to [-1.0, 1.0]
					signalData[i] = (float32(data[i*numChannels]) - 128.0) / 128.0
				}
			default:
				t.Fatalf("Unsupported data type: %v", result.DataType())
			}

			// Create vector from signal data
			signal := vec.NewFrom(signalData...)

			// Use Goertzel algorithm to detect 1kHz frequency
			freqMeas := dsp.Goertzel(signal, targetFreq, sampleRate)

			// Verify that the amplitude at 1kHz is significant
			// For a pure sine wave, we expect high amplitude
			assert.Greater(t, freqMeas.Amplitude, float32(0.1),
				"1kHz signal should have significant amplitude (got %.4f)", freqMeas.Amplitude)

			// Verify frequency is approximately 1kHz (within 1% tolerance)
			assert.InDelta(t, float64(targetFreq), float64(freqMeas.Frequency), float64(targetFreq*0.01),
				"Detected frequency should be approximately 1kHz")

			// For stereo files, also check second channel
			if numChannels > 1 {
				var signalData2 []float32
				switch result.DataType() {
				case tensortypes.INT16:
					data := result.Data().([]int16)
					signalData2 = make([]float32, numFrames)
					for i := 0; i < numFrames; i++ {
						signalData2[i] = float32(data[i*numChannels+1]) / 32768.0
					}
				case tensortypes.INT32:
					data := result.Data().([]int32)
					signalData2 = make([]float32, numFrames)
					for i := 0; i < numFrames; i++ {
						signalData2[i] = float32(data[i*numChannels+1]) / 2147483648.0
					}
				case tensortypes.FP32:
					data := result.Data().([]float32)
					signalData2 = make([]float32, numFrames)
					for i := 0; i < numFrames; i++ {
						signalData2[i] = data[i*numChannels+1]
					}
				case tensortypes.UINT8:
					data := result.Data().([]uint8)
					signalData2 = make([]float32, numFrames)
					for i := 0; i < numFrames; i++ {
						signalData2[i] = (float32(data[i*numChannels+1]) - 128.0) / 128.0
					}
				}

				signal2 := vec.NewFrom(signalData2...)
				freqMeas2 := dsp.Goertzel(signal2, targetFreq, sampleRate)
				assert.Greater(t, freqMeas2.Amplitude, float32(0.1),
					"Second channel should also have 1kHz signal")
			}
		})
	}
}

// TestUnmarshaller_ReadSingleWAV_PhaseShiftVerification verifies phase shift between two files
func TestUnmarshaller_ReadSingleWAV_PhaseShiftVerification(t *testing.T) {
	path0 := filepath.Join("testdata", "mono_int16_phase0.wav")
	path10 := filepath.Join("testdata", "mono_int16_phase10.wav")

	if _, err := os.Stat(path0); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path0)
	}
	if _, err := os.Stat(path10); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s (run tmp/generate_wav_testdata.go first)", path10)
	}

	targetFreq := float32(testFrequency)
	sampleRate := float32(testSampleRate)

	// Read both files
	u0 := NewUnmarshaller(WithWAVPath(path0))
	var result0 types.Tensor
	err := u0.Unmarshal(nil, &result0)
	require.NoError(t, err)

	u10 := NewUnmarshaller(WithWAVPath(path10))
	var result10 types.Tensor
	err = u10.Unmarshal(nil, &result10)
	require.NoError(t, err)

	// Convert to float32 vectors
	data0 := result0.Data().([]int16)
	data10 := result10.Data().([]int16)
	numFrames := result0.Shape()[0]

	signal0 := make([]float32, numFrames)
	signal10 := make([]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		signal0[i] = float32(data0[i]) / 32768.0
		signal10[i] = float32(data10[i]) / 32768.0
	}

	vec0 := vec.NewFrom(signal0...)
	vec10 := vec.NewFrom(signal10...)

	// Measure frequency and phase for both signals
	freqMeas0 := dsp.Goertzel(vec0, targetFreq, sampleRate)
	freqMeas10 := dsp.Goertzel(vec10, targetFreq, sampleRate)

	// Both should have 1kHz signal
	assert.Greater(t, freqMeas0.Amplitude, float32(0.1), "Phase 0 signal should have 1kHz")
	assert.Greater(t, freqMeas10.Amplitude, float32(0.1), "Phase 10 signal should have 1kHz")

	// Calculate phase difference (should be approximately 10 degrees = π/18 radians)
	phaseDiff := freqMeas10.Phase - freqMeas0.Phase
	expectedPhaseDiff := float32(10.0 * 3.14159265359 / 180.0) // 10 degrees in radians

	// Allow some tolerance (within 2 degrees)
	tolerance := float32(2.0 * 3.14159265359 / 180.0)
	assert.InDelta(t, float64(expectedPhaseDiff), float64(phaseDiff), float64(tolerance),
		"Phase difference should be approximately 10 degrees (got %.4f radians, expected %.4f)",
		phaseDiff, expectedPhaseDiff)
}
