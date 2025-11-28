package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/tc"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenTCCmd() *cobra.Command {
	var (
		output     string
		deviceID   string
		amplitude  string
		frameRate  string
		timecode   string
		sampleRate int
		dataType   string
	)

	cmd := &cobra.Command{
		Use:   "gen-tc",
		Short: "Generate timecode signal",
		Long:  "Generate a SMPTE timecode signal (LTC - Linear Timecode) from timecode string.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenTC(output, deviceID, amplitude, frameRate, timecode, sampleRate, dataType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (or use --device-id for audio device)")
	cmd.Flags().StringVar(&deviceID, "device-id", "", "Playback device ID (alternative to --output)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().StringVar(&frameRate, "frame-rate", "30", "Frame rate: \"24\", \"25\", \"29.97\", or \"30\" (default: 30)")
	cmd.Flags().StringVar(&timecode, "timecode", "00:00:00:00", "Timecode in HH:MM:SS:FF format (default: 00:00:00:00)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")

	return cmd
}

func runGenTC(output, deviceID, amplitude, frameRate, timecode string, sampleRate int, dataType string) error {
	// Validate output
	if output == "" && deviceID == "" {
		return fmt.Errorf("either --output or --device-id must be specified")
	}

	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Parse frame rate
	var tcFrameRate tc.FrameRate
	switch frameRate {
	case "24":
		tcFrameRate = tc.FrameRate24
	case "25":
		tcFrameRate = tc.FrameRate25
	case "29.97":
		tcFrameRate = tc.FrameRate2997
	case "30":
		tcFrameRate = tc.FrameRate30
	default:
		return fmt.Errorf("invalid frame rate: %s (use 24, 25, 29.97, or 30)", frameRate)
	}

	// Parse timecode
	parts := strings.Split(timecode, ":")
	if len(parts) != 4 {
		return fmt.Errorf("invalid timecode format: %s (expected HH:MM:SS:FF)", timecode)
	}

	hours, err := strconv.Atoi(parts[0])
	if err != nil || hours < 0 || hours > 23 {
		return fmt.Errorf("invalid hours: %s", parts[0])
	}

	minutes, err := strconv.Atoi(parts[1])
	if err != nil || minutes < 0 || minutes > 59 {
		return fmt.Errorf("invalid minutes: %s", parts[1])
	}

	seconds, err := strconv.Atoi(parts[2])
	if err != nil || seconds < 0 || seconds > 59 {
		return fmt.Errorf("invalid seconds: %s", parts[2])
	}

	frames, err := strconv.Atoi(parts[3])
	maxFrames := 29
	if tcFrameRate == tc.FrameRate24 {
		maxFrames = 23
	} else if tcFrameRate == tc.FrameRate25 {
		maxFrames = 24
	}
	if err != nil || frames < 0 || frames > maxFrames {
		return fmt.Errorf("invalid frames: %s (max %d for frame rate %s)", parts[3], maxFrames, frameRate)
	}

	// Create timecode structure
	smpteTC := &tc.SMPTETimeCode{
		Hours:     hours,
		Minutes:   minutes,
		Seconds:   seconds,
		Frames:    frames,
		FrameRate: tcFrameRate,
		DropFrame: tcFrameRate == tc.FrameRate2997,
	}

	// Create LTC modulator
	ltcModulator := tc.NewLTCModulator(float32(sampleRate), tcFrameRate)

	// Estimate length needed (one frame period)
	expectedSamples := ltcModulator.Encode(nil, smpteTC)
	if expectedSamples == 0 {
		return fmt.Errorf("failed to estimate signal length")
	}

	// Generate signal
	modulatedVec := vec.New(expectedSamples)
	samplesWritten := ltcModulator.Encode(modulatedVec, smpteTC)
	if samplesWritten == 0 {
		return fmt.Errorf("failed to generate signal")
	}

	modData := []float32(modulatedVec)[:samplesWritten]

	// Apply amplitude scaling
	for i := range modData {
		modData[i] *= ampLinear
	}

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	numSamples := len(modData)

	// Create tensor [frames, 1]
	shape := tensor.NewShape(numSamples, 1)
	t := tensor.New(tensortypes.DataType(dt), shape)

	// Fill tensor
	switch dt {
	case types.INT16:
		data := t.Data().([]int16)
		for i := range numSamples {
			val := modData[i] * 32767.0
			if val > 32767.0 {
				val = 32767.0
			} else if val < -32768.0 {
				val = -32768.0
			}
			data[i] = int16(val)
		}
	case types.INT32:
		data := t.Data().([]int32)
		for i := range numSamples {
			val := modData[i] * 2147483647.0
			if val > 2147483647.0 {
				val = 2147483647.0
			} else if val < -2147483648.0 {
				val = -2147483648.0
			}
			data[i] = int32(val)
		}
	case types.FP32:
		data := t.Data().([]float32)
		copy(data, modData)
	case types.UINT8:
		data := t.Data().([]uint8)
		for i := range numSamples {
			val := (modData[i] + 1.0) * 127.5
			if val > 255.0 {
				val = 255.0
			} else if val < 0.0 {
				val = 0.0
			}
			data[i] = uint8(val)
		}
	default:
		return fmt.Errorf("unsupported data type: %v", dt)
	}

	// Write to WAV file or device
	if output != "" {
		file, err := os.Create(output)
		if err != nil {
			return fmt.Errorf("failed to create output file: %w", err)
		}
		defer file.Close()

		m := audio.NewMarshaller(
			audio.WithWAVPath(output),
			audio.WithSampleRate(sampleRate),
			audio.WithChannels(1),
			types.WithDestinationType(dt),
		)

		if err := m.Marshal(file, t); err != nil {
			return fmt.Errorf("failed to write WAV file: %w", err)
		}

		fmt.Printf("Generated timecode signal: %s\n", output)
	} else {
		// TODO: Implement device playback
		return fmt.Errorf("device playback not yet implemented (use --output to save to WAV file)")
	}

	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Frame rate: %s fps\n", frameRate)
	fmt.Printf("  Timecode: %s\n", timecode)

	return nil
}
