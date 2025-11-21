package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/spf13/cobra"
)

func newGenNoiseCmd() *cobra.Command {
	var (
		output     string
		amplitude  string
		noiseType  string
		channels   int
		sampleRate int
		duration   float64
		dataType   string
	)

	cmd := &cobra.Command{
		Use:   "gen-noise",
		Short: "Generate noise signal(s)",
		Long:  "Generate white, pink, or brown noise and save to a WAV file.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenNoise(output, amplitude, noiseType, channels, sampleRate, duration, dataType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB per channel (comma-separated, default: -6dB)")
	cmd.Flags().StringVarP(&noiseType, "type", "t", "white", "Noise type (white, pink, brown, default: white)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 1, "Number of channels")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64VarP(&duration, "duration", "d", 5.0, "Duration in seconds")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenNoise(output, amplitude, noiseType string, channels, sampleRate int, duration float64, dataType string) error {
	// Parse amplitudes (dB)
	ampDbs, err := parseFloatSlice(amplitude)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	// Extend to channels if needed
	for len(ampDbs) < channels {
		if len(ampDbs) > 0 {
			ampDbs = append(ampDbs, ampDbs[len(ampDbs)-1])
		} else {
			ampDbs = append(ampDbs, -6.0)
		}
	}
	ampLinears := make([]float32, channels)
	for i := 0; i < channels; i++ {
		ampLinears[i] = dbToAmplitude(ampDbs[i])
	}

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	// Generate noise
	numSamples := int(float64(sampleRate) * duration)
	sg := dsp.NewSignalGenerator1D()

	// Generate per-channel noise
	channelSignals := make([]vecTypes.Vector, channels)
	noiseType = strings.ToLower(noiseType)
	for ch := 0; ch < channels; ch++ {
		switch noiseType {
		case "white":
			channelSignals[ch] = sg.NoiseWhite(numSamples, ampLinears[ch])
		case "pink":
			channelSignals[ch] = sg.NoisePink(numSamples, ampLinears[ch])
		case "brown":
			channelSignals[ch] = sg.NoiseBrown(numSamples, ampLinears[ch])
		default:
			return fmt.Errorf("unknown noise type: %s (supported: white, pink, brown)", noiseType)
		}
	}

	// Combine channels into tensor [frames, channels]
	shape := tensor.NewShape(numSamples, channels)
	t := tensor.New(tensor.DataType(dt), shape)

	// Fill tensor with channel data
	switch dt {
		case types.INT16:
		data := t.Data().([]int16)
		for i := 0; i < numSamples; i++ {
			for ch := 0; ch < channels; ch++ {
				sigData := channelSignals[ch].View().(vec.Vector)
				val := sigData[i] * 32767.0
				if val > 32767.0 {
					val = 32767.0
				} else if val < -32768.0 {
					val = -32768.0
				}
				data[i*channels+ch] = int16(val)
			}
		}
		case types.INT32:
		data := t.Data().([]int32)
		for i := 0; i < numSamples; i++ {
			for ch := 0; ch < channels; ch++ {
				sigData := channelSignals[ch].View().(vec.Vector)
				val := sigData[i] * 2147483647.0
				if val > 2147483647.0 {
					val = 2147483647.0
				} else if val < -2147483648.0 {
					val = -2147483648.0
				}
				data[i*channels+ch] = int32(val)
			}
		}
		case types.FP32:
		data := t.Data().([]float32)
		for i := 0; i < numSamples; i++ {
			for ch := 0; ch < channels; ch++ {
				sigData := channelSignals[ch].View().(vec.Vector)
				data[i*channels+ch] = sigData[i]
			}
		}
		case types.UINT8:
		data := t.Data().([]uint8)
		for i := 0; i < numSamples; i++ {
			for ch := 0; ch < channels; ch++ {
				sigData := channelSignals[ch].View().(vec.Vector)
				val := (sigData[i] + 1.0) * 127.5
				if val > 255.0 {
					val = 255.0
				} else if val < 0.0 {
					val = 0.0
				}
				data[i*channels+ch] = uint8(val)
			}
		}
	default:
		return fmt.Errorf("unsupported data type: %v", dt)
	}

	// Write to WAV file
	file, err := os.Create(output)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	m := audio.NewMarshaller(
		audio.WithWAVPath(output),
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(channels),
		types.WithDestinationType(dt),
	)

	if err := m.Marshal(file, t); err != nil {
		return fmt.Errorf("failed to write WAV file: %w", err)
	}

	fmt.Printf("Generated %s noise: %s\n", noiseType, output)
	fmt.Printf("  Channels: %d\n", channels)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Duration: %.2f s\n", duration)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitudes: %s dB\n", strings.Join(strings.Fields(fmt.Sprint(ampDbs[:channels])), ", "))

	return nil
}

