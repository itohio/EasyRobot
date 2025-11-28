package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenChirpCmd() *cobra.Command {
	var (
		output     string
		amplitude  string
		freqMin    float64
		freqMax    float64
		channels   int
		sampleRate int
		duration   time.Duration
		dataType   string
		chirpType  string
	)

	cmd := &cobra.Command{
		Use:   "gen-chirp",
		Short: "Generate chirp signal (frequency sweep)",
		Long:  "Generate a linear chirp signal that sweeps from minimum to maximum frequency.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenChirp(output, amplitude, freqMin, freqMax, channels, sampleRate, duration, dataType, chirpType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64Var(&freqMin, "freq-min", 100, "Minimum frequency in Hz (default: 100)")
	cmd.Flags().Float64Var(&freqMax, "freq-max", 10000, "Maximum frequency in Hz (default: 10000)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 1, "Number of channels")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().DurationVarP(&duration, "duration", "d", 5*time.Second, "Duration (e.g., 5s, 100ms)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().StringVar(&chirpType, "chirp-type", "linear", "Chirp type: linear, log, or pink-log (default: linear)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenChirp(output, amplitude string, freqMin, freqMax float64, channels, sampleRate int, duration time.Duration, dataType string, chirpType string) error {
	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	numSamples := int(float64(sampleRate) * duration.Seconds())
	if numSamples <= 0 {
		return fmt.Errorf("duration %v yields no samples at %d Hz", duration, sampleRate)
	}

	chirpBuffer := make([]float32, numSamples)
	switch strings.ToLower(chirpType) {
	case "linear":
		gen := dsp.NewChirpGenerator(float32(sampleRate), float32(freqMin), float32(freqMax), ampLinear)
		gen.Generate(chirpBuffer)
	case "log", "logarithmic":
		gen := dsp.NewLogarithmicChirpGenerator(float32(sampleRate), float32(freqMin), float32(freqMax), ampLinear)
		gen.Generate(chirpBuffer)
	case "pink-log", "pink":
		gen := dsp.NewPinkLogarithmicChirpGenerator(float32(sampleRate), float32(freqMin), float32(freqMax), ampLinear)
		gen.Generate(chirpBuffer)
	default:
		return fmt.Errorf("unsupported chirp type: %s (use linear, log, or pink-log)", chirpType)
	}

	chirpData := vec.NewFrom(chirpBuffer...)

	// Combine channels into tensor [frames, channels]
	shape := tensor.NewShape(numSamples, channels)
	t := tensor.New(tensor.DataType(dt), shape)

	// Fill tensor with channel data (duplicate chirp to all channels)
	switch dt {
	case types.INT16:
		data := t.Data().([]int16)
		for i := 0; i < numSamples; i++ {
			val := chirpData[i] * 32767.0
			if val > 32767.0 {
				val = 32767.0
			} else if val < -32768.0 {
				val = -32768.0
			}
			for ch := 0; ch < channels; ch++ {
				data[i*channels+ch] = int16(val)
			}
		}
	case types.INT32:
		data := t.Data().([]int32)
		for i := 0; i < numSamples; i++ {
			val := chirpData[i] * 2147483647.0
			if val > 2147483647.0 {
				val = 2147483647.0
			} else if val < -2147483648.0 {
				val = -2147483648.0
			}
			for ch := 0; ch < channels; ch++ {
				data[i*channels+ch] = int32(val)
			}
		}
	case types.FP32:
		data := t.Data().([]float32)
		for i := 0; i < numSamples; i++ {
			for ch := 0; ch < channels; ch++ {
				data[i*channels+ch] = chirpData[i]
			}
		}
	case types.UINT8:
		data := t.Data().([]uint8)
		for i := 0; i < numSamples; i++ {
			val := (chirpData[i] + 1.0) * 127.5
			if val > 255.0 {
				val = 255.0
			} else if val < 0.0 {
				val = 0.0
			}
			u8Val := uint8(val)
			for ch := 0; ch < channels; ch++ {
				data[i*channels+ch] = u8Val
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

	fmt.Printf("Generated chirp: %s\n", output)
	fmt.Printf("  Channels: %d\n", channels)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Duration: %v\n", duration)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Frequency range: %.0f - %.0f Hz\n", freqMin, freqMax)
	fmt.Printf("  Chirp type: %s\n", chirpType)

	return nil
}
