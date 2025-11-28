package main

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/spf13/cobra"
)

func newGenSineCmd() *cobra.Command {
	var (
		output     string
		amplitude  string
		frequency  string
		phase      string
		channels   int
		sampleRate int
		duration   time.Duration
		dataType   string
	)

	cmd := &cobra.Command{
		Use:   "gen-sine",
		Short: "Generate sine wave signal(s)",
		Long:  "Generate one or more sine waves and save to a WAV file. Supports per-channel frequency and phase.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenSine(output, amplitude, frequency, phase, channels, sampleRate, duration, dataType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().StringVarP(&frequency, "frequency", "f", "1000", "Frequency in Hz per channel (comma-separated, default: 1000)")
	cmd.Flags().StringVarP(&phase, "phase", "p", "0", "Phase in degrees per channel (comma-separated, default: 0)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 1, "Number of channels")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().DurationVarP(&duration, "duration", "d", 5*time.Second, "Duration (e.g., 5s, 100ms)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenSine(output, amplitude, frequency, phase string, channels, sampleRate int, duration time.Duration, dataType string) error {
	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Parse frequencies
	freqs, err := parseFloatSlice(frequency)
	if err != nil {
		return fmt.Errorf("invalid frequency: %w", err)
	}
	// Extend to channels if needed
	for len(freqs) < channels {
		freqs = append(freqs, freqs[len(freqs)-1])
	}

	// Parse phases (degrees to radians)
	phasesDeg, err := parseFloatSlice(phase)
	if err != nil {
		return fmt.Errorf("invalid phase: %w", err)
	}
	phases := make([]float32, channels)
	for i := 0; i < channels; i++ {
		if i < len(phasesDeg) {
			phases[i] = float32(phasesDeg[i] * math.Pi / 180.0)
		} else if len(phasesDeg) > 0 {
			phases[i] = float32(phasesDeg[len(phasesDeg)-1] * math.Pi / 180.0)
		} else {
			phases[i] = 0
		}
	}

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	// Generate signals
	numSamples := int(float64(sampleRate) * duration.Seconds())

	// Generate per-channel signals
	channelSignals := make([][]float32, channels)
	for ch := 0; ch < channels; ch++ {
		signal := make([]float32, numSamples)
		gen := dsp.NewSineGenerator(float32(sampleRate), float32(freqs[ch]), ampLinear, phases[ch])
		gen.Generate(signal)
		channelSignals[ch] = signal
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
				val := channelSignals[ch][i] * 32767.0
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
				val := channelSignals[ch][i] * 2147483647.0
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
				data[i*channels+ch] = channelSignals[ch][i]
			}
		}
	case types.UINT8:
		data := t.Data().([]uint8)
		for i := 0; i < numSamples; i++ {
			for ch := 0; ch < channels; ch++ {
				val := (channelSignals[ch][i] + 1.0) * 127.5
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

	fmt.Printf("Generated sine wave: %s\n", output)
	fmt.Printf("  Channels: %d\n", channels)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Duration: %v\n", duration)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Frequencies: %s Hz\n", strings.Join(strings.Fields(fmt.Sprint(freqs[:channels])), ", "))
	if len(phasesDeg) > 0 {
		fmt.Printf("  Phases: %s degrees\n", strings.Join(strings.Fields(fmt.Sprint(phasesDeg[:channels])), ", "))
	}

	return nil
}
