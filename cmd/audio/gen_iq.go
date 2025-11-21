package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenIQCmd() *cobra.Command {
	var (
		output     string
		amplitude  string
		frequency  float64
		channels   int
		sampleRate int
		duration   float64
		dataType   string
	)

	cmd := &cobra.Command{
		Use:   "gen-iq",
		Short: "Generate IQ modulated signal",
		Long:  "Generate an IQ (In-phase/Quadrature) modulated signal with I and Q components.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenIQ(output, amplitude, frequency, channels, sampleRate, duration, dataType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 1, "Number of channels")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64VarP(&duration, "duration", "d", 5.0, "Duration in seconds")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenIQ(output, amplitude string, frequency float64, channels, sampleRate int, duration float64, dataType string) error {
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

	// Generate I and Q signals (sine and cosine)
	numSamples := int(float64(sampleRate) * duration)
	sg := dsp.NewSignalGenerator1D()

	// Generate I (in-phase) and Q (quadrature) signals
	I := sg.Cosine(numSamples, float32(frequency), ampLinear, 0, float32(sampleRate))
	Q := sg.Sinusoid(numSamples, float32(frequency), ampLinear, 0, float32(sampleRate))

	// Modulate using IQ encoder
	iqEncoder := dsp.NewIQEncoder(float32(sampleRate), float32(frequency))
	modulated := iqEncoder.GenerateIQSignal(I, Q)
	modData := modulated.View().(vec.Vector)

	// Combine channels into tensor [frames, channels]
	shape := tensor.NewShape(numSamples, channels)
	t := tensor.New(tensor.DataType(dt), shape)

	// Fill tensor with channel data
	switch dt {
	case types.INT16:
		data := t.Data().([]int16)
		for i := 0; i < numSamples; i++ {
			val := modData[i] * 32767.0
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
			val := modData[i] * 2147483647.0
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
				data[i*channels+ch] = modData[i]
			}
		}
	case types.UINT8:
		data := t.Data().([]uint8)
		for i := 0; i < numSamples; i++ {
			val := (modData[i] + 1.0) * 127.5
			if val > 255.0 {
				val = 255.0
			} else if val < 0.0 {
				val = 0.0
			}
			for ch := 0; ch < channels; ch++ {
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

	fmt.Printf("Generated IQ signal: %s\n", output)
	fmt.Printf("  Channels: %d\n", channels)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Duration: %.2f s\n", duration)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Carrier frequency: %.0f Hz\n", frequency)

	return nil
}

