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
	"github.com/spf13/cobra"
)

func newGenChirpCmd() *cobra.Command {
	var (
		output        string
		amplitude     string
		freqMin       float64
		freqMax       float64
		channels      int
		sampleRate    int
		duration      time.Duration
		dataType      string
		chirpType     string
		refChannel    int
		signalChannel int
		refAmplitude  string
		refDuration   time.Duration
		refFrequency  float64
		delay         time.Duration
		silenceBefore time.Duration
		silenceAfter  time.Duration
	)

	cmd := &cobra.Command{
		Use:   "gen-chirp",
		Short: "Generate chirp signal (frequency sweep)",
		Long:  "Generate a linear chirp signal that sweeps from minimum to maximum frequency. Supports reference signal, delays, and multi-channel output.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenChirp(output, amplitude, freqMin, freqMax, channels, sampleRate, duration, dataType, chirpType,
				refChannel, signalChannel, refAmplitude, refDuration, refFrequency, delay, silenceBefore, silenceAfter)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Chirp signal amplitude in dB (default: -6dB)")
	cmd.Flags().Float64Var(&freqMin, "freq-min", 20, "Minimum frequency in Hz (default: 20)")
	cmd.Flags().Float64Var(&freqMax, "freq-max", 22000, "Maximum frequency in Hz (default: 22000)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 2, "Number of channels")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().DurationVarP(&duration, "duration", "d", 5*time.Second, "Chirp signal duration (e.g., 5s, 100ms)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().StringVar(&chirpType, "chirp-type", "pink-log", "Chirp type: linear, log, or pink-log (default: pink-log)")
	cmd.Flags().IntVar(&refChannel, "ref-channel", 0, "Channel index for reference signal (default: 0)")
	cmd.Flags().IntVar(&signalChannel, "signal-channel", 1, "Channel index for chirp signal (default: 0)")
	cmd.Flags().StringVar(&refAmplitude, "ref-amplitude", "-6", "Reference signal amplitude in dB (default: -6dB)")
	cmd.Flags().DurationVar(&refDuration, "ref-duration", 250*time.Millisecond, "Reference signal duration (e.g., 100ms)")
	cmd.Flags().Float64Var(&refFrequency, "ref-frequency", 4400, "Reference signal frequency in Hz (default: 4400)")
	cmd.Flags().DurationVar(&delay, "delay", time.Second, "Delay between ref signal end and chirp signal start (e.g., 50ms)")
	cmd.Flags().DurationVar(&silenceBefore, "silence-before", time.Second, "Silence duration before ref signal (e.g., 100ms)")
	cmd.Flags().DurationVar(&silenceAfter, "silence-after", time.Second, "Silence duration after chirp signal (e.g., 100ms)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenChirp(output, amplitude string, freqMin, freqMax float64, channels, sampleRate int, duration time.Duration, dataType string, chirpType string,
	refChannel, signalChannel int, refAmplitude string, refDuration time.Duration, refFrequency float64, delay, silenceBefore, silenceAfter time.Duration) error {
	// Validate channel indices
	if refChannel < 0 || refChannel >= channels {
		return fmt.Errorf("ref-channel %d is out of range [0, %d)", refChannel, channels)
	}
	if signalChannel < 0 || signalChannel >= channels {
		return fmt.Errorf("signal-channel %d is out of range [0, %d)", signalChannel, channels)
	}

	// Parse amplitudes (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	refAmpDb, err := strconv.ParseFloat(refAmplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid ref-amplitude: %w", err)
	}
	refAmpLinear := dbToAmplitude(refAmpDb)

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	// Calculate buffer sizes
	silenceBeforeSamples := int(float64(sampleRate) * silenceBefore.Seconds())
	refSamples := int(float64(sampleRate) * refDuration.Seconds())
	delaySamples := int(float64(sampleRate) * delay.Seconds())
	chirpSamples := int(float64(sampleRate) * duration.Seconds())
	silenceAfterSamples := int(float64(sampleRate) * silenceAfter.Seconds())

	if chirpSamples <= 0 {
		return fmt.Errorf("chirp duration %v yields no samples at %d Hz", duration, sampleRate)
	}
	if refSamples <= 0 && refDuration > 0 {
		return fmt.Errorf("ref duration %v yields no samples at %d Hz", refDuration, sampleRate)
	}

	totalSamples := silenceBeforeSamples + refSamples + delaySamples + chirpSamples + silenceAfterSamples
	if totalSamples <= 0 {
		return fmt.Errorf("total duration yields no samples")
	}

	// Generate reference sine wave
	refBuffer := make([]float32, refSamples)
	if refSamples > 0 {
		refGen := dsp.NewSineGenerator(float32(sampleRate), float32(refFrequency), refAmpLinear, 0.0)
		refGen.Generate(refBuffer)
	}

	// Generate chirp signal
	chirpBuffer := make([]float32, chirpSamples)
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

	// Create multi-channel tensor [frames, channels]
	shape := tensor.NewShape(totalSamples, channels)
	t := tensor.New(tensor.DataType(dt), shape)

	// Fill tensor: all channels start as zero (silence)
	// Then add ref signal in refChannel, chirp in signalChannel
	switch dt {
	case types.INT16:
		data := t.Data().([]int16)
		// Initialize all to zero
		for i := range data {
			data[i] = 0
		}

		// Add reference signal in refChannel
		refOffset := silenceBeforeSamples
		for i := 0; i < refSamples; i++ {
			val := refBuffer[i] * 32767.0
			if val > 32767.0 {
				val = 32767.0
			} else if val < -32768.0 {
				val = -32768.0
			}
			idx := (refOffset+i)*channels + refChannel
			data[idx] = int16(val)
		}

		// Add chirp signal in signalChannel
		chirpOffset := silenceBeforeSamples + refSamples + delaySamples
		for i := 0; i < chirpSamples; i++ {
			val := chirpBuffer[i] * 32767.0
			if val > 32767.0 {
				val = 32767.0
			} else if val < -32768.0 {
				val = -32768.0
			}
			idx := (chirpOffset+i)*channels + signalChannel
			data[idx] = int16(val)
		}

	case types.INT32:
		data := t.Data().([]int32)
		// Initialize all to zero
		for i := range data {
			data[i] = 0
		}

		// Add reference signal in refChannel
		refOffset := silenceBeforeSamples
		for i := 0; i < refSamples; i++ {
			val := refBuffer[i] * 2147483647.0
			if val > 2147483647.0 {
				val = 2147483647.0
			} else if val < -2147483648.0 {
				val = -2147483648.0
			}
			idx := (refOffset+i)*channels + refChannel
			data[idx] = int32(val)
		}

		// Add chirp signal in signalChannel
		chirpOffset := silenceBeforeSamples + refSamples + delaySamples
		for i := 0; i < chirpSamples; i++ {
			val := chirpBuffer[i] * 2147483647.0
			if val > 2147483647.0 {
				val = 2147483647.0
			} else if val < -2147483648.0 {
				val = -2147483648.0
			}
			idx := (chirpOffset+i)*channels + signalChannel
			data[idx] = int32(val)
		}

	case types.FP32:
		data := t.Data().([]float32)
		// Initialize all to zero
		for i := range data {
			data[i] = 0
		}

		// Add reference signal in refChannel
		refOffset := silenceBeforeSamples
		for i := 0; i < refSamples; i++ {
			idx := (refOffset+i)*channels + refChannel
			data[idx] = refBuffer[i]
		}

		// Add chirp signal in signalChannel
		chirpOffset := silenceBeforeSamples + refSamples + delaySamples
		for i := 0; i < chirpSamples; i++ {
			idx := (chirpOffset+i)*channels + signalChannel
			data[idx] = chirpBuffer[i]
		}

	case types.UINT8:
		data := t.Data().([]uint8)
		// Initialize all to zero (128 for uint8 = 0.0 in float32)
		for i := range data {
			data[i] = 128
		}

		// Add reference signal in refChannel
		refOffset := silenceBeforeSamples
		for i := 0; i < refSamples; i++ {
			val := (refBuffer[i] + 1.0) * 127.5
			if val > 255.0 {
				val = 255.0
			} else if val < 0.0 {
				val = 0.0
			}
			idx := (refOffset+i)*channels + refChannel
			data[idx] = uint8(val)
		}

		// Add chirp signal in signalChannel
		chirpOffset := silenceBeforeSamples + refSamples + delaySamples
		for i := 0; i < chirpSamples; i++ {
			val := (chirpBuffer[i] + 1.0) * 127.5
			if val > 255.0 {
				val = 255.0
			} else if val < 0.0 {
				val = 0.0
			}
			idx := (chirpOffset+i)*channels + signalChannel
			data[idx] = uint8(val)
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
	fmt.Printf("  Total duration: %.3f s\n", float64(totalSamples)/float64(sampleRate))
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Chirp: channel %d, %.2f dB, %.0f-%.0f Hz, %v, %s\n", signalChannel, ampDb, freqMin, freqMax, duration, chirpType)
	if refSamples > 0 {
		fmt.Printf("  Reference: channel %d, %.2f dB, %.0f Hz, %v\n", refChannel, refAmpDb, refFrequency, refDuration)
		fmt.Printf("  Delay: %v\n", delay)
	}
	if silenceBefore > 0 {
		fmt.Printf("  Silence before: %v\n", silenceBefore)
	}
	if silenceAfter > 0 {
		fmt.Printf("  Silence after: %v\n", silenceAfter)
	}

	return nil
}
