package main

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mpsk"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenPSKCmd() *cobra.Command {
	var (
		output          string
		deviceID        string
		amplitude       string
		frequency       float64
		data            string
		hex             string
		sampleRate      int
		symbolRate      float64
		modulationOrder int
		dataType        string
		pad             time.Duration
		padType         string
		padAmplitude    string
		noise           string
		noiseType       string
		encoding        string
		rsN             int
		rsK             int
	)

	cmd := &cobra.Command{
		Use:   "gen-psk",
		Short: "Generate PSK modulated signal",
		Long:  "Generate a Phase Shift Keying (PSK) modulated signal from data. Supports Hamming/Reed-Solomon/Gray encoding, protocol wrapping, padding, and noise.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenPSK(output, deviceID, amplitude, frequency, data, hex, sampleRate, symbolRate, modulationOrder, dataType, pad, padType, padAmplitude, noise, noiseType, encoding, rsN, rsK)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (or use --device-id for audio device)")
	cmd.Flags().StringVar(&deviceID, "device-id", "", "Playback device ID (alternative to --output)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 5000, "Carrier frequency in Hz (default: 5000)")
	cmd.Flags().StringVar(&data, "data", "Hello", "Data as string (default: \"Hello\")")
	cmd.Flags().StringVar(&hex, "hex", "", "Data as hex string (alternative to -data)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (2=BPSK, 4=QPSK, 8=8-PSK, 16=16-PSK, default: 4)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().DurationVar(&pad, "pad", 1*time.Second, "Padding duration (adds padding at beginning and end, e.g., 1s, 100ms)")
	cmd.Flags().StringVar(&padType, "pad-type", "zero", "Padding type: \"zero\" or \"noise\" (default: zero)")
	cmd.Flags().StringVar(&padAmplitude, "pad-amplitude", "-20", "Padding noise amplitude in dB (only used if pad-type=noise, default: -20dB)")
	cmd.Flags().StringVar(&noise, "noise", "-120", "Noise amplitude in dB to add on top of signal (default: -120, meaning no noise)")
	cmd.Flags().StringVar(&noiseType, "noise-type", "white", "Noise type: \"white\", \"pink\", or \"brown\" (default: white)")
	cmd.Flags().StringVar(&encoding, "encoding", "none", "Data encoding: \"none\", \"gray\", \"hamming\", or \"rs\" (default: none)")
	cmd.Flags().IntVar(&rsN, "rs-n", 255, "Reed-Solomon total codeword length (n, default: 255)")
	cmd.Flags().IntVar(&rsK, "rs-k", 223, "Reed-Solomon data symbols per codeword (k, default: 223)")

	return cmd
}

func runGenPSK(output, deviceID, amplitude string, frequency float64, data, hex string, sampleRate int, symbolRate float64, modulationOrder int, dataType string, pad time.Duration, padType, padAmplitude, noise, noiseType, encoding string, rsN, rsK int) error {
	// Validate output
	if output == "" && deviceID == "" {
		return fmt.Errorf("either --output or --device-id must be specified")
	}

	// Parse data
	dataBytes, err := parseData(data, hex)
	if err != nil {
		return err
	}

	// Work directly with user data - no protocol wrapping

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

	if modulationOrder != 2 && modulationOrder != 4 && modulationOrder != 8 && modulationOrder != 16 {
		return fmt.Errorf("modulation order must be 2, 4, 8, or 16, got %d", modulationOrder)
	}

	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	if bitsPerSymbol <= 0 {
		return fmt.Errorf("invalid modulation order %d", modulationOrder)
	}

	samplesPerSymbol := int(math.Round(float64(sampleRate) / symbolRate))
	if samplesPerSymbol < 2 {
		return fmt.Errorf("sample-rate %d and symbol-rate %.2f yield <2 samples per symbol", sampleRate, symbolRate)
	}

	cfg := mpsk.DefaultConfig()
	cfg.SampleRate = float32(sampleRate)
	cfg.CarrierHz = float32(frequency)
	cfg.BitsPerSymbol = bitsPerSymbol
	cfg.SamplesPerSymbol = samplesPerSymbol
	cfg.Amplitude = ampLinear

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid PSK config: %w", err)
	}

	pskMod, err := mpsk.NewModulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to build PSK modulator: %w", err)
	}

	// Create encoded modulator if encoding is requested
	encodedMod, encodingName, err := createEncodedModulator(pskMod, encoding, rsN, rsK)
	if err != nil {
		return fmt.Errorf("failed to create encoded modulator: %w", err)
	}

	modData, err := modulateSamples(encodedMod, bitsPerSymbol, dataBytes)
	if err != nil {
		return fmt.Errorf("failed to modulate data: %w", err)
	}
	if len(modData) == 0 {
		return fmt.Errorf("PSK modulator generated no samples")
	}

	// Calculate RMS statistics (before padding/noise)
	if len(modData) > 0 {
		signalVec := vec.NewFrom(modData...)
		rmsStats := dsp.CalculateRMSStatistics(signalVec)
		fmt.Printf("  RMS Statistics:\n")
		fmt.Printf("    MIN RMS: %.2f dB\n", rmsStats.MinRMSDb)
		fmt.Printf("    MAX RMS: %.2f dB\n", rmsStats.MaxRMSDb)
		fmt.Printf("    AVG RMS: %.2f dB\n", rmsStats.AvgRMSDb)
	}

	// Apply padding if requested
	if pad > 0 {
		padScalars := int(float64(sampleRate)*pad.Seconds()) * 2
		vector := applyPadding(vec.NewFrom(modData...), padScalars, padType, padAmplitude, float32(sampleRate))
		modData = []float32(vector)
	}

	// Apply noise if requested (if noise > -120 dB)
	noiseDb, err := strconv.ParseFloat(noise, 64)
	if err != nil {
		return fmt.Errorf("invalid noise amplitude: %w", err)
	}
	if noiseDb > -120 {
		noiseAmp := dbToAmplitude(noiseDb)
		vector := applyNoise(vec.NewFrom(modData...), noiseAmp, noiseType, float32(sampleRate))
		modData = []float32(vector)
	}

	// Create FP32 tensor from samples, marshaller will handle type conversion
	frames := len(modData) / 2
	shape := tensor.NewShape(frames, 2)
	t := tensor.FromFloat32(shape, modData)
	if t.Empty() {
		return fmt.Errorf("failed to create tensor")
	}
	defer t.Release()

	// Create marshaller based on output config
	ctx := context.Background()
	opts := []types.Option{
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(2),
		types.WithDestinationType(dt),
	}
	if output != "" {
		opts = append(opts, audio.WithWAVPath(output))
	}
	if deviceID != "" {
		opts = append(opts, audio.WithPlaybackDevice(deviceID))
	}

	m := audio.NewMarshaller(opts...)

	// Write using marshaller (works with WAV files or devices)
	if err := m.Marshal(nil, t, types.WithContext(ctx)); err != nil {
		return fmt.Errorf("failed to write: %w", err)
	}

	if output != "" {
		fmt.Printf("Generated PSK signal: %s\n", output)
	} else if deviceID != "" {
		fmt.Printf("Generated PSK signal: playing on device %s\n", deviceID)
	}

	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Channels: 2 (IQ interleaved)\n")
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Carrier frequency: %.0f Hz\n", frequency)
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Modulation order: %d (", modulationOrder)
	switch modulationOrder {
	case 2:
		fmt.Print("BPSK")
	case 4:
		fmt.Print("QPSK")
	case 8:
		fmt.Print("8-PSK")
	case 16:
		fmt.Print("16-PSK")
	}
	fmt.Printf(")\n")
	if encodingName != "none" {
		fmt.Printf("  Encoding: %s\n", encodingName)
	}
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))
	if pad > 0 {
		fmt.Printf("  Padding: %v (%s)\n", pad, padType)
	}
	if noiseDb > -120 {
		fmt.Printf("  Noise: %.2f dB (%s)\n", noiseDb, noiseType)
	}

	return nil
}
