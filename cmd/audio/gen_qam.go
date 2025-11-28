package main

import (
	"context"
	"fmt"
	"math"
	"strconv"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/qam"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenQAMCmd() *cobra.Command {
	var (
		output          string
		amplitude       string
		frequency       float64
		poles           int
		data            string
		hex             string
		sampleRate      int
		symbolRate      float64
		modulationOrder int
		dataType        string
		encoding        string
		rsN             int
		rsK             int
	)

	cmd := &cobra.Command{
		Use:   "gen-qam",
		Short: "Generate QAM modulated signal",
		Long:  "Generate a Quadrature Amplitude Modulation (QAM) signal from data. Supports Hamming/Reed-Solomon/Gray encoding.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenQAM(output, amplitude, frequency, poles, data, hex, sampleRate, symbolRate, modulationOrder, dataType, encoding, rsN, rsK)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 5000, "Carrier frequency in Hz (default: 5000)")
	cmd.Flags().IntVarP(&poles, "poles", "n", 16, "Number of constellation points (4, 16, 64, 256, default: 16)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (4, 16, 64, 256, default: 4)")
	cmd.Flags().StringVar(&data, "data", "Hello", "Data as string (default: \"Hello\")")
	cmd.Flags().StringVar(&hex, "hex", "", "Data as hex string (alternative to -data)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().StringVar(&encoding, "encoding", "none", "Data encoding: \"none\", \"gray\", \"hamming\", or \"rs\" (default: none)")
	cmd.Flags().IntVar(&rsN, "rs-n", 255, "Reed-Solomon total codeword length (n, default: 255)")
	cmd.Flags().IntVar(&rsK, "rs-k", 223, "Reed-Solomon data symbols per codeword (k, default: 223)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenQAM(output, amplitude string, frequency float64, poles int, data, hex string, sampleRate int, symbolRate float64, modulationOrder int, dataType string, encoding string, rsN, rsK int) error {
	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Use modulation order if provided, otherwise use poles
	if modulationOrder > 0 {
		// Validate modulation order
		if modulationOrder != 4 && modulationOrder != 16 && modulationOrder != 64 && modulationOrder != 256 {
			return fmt.Errorf("modulation order must be 4, 16, 64, or 256, got %d", modulationOrder)
		}
		// Override poles with modulation order
		poles = modulationOrder
	} else {
		// Validate poles
		if poles != 4 && poles != 16 && poles != 64 && poles != 256 {
			return fmt.Errorf("poles must be 4, 16, 64, or 256, got %d", poles)
		}
	}

	// Parse data
	dataBytes, err := parseData(data, hex)
	if err != nil {
		return err
	}

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	bitsPerSymbol := int(math.Log2(float64(poles)))
	if bitsPerSymbol <= 0 || bitsPerSymbol%2 == 1 {
		return fmt.Errorf("poles must be an even power of two, got %d", poles)
	}

	samplesPerSymbol := int(math.Round(float64(sampleRate) / symbolRate))
	if samplesPerSymbol < 2 {
		return fmt.Errorf("sample-rate %d and symbol-rate %.2f yield <2 samples per symbol", sampleRate, symbolRate)
	}

	cfg := qam.DefaultConfig()
	cfg.SampleRate = float32(sampleRate)
	cfg.CarrierHz = float32(frequency)
	cfg.BitsPerSymbol = bitsPerSymbol
	cfg.SamplesPerSymbol = samplesPerSymbol
	cfg.Amplitude = ampLinear

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid QAM config: %w", err)
	}

	qamMod, err := qam.NewModulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to create QAM modulator: %w", err)
	}

	// Create encoded modulator if encoding is requested
	encodedMod, encodingName, err := createEncodedModulator(qamMod, encoding, rsN, rsK)
	if err != nil {
		return fmt.Errorf("failed to create encoded modulator: %w", err)
	}

	modData, err := modulateSamples(encodedMod, bitsPerSymbol, dataBytes)
	if err != nil {
		return fmt.Errorf("failed to modulate data: %w", err)
	}
	if len(modData) == 0 {
		return fmt.Errorf("QAM modulator produced no samples")
	}

	if len(modData) > 0 {
		signalVec := vec.NewFrom(modData...)
		rmsStats := dsp.CalculateRMSStatistics(signalVec)
		fmt.Printf("  RMS Statistics:\n")
		fmt.Printf("    MIN RMS: %.2f dB\n", rmsStats.MinRMSDb)
		fmt.Printf("    MAX RMS: %.2f dB\n", rmsStats.MaxRMSDb)
		fmt.Printf("    AVG RMS: %.2f dB\n", rmsStats.AvgRMSDb)
	}

	// Create FP32 tensor from samples, marshaller will handle type conversion
	frames := len(modData) / 2
	shape := tensor.NewShape(frames, 2)
	t := tensor.FromFloat32(shape, modData)
	if t.Empty() {
		return fmt.Errorf("failed to create tensor")
	}
	defer t.Release()

	ctx := context.Background()
	m := audio.NewMarshaller(
		audio.WithWAVPath(output),
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(2),
		types.WithDestinationType(dt),
	)

	if err := m.Marshal(nil, t, types.WithContext(ctx)); err != nil {
		return fmt.Errorf("failed to write: %w", err)
	}

	fmt.Printf("Generated QAM signal: %s\n", output)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Channels: 2 (IQ interleaved)\n")
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Carrier frequency: %.0f Hz\n", frequency)
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Constellation points: %d\n", poles)
	if encodingName != "none" {
		fmt.Printf("  Encoding: %s\n", encodingName)
	}
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))

	return nil
}
