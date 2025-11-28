package main

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenFSKCmd() *cobra.Command {
	var (
		output          string
		amplitude       string
		frequencies     string
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
		Use:   "gen-fsk",
		Short: "Generate FSK modulated signal",
		Long:  "Generate a Frequency Shift Keying (FSK) modulated signal from data. Supports 2, 4, 8, or 16 frequencies. Supports Hamming/Reed-Solomon/Gray encoding.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenFSK(output, amplitude, frequencies, data, hex, sampleRate, symbolRate, modulationOrder, dataType, encoding, rsN, rsK)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().StringVarP(&frequencies, "frequencies", "f", "1000,2000", "Frequencies in Hz (comma-separated, 2-16 frequencies, default: 1000,2000)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (2, 4, 8, 16, default: 4)")
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

func runGenFSK(output, amplitude, frequencies string, data, hex string, sampleRate int, symbolRate float64, modulationOrder int, dataType string, encoding string, rsN, rsK int) error {
	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Parse frequencies
	freqs, err := parseFloatSlice(frequencies)
	if err != nil {
		return fmt.Errorf("invalid frequencies: %w", err)
	}
	if len(freqs) < 2 {
		return fmt.Errorf("at least 2 frequencies required, got %d", len(freqs))
	}
	if len(freqs) > 16 {
		return fmt.Errorf("maximum 16 frequencies supported, got %d", len(freqs))
	}
	// Validate modulation order is power of 2
	if modulationOrder != 2 && modulationOrder != 4 && modulationOrder != 8 && modulationOrder != 16 {
		return fmt.Errorf("modulation order must be 2, 4, 8, or 16, got %d", modulationOrder)
	}

	// If frequencies are provided, validate they match modulation order
	if len(freqs) != modulationOrder {
		// Generate frequencies based on modulation order if not matching
		if len(freqs) == 2 {
			// Use provided frequencies as base and generate evenly spaced frequencies
			baseFreq := freqs[0]
			spacing := freqs[1] - freqs[0]
			newFreqs := make([]float64, modulationOrder)
			for i := range modulationOrder {
				newFreqs[i] = baseFreq + float64(i)*spacing
			}
			freqs = newFreqs
		} else {
			return fmt.Errorf("number of frequencies (%d) must match modulation order (%d)", len(freqs), modulationOrder)
		}
	}

	modOrder := modulationOrder

	// Convert to float32
	freqs32 := make([]float32, len(freqs))
	for i, f := range freqs {
		freqs32[i] = float32(f)
	}

	// Parse data
	dataBytes, err := parseData(data, hex)
	if err != nil {
		return err
	}

	// Add sync bytes: 0xAA 0xAA at beginning, 0x55 0x55 at end
	syncPrefix := []byte{0xAA, 0xAA}
	syncSuffix := []byte{0x55, 0x55}
	dataBytes = append(syncPrefix, dataBytes...)
	dataBytes = append(dataBytes, syncSuffix...)

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	spacing := freqs32[1] - freqs32[0]
	for i := 2; i < len(freqs32); i++ {
		if math.Abs(float64((freqs32[i]-freqs32[i-1])-spacing)) > 1e-3 {
			return fmt.Errorf("frequencies must be evenly spaced for the current modulator, got %v", freqs)
		}
	}

	bitsPerSymbol := int(math.Log2(float64(modOrder)))
	samplesPerSymbol := int(math.Round(float64(sampleRate) / symbolRate))
	if samplesPerSymbol < 2 {
		return fmt.Errorf("sample-rate %d and symbol-rate %.2f yield <2 samples per symbol", sampleRate, symbolRate)
	}

	cfg := mfsk.DefaultConfig()
	cfg.SampleRate = float32(sampleRate)
	cfg.CarrierHz = freqs32[0]
	cfg.ToneSpacingHz = spacing
	cfg.BitsPerSymbol = bitsPerSymbol
	cfg.SamplesPerSymbol = samplesPerSymbol
	cfg.Amplitude = ampLinear

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid FSK config: %w", err)
	}

	fskMod, err := mfsk.NewModulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to create FSK modulator: %w", err)
	}

	// Create encoded modulator if encoding is requested
	encodedMod, encodingName, err := createEncodedModulator(fskMod, encoding, rsN, rsK)
	if err != nil {
		return fmt.Errorf("failed to create encoded modulator: %w", err)
	}

	modData, err := modulateSamples(encodedMod, bitsPerSymbol, dataBytes)
	if err != nil {
		return fmt.Errorf("failed to modulate data: %w", err)
	}
	if len(modData) == 0 {
		return fmt.Errorf("FSK modulator produced no samples")
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
	shape := tensor.NewShape(len(modData), 1)
	t := tensor.FromFloat32(shape, modData)
	if t.Empty() {
		return fmt.Errorf("failed to create tensor")
	}
	defer t.Release()

	ctx := context.Background()
	m := audio.NewMarshaller(
		audio.WithWAVPath(output),
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(1),
		types.WithDestinationType(dt),
	)

	if err := m.Marshal(nil, t, types.WithContext(ctx)); err != nil {
		return fmt.Errorf("failed to write: %w", err)
	}

	fmt.Printf("Generated FSK signal: %s\n", output)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Frequencies: %s Hz\n", strings.Join(strings.Fields(fmt.Sprint(freqs)), ", "))
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Modulation order: %d\n", modOrder)
	if encodingName != "none" {
		fmt.Printf("  Encoding: %s\n", encodingName)
	}
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))

	return nil
}
