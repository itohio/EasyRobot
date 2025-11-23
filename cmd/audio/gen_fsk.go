package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenFSKCmd() *cobra.Command {
	var (
		output      string
		amplitude   string
		frequencies string
		data        string
		hex         string
		sampleRate  int
		symbolRate  float64
		dataType    string
	)

	cmd := &cobra.Command{
		Use:   "gen-fsk",
		Short: "Generate FSK modulated signal",
		Long:  "Generate a Frequency Shift Keying (FSK) modulated signal from data. Supports 2, 4, 8, or 16 frequencies.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenFSK(output, amplitude, frequencies, data, hex, sampleRate, symbolRate, dataType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().StringVarP(&frequencies, "frequencies", "f", "1000,2000", "Frequencies in Hz (comma-separated, 2-16 frequencies, default: 1000,2000)")
	cmd.Flags().StringVar(&data, "data", "Hello", "Data as string (default: \"Hello\")")
	cmd.Flags().StringVar(&hex, "hex", "", "Data as hex string (alternative to -data)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenFSK(output, amplitude, frequencies string, data, hex string, sampleRate int, symbolRate float64, dataType string) error {
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
	modOrder := len(freqs)
	if modOrder != 2 && modOrder != 4 && modOrder != 8 && modOrder != 16 {
		return fmt.Errorf("number of frequencies must be 2, 4, 8, or 16, got %d", modOrder)
	}

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

	// Create FSK encoder
	fskEncoder := mfsk.NewMModulator(float32(sampleRate), float32(symbolRate), freqs32...)
	fskEncoder.SetAmplitude(ampLinear)

	// Create marshaller based on output config
	ctx := context.Background()
	m := audio.NewMarshaller(
		audio.WithWAVPath(output),
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(1),
		types.WithDestinationType(dt),
	)

	// Encode entire signal first to calculate RMS statistics
	expectedSize := fskEncoder.Encode(nil, dataBytes)
	if expectedSize == 0 {
		return fmt.Errorf("FSK modulator expects 0 samples (check modulator configuration or data)")
	}

	modulatedVec := vec.New(expectedSize)
	samplesWritten := fskEncoder.Encode(modulatedVec, dataBytes)
	if samplesWritten == 0 {
		return fmt.Errorf("FSK modulator wrote 0 samples (check modulator configuration)")
	}

	modData := []float32(modulatedVec)[:samplesWritten]

	// Calculate RMS statistics
	if len(modData) > 0 {
		signalVec := vec.NewFrom(modData...)
		rmsStats := dsp.CalculateRMSStatistics(signalVec)
		fmt.Printf("  RMS Statistics:\n")
		fmt.Printf("    MIN RMS: %.2f dB\n", rmsStats.MinRMSDb)
		fmt.Printf("    MAX RMS: %.2f dB\n", rmsStats.MaxRMSDb)
		fmt.Printf("    AVG RMS: %.2f dB\n", rmsStats.AvgRMSDb)
	}

	// Reset encoder and use generic processAndWrite function
	fskEncoder.Reset()
	if err := processAndWrite(m, fskEncoder, dataBytes, sampleRate, dt, ctx); err != nil {
		return fmt.Errorf("failed to process and write: %w", err)
	}

	fmt.Printf("Generated FSK signal: %s\n", output)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Frequencies: %s Hz\n", strings.Join(strings.Fields(fmt.Sprint(freqs)), ", "))
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Modulation order: %d\n", modOrder)
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))

	return nil
}
