package main

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenFSKCmd() *cobra.Command {
	var (
		output     string
		amplitude  string
		frequencies string
		data       string
		hex        string
		sampleRate int
		symbolRate float64
		dataType   string
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

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	// Create FSK encoder
	fskEncoder := dsp.NewMFSKEncoder(float32(sampleRate), float32(symbolRate), freqs32...)

	// Estimate length needed
	bitsPerSymbol := int(math.Log2(float64(modOrder)))
	numSymbols := (len(dataBytes) * 8) / bitsPerSymbol
	if (len(dataBytes)*8)%bitsPerSymbol != 0 {
		numSymbols++
	}
	estimatedLength := numSymbols * int(float32(sampleRate)/float32(symbolRate))

	// Generate signal
	modulated := fskEncoder.GenerateSignal(dataBytes, ampLinear, estimatedLength)
	modData := modulated.View().(vec.Vector)
	numSamples := len(modData)

	// Create tensor [frames, 1]
	shape := tensor.NewShape(numSamples, 1)
	t := tensor.New(tensor.DataType(dt), shape)

	// Fill tensor
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
			data[i] = int16(val)
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
			data[i] = int32(val)
		}
	case types.FP32:
		data := t.Data().([]float32)
		for i := 0; i < numSamples; i++ {
			data[i] = modData[i]
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
			data[i] = uint8(val)
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
		audio.WithChannels(1),
		types.WithDestinationType(dt),
	)

	if err := m.Marshal(file, t); err != nil {
		return fmt.Errorf("failed to write WAV file: %w", err)
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

