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

func newGenQAMCmd() *cobra.Command {
	var (
		output     string
		amplitude  string
		frequency  float64
		poles      int
		data       string
		hex        string
		sampleRate int
		symbolRate float64
		dataType   string
	)

	cmd := &cobra.Command{
		Use:   "gen-qam",
		Short: "Generate QAM modulated signal",
		Long:  "Generate a Quadrature Amplitude Modulation (QAM) signal from data.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenQAM(output, amplitude, frequency, poles, data, hex, sampleRate, symbolRate, dataType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().IntVarP(&poles, "poles", "n", 16, "Number of constellation points (4, 16, 64, 256, default: 16)")
	cmd.Flags().StringVar(&data, "data", "Hello", "Data as string (default: \"Hello\")")
	cmd.Flags().StringVar(&hex, "hex", "", "Data as hex string (alternative to -data)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runGenQAM(output, amplitude string, frequency float64, poles int, data, hex string, sampleRate int, symbolRate float64, dataType string) error {
	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Validate poles
	if poles != 4 && poles != 16 && poles != 64 && poles != 256 {
		return fmt.Errorf("poles must be 4, 16, 64, or 256, got %d", poles)
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

	// For now, use QAM16Encoder (can be extended later for other orders)
	if poles != 16 {
		return fmt.Errorf("only 16-QAM is currently supported (poles=16), got %d", poles)
	}

	// Create QAM encoder
	qamEncoder := dsp.NewQAM16Encoder(float32(sampleRate), float32(frequency), float32(symbolRate))

	// Estimate length needed (4 bits per symbol for 16-QAM)
	numSymbols := (len(dataBytes) * 8) / 4
	if (len(dataBytes)*8)%4 != 0 {
		numSymbols++
	}
	estimatedLength := numSymbols * int(float32(sampleRate)/float32(symbolRate))

	// Generate signal
	modulated := qamEncoder.GenerateSignal(dataBytes, ampLinear, estimatedLength)
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

	fmt.Printf("Generated QAM signal: %s\n", output)
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Carrier frequency: %.0f Hz\n", frequency)
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Constellation points: %d\n", poles)
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))

	return nil
}

