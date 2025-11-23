package main

import (
	"context"
	"fmt"
	"math"
	"strconv"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mpsk"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenPSKCmd() *cobra.Command {
	var (
		output         string
		deviceID       string
		amplitude      string
		frequency      float64
		data           string
		hex            string
		sampleRate     int
		symbolRate     float64
		modulationOrder int
		dataType       string
		pad            float64
		padType        string
		padAmplitude   string
		noise          string
		noiseType      string
	)

	cmd := &cobra.Command{
		Use:   "gen-psk",
		Short: "Generate PSK modulated signal",
		Long:  "Generate a Phase Shift Keying (PSK) modulated signal from data. Supports protocol wrapping, padding, and noise.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenPSK(output, deviceID, amplitude, frequency, data, hex, sampleRate, symbolRate, modulationOrder, dataType, pad, padType, padAmplitude, noise, noiseType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (or use --device-id for audio device)")
	cmd.Flags().StringVar(&deviceID, "device-id", "", "Playback device ID (alternative to --output)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().StringVar(&data, "data", "Hello", "Data as string (default: \"Hello\")")
	cmd.Flags().StringVar(&hex, "hex", "", "Data as hex string (alternative to -data)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVar(&modulationOrder, "modulation-order", 2, "Modulation order (2=BPSK, 4=QPSK, default: 2)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().Float64Var(&pad, "pad", 1.0, "Padding duration in seconds (adds padding at beginning and end, default: 1.0)")
	cmd.Flags().StringVar(&padType, "pad-type", "zero", "Padding type: \"zero\" or \"noise\" (default: zero)")
	cmd.Flags().StringVar(&padAmplitude, "pad-amplitude", "-20", "Padding noise amplitude in dB (only used if pad-type=noise, default: -20dB)")
	cmd.Flags().StringVar(&noise, "noise", "-120", "Noise amplitude in dB to add on top of signal (default: -120, meaning no noise)")
	cmd.Flags().StringVar(&noiseType, "noise-type", "white", "Noise type: \"white\", \"pink\", or \"brown\" (default: white)")

	return cmd
}

func runGenPSK(output, deviceID, amplitude string, frequency float64, data, hex string, sampleRate int, symbolRate float64, modulationOrder int, dataType string, pad float64, padType, padAmplitude, noise, noiseType string) error {
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

	// Validate modulation order
	if modulationOrder != 2 && modulationOrder != 4 && modulationOrder != 8 && modulationOrder != 16 {
		return fmt.Errorf("modulation order must be 2, 4, 8, or 16, got %d", modulationOrder)
	}

	// Create PSK encoder
	pskEncoder := mpsk.NewModulator(float32(sampleRate), float32(frequency), float32(symbolRate), modulationOrder)
	pskEncoder.SetAmplitude(ampLinear)

	// Estimate length needed (symbols * samples per symbol)
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	numSymbols := (len(dataBytes) * 8) / bitsPerSymbol
	if (len(dataBytes)*8)%bitsPerSymbol != 0 {
		numSymbols++
	}
	estimatedLength := numSymbols * int(float32(sampleRate)/float32(symbolRate))

	// Generate signal
	modulated := pskEncoder.GenerateSignal(dataBytes, ampLinear, estimatedLength)
	modData := []float32(modulated)
	numSamples := len(modData)

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
		padSamples := int(float64(sampleRate) * pad)
		modData = applyPadding(modData, padSamples, padType, padAmplitude, float32(sampleRate))
		numSamples = len(modData)
	}

	// Apply noise if requested (if noise > -120 dB)
	noiseDb, err := strconv.ParseFloat(noise, 64)
	if err != nil {
		return fmt.Errorf("invalid noise amplitude: %w", err)
	}
	if noiseDb > -120 {
		noiseAmp := dbToAmplitude(noiseDb)
		modData = applyNoise(modData, noiseAmp, noiseType, float32(sampleRate))
	}

	// Create tensor [frames, 1]
	shape := tensor.NewShape(numSamples, 1)
	t := tensor.New(tensortypes.DataType(dt), shape)

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

	// Create marshaller based on output config
	ctx := context.Background()
	opts := []types.Option{
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(1),
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
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))
	if pad > 0 {
		fmt.Printf("  Padding: %.2f s (%s)\n", pad, padType)
	}
	if noiseDb > -120 {
		fmt.Printf("  Noise: %.2f dB (%s)\n", noiseDb, noiseType)
	}

	return nil
}

