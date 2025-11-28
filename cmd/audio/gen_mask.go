package main

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/gray"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/hamming"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/manchester"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/rs"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mask"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newGenMASKCmd() *cobra.Command {
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
		lineCode        string
		ecc             string
		rsN             int
		rsK             int
	)

	cmd := &cobra.Command{
		Use:   "gen-mask",
		Short: "Generate MASK modulated signal",
		Long:  "Generate an M-ary Amplitude Shift Keying (MASK) modulated signal from data. Supports Hamming/Reed-Solomon ECC, Gray/Manchester line coding, padding, and noise.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenMASK(output, deviceID, amplitude, frequency, data, hex, sampleRate, symbolRate, modulationOrder, dataType, pad, padType, padAmplitude, noise, noiseType, lineCode, ecc, rsN, rsK)
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
	cmd.Flags().IntVar(&modulationOrder, "modulation-order", 4, "Modulation order (2, 4, 8, 16, 32, 64, 128, 256, default: 4)")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().DurationVar(&pad, "pad", 1*time.Second, "Padding duration (adds padding at beginning and end, e.g., 1s, 100ms)")
	cmd.Flags().StringVar(&padType, "pad-type", "zero", "Padding type: \"zero\" or \"noise\" (default: zero)")
	cmd.Flags().StringVar(&padAmplitude, "pad-amplitude", "-20", "Padding noise amplitude in dB (only used if pad-type=noise, default: -20dB)")
	cmd.Flags().StringVar(&noise, "noise", "-120", "Noise amplitude in dB to add on top of signal (default: -120, meaning no noise)")
	cmd.Flags().StringVar(&noiseType, "noise-type", "white", "Noise type: \"white\", \"pink\", or \"brown\" (default: white)")
	cmd.Flags().StringVar(&lineCode, "line-code", "none", "Line coding: \"none\", \"gray\", or \"manchester\" (default: none)")
	cmd.Flags().StringVar(&ecc, "ecc", "none", "Error correction code: \"none\", \"hamming\", or \"rs\" (default: none)")
	cmd.Flags().IntVar(&rsN, "rs-n", 255, "Reed-Solomon total codeword length (n, default: 255)")
	cmd.Flags().IntVar(&rsK, "rs-k", 223, "Reed-Solomon data symbols per codeword (k, default: 223)")

	return cmd
}

func runGenMASK(output, deviceID, amplitude string, frequency float64, data, hex string, sampleRate int, symbolRate float64, modulationOrder int, dataType string, pad time.Duration, padType, padAmplitude, noise, noiseType, lineCode, ecc string, rsN, rsK int) error {
	// Validate output
	if output == "" && deviceID == "" {
		return fmt.Errorf("either --output or --device-id must be specified")
	}

	// Parse data
	dataBytes, err := parseData(data, hex)
	if err != nil {
		return err
	}

	// Apply ECC encoding if requested (before line coding)
	if ecc != "none" {
		reader := buffer.New(dataBytes)
		var encodedBuf []byte
		switch ecc {
		case "hamming":
			hamEnc := hamming.NewEncoder()
			// Estimate size (1:2 for Hamming)
			estimate := hamEnc.Process(nil, reader)
			reader.Reset()
			encodedBuf = make([]byte, 0, estimate)
			writer := buffer.New(encodedBuf)
			hamEnc.Process(writer, reader)
			dataBytes = writer.Items()
		case "rs":
			if rsN <= rsK {
				return fmt.Errorf("RS encoder: n (%d) must be greater than k (%d)", rsN, rsK)
			}
			if rsK <= 0 {
				return fmt.Errorf("RS encoder: k (%d) must be positive", rsK)
			}
			if rsN > 255 {
				return fmt.Errorf("RS encoder: n (%d) must be <= 255 for byte-level encoding", rsN)
			}
			rsEnc := rs.NewEncoder(rsN, rsK)
			// Estimate size
			estimate := rsEnc.Process(nil, reader)
			reader.Reset()
			encodedBuf = make([]byte, 0, estimate)
			writer := buffer.New(encodedBuf)
			rsEnc.Process(writer, reader)
			dataBytes = writer.Items()
		default:
			return fmt.Errorf("invalid ecc: %s (must be none, hamming, or rs)", ecc)
		}
	}

	// Apply line coding if requested
	if lineCode != "none" {
		reader := buffer.New(dataBytes)
		var encodedBuf []byte
		switch lineCode {
		case "gray":
			grayEnc := gray.NewEncoderWithBits(8)
			// Estimate size (1:1 for Gray)
			estimate := grayEnc.Process(nil, reader)
			reader.Reset()
			encodedBuf = make([]byte, 0, estimate)
			writer := buffer.New(encodedBuf)
			grayEnc.Process(writer, reader)
			dataBytes = writer.Items()
		case "manchester":
			manEnc := manchester.NewEncoder()
			// Estimate size (2x for Manchester)
			estimate := manEnc.Process(nil, reader)
			reader.Reset()
			encodedBuf = make([]byte, 0, estimate)
			writer := buffer.New(encodedBuf)
			manEnc.Process(writer, reader)
			dataBytes = writer.Items()
		default:
			return fmt.Errorf("invalid line-code: %s (must be none, gray, or manchester)", lineCode)
		}
	}

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
	validOrders := []int{2, 4, 8, 16, 32, 64, 128, 256}
	valid := false
	for _, v := range validOrders {
		if modulationOrder == v {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("modulation order must be one of %v, got %d", validOrders, modulationOrder)
	}

	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	if bitsPerSymbol <= 0 {
		return fmt.Errorf("invalid modulation order %d", modulationOrder)
	}

	samplesPerSymbol := int(math.Round(float64(sampleRate) / symbolRate))
	if samplesPerSymbol < 2 {
		return fmt.Errorf("sample-rate %d and symbol-rate %.2f yield <2 samples per symbol", sampleRate, symbolRate)
	}

	cfg := mask.DefaultConfig()
	cfg.SampleRate = float32(sampleRate)
	cfg.CarrierHz = float32(frequency)
	cfg.BitsPerSymbol = bitsPerSymbol
	cfg.SamplesPerSymbol = samplesPerSymbol
	cfg.Amplitude = ampLinear

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid MASK config: %w", err)
	}

	maskMod, err := mask.NewModulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to build MASK modulator: %w", err)
	}

	modData, err := modulateSamples(maskMod, bitsPerSymbol, dataBytes)
	if err != nil {
		return fmt.Errorf("failed to modulate data: %w", err)
	}
	if len(modData) == 0 {
		return fmt.Errorf("MASK modulator generated no samples")
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
		padSamples := int(float64(sampleRate) * pad.Seconds())
		vector := applyPadding(vec.NewFrom(modData...), padSamples, padType, padAmplitude, float32(sampleRate))
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
	shape := tensor.NewShape(len(modData), 1)
	t := tensor.FromFloat32(shape, modData)
	if t.Empty() {
		return fmt.Errorf("failed to create tensor")
	}
	defer t.Release()

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
		fmt.Printf("Generated MASK signal: %s\n", output)
	} else if deviceID != "" {
		fmt.Printf("Generated MASK signal: playing on device %s\n", deviceID)
	}

	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Channels: 1 (real samples)\n")
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Carrier frequency: %.0f Hz\n", frequency)
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Modulation order: %d (%d-ASK, %d bits per symbol)\n", modulationOrder, modulationOrder, bitsPerSymbol)
	if ecc != "none" {
		if ecc == "rs" {
			fmt.Printf("  ECC: %s (n=%d, k=%d, t=%d)\n", ecc, rsN, rsK, (rsN-rsK)/2)
		} else {
			fmt.Printf("  ECC: %s\n", ecc)
		}
	}
	if lineCode != "none" {
		fmt.Printf("  Line coding: %s\n", lineCode)
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
