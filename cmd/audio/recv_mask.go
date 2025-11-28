package main

import (
	"context"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/gray"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/hamming"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/manchester"
	"github.com/itohio/EasyRobot/x/math/dsp/coding/rs"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mask"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvMASKCmd() *cobra.Command {
	var (
		deviceID        string
		input           string
		frequency       float64
		sampleRate      int
		symbolRate      float64
		modulationOrder int
		lineCode        string
		ecc             string
		rsN             int
		rsK             int
	)

	cmd := &cobra.Command{
		Use:   "recv-mask",
		Short: "Receive MASK modulated data from audio device or WAV file",
		Long:  "Listens for MASK modulated signals and decodes them, printing received data. Supports Hamming/Reed-Solomon ECC and Gray/Manchester line coding.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvMASK(deviceID, input, frequency, sampleRate, symbolRate, modulationOrder, lineCode, ecc, rsN, rsK)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVar(&modulationOrder, "modulation-order", 4, "Modulation order (2, 4, 8, 16, 32, 64, 128, 256, default: 4)")
	cmd.Flags().StringVar(&lineCode, "line-code", "none", "Line coding: \"none\", \"gray\", or \"manchester\" (default: none)")
	cmd.Flags().StringVar(&ecc, "ecc", "none", "Error correction code: \"none\", \"hamming\", or \"rs\" (default: none)")
	cmd.Flags().IntVar(&rsN, "rs-n", 255, "Reed-Solomon total codeword length (n, default: 255)")
	cmd.Flags().IntVar(&rsK, "rs-k", 223, "Reed-Solomon data symbols per codeword (k, default: 223)")

	return cmd
}

func runRecvMASK(deviceID, input string, frequency float64, sampleRate int, symbolRate float64, modulationOrder int, lineCode, ecc string, rsN, rsK int) error {
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

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid MASK demod config: %w", err)
	}

	maskDecoder, err := mask.NewDemodulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to create MASK demodulator: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		fmt.Println("\nStopping...")
		cancel()
	}()

	// Create unmarshaller based on input config
	var unmarshaller types.Unmarshaller
	if input != "" {
		fmt.Printf("Reading MASK signal from WAV file: %s\n", input)
		unmarshaller = audio.NewUnmarshaller(audio.WithWAVPath(input))
	} else {
		fmt.Printf("Listening for MASK signals on audio device...\n")
		opts := []types.Option{
			audio.WithSampleRate(sampleRate),
			audio.WithChannels(1),
		}
		if deviceID != "" {
			opts = append(opts, audio.WithCaptureDevice(deviceID))
		}
		unmarshaller = audio.NewUnmarshaller(opts...)
	}

	fmt.Println("Press Ctrl+C to stop")

	onDecoded := func(decodedData []byte) error {
		if len(decodedData) == 0 {
			return nil
		}

		// Apply line coding decoding if requested (before ECC)
		if lineCode != "none" {
			reader := buffer.New(decodedData)
			var decodedBuf []byte
			switch lineCode {
			case "gray":
				grayDec := gray.NewDecoderWithBits(8)
				// Estimate size (1:1 for Gray)
				estimate := grayDec.Process(nil, reader)
				reader.Reset()
				decodedBuf = make([]byte, 0, estimate)
				writer := buffer.New(decodedBuf)
				grayDec.Process(writer, reader)
				decodedData = writer.Items()
			case "manchester":
				manDec := manchester.NewDecoder()
				// Estimate size (1:2 for Manchester)
				estimate := manDec.Process(nil, reader)
				reader.Reset()
				decodedBuf = make([]byte, 0, estimate)
				writer := buffer.New(decodedBuf)
				manDec.Process(writer, reader)
				decodedData = writer.Items()
			default:
				return fmt.Errorf("invalid line-code: %s (must be none, gray, or manchester)", lineCode)
			}
		}

		// Apply ECC decoding if requested (after line coding)
		if ecc != "none" {
			reader := buffer.New(decodedData)
			var decodedBuf []byte
			switch ecc {
			case "hamming":
				hamDec := hamming.NewDecoder()
				// Estimate size (2:1 for Hamming)
				estimate := hamDec.Process(nil, reader)
				reader.Reset()
				decodedBuf = make([]byte, 0, estimate)
				writer := buffer.New(decodedBuf)
				hamDec.Process(writer, reader)
				decodedData = writer.Items()
			case "rs":
				if rsN <= rsK {
					return fmt.Errorf("RS decoder: n (%d) must be greater than k (%d)", rsN, rsK)
				}
				if rsK <= 0 {
					return fmt.Errorf("RS decoder: k (%d) must be positive", rsK)
				}
				if rsN > 255 {
					return fmt.Errorf("RS decoder: n (%d) must be <= 255 for byte-level decoding", rsN)
				}
				rsDec := rs.NewDecoder(rsN, rsK)
				// Estimate size
				estimate := rsDec.Process(nil, reader)
				reader.Reset()
				decodedBuf = make([]byte, 0, estimate)
				writer := buffer.New(decodedBuf)
				rsDec.Process(writer, reader)
				decodedData = writer.Items()
			default:
				return fmt.Errorf("invalid ecc: %s (must be none, hamming, or rs)", ecc)
			}
		}

		if rx, ok := any(maskDecoder).(modulation.Receiver); ok {
			stats := rx.Transmission()
			fmt.Printf("  Detector lock: %v (locks=%d unlocks=%d)\n", stats.IsLocked, stats.LockCount, stats.UnlockCount)
		}
		if isASCII(decodedData) {
			fmt.Printf("  Decoded: %s\n", string(decodedData))
		} else {
			fmt.Printf("  Decoded (hex): %s\n", hex.EncodeToString(decodedData))
		}
		return nil
	}

	onFrame := func(floatData []float32) error {
		if len(floatData) == 0 {
			return nil
		}
		signalVec := vec.NewFrom(floatData...)
		rmsStats := dsp.CalculateRMSStatistics(signalVec)
		fmt.Printf("  Frame RMS: %.2f dB\n", rmsStats.AvgRMSDb)
		return nil
	}

	cfgStream := demodProcessConfig{
		BitsPerSymbol: bitsPerSymbol,
		Channels:      1,
		OnDecoded:     onDecoded,
		OnFrame:       onFrame,
	}

	if err := demodulateAudio(unmarshaller, maskDecoder, ctx, cfgStream); err != nil {
		return err
	}

	if rx, ok := any(maskDecoder).(modulation.Receiver); ok {
		quality := rx.Quality()
		stats := rx.Transmission()
		fmt.Printf("Final SNR: %.2f dB, Signal Strength: %.2f dB\n", quality.SNR, quality.SignalStrength)
		fmt.Printf("Total locks: %d, unlocks: %d\n", stats.LockCount, stats.UnlockCount)
	}
	return nil
}

