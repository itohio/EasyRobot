package main

import (
	"context"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mpsk"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvPSKCmd() *cobra.Command {
	var (
		deviceID        string
		input           string // WAV file input (alternative to device)
		frequency       float64
		sampleRate      int
		symbolRate      float64
		modulationOrder int
		decoding        string
		rsN             int
		rsK             int
	)

	cmd := &cobra.Command{
		Use:   "recv-psk",
		Short: "Receive PSK modulated data from audio device or WAV file",
		Long:  "Listens for PSK modulated signals and decodes them, printing received data. Supports Hamming/Reed-Solomon/Gray decoding.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvPSK(deviceID, input, frequency, sampleRate, symbolRate, modulationOrder, decoding, rsN, rsK)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 5000, "Carrier frequency in Hz (default: 5000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (2=BPSK, 4=QPSK, 8=8-PSK, 16=16-PSK, default: 4)")
	cmd.Flags().StringVar(&decoding, "decoding", "none", "Data decoding: \"none\", \"gray\", \"hamming\", or \"rs\" (default: none)")
	cmd.Flags().IntVar(&rsN, "rs-n", 255, "Reed-Solomon total codeword length (n, default: 255)")
	cmd.Flags().IntVar(&rsK, "rs-k", 223, "Reed-Solomon data symbols per codeword (k, default: 223)")

	return cmd
}

func runRecvPSK(deviceID, input string, frequency float64, sampleRate int, symbolRate float64, modulationOrder int, decoding string, rsN, rsK int) error {
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

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid PSK demod config: %w", err)
	}

	pskDecoder, err := mpsk.NewDemodulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to create PSK demodulator: %w", err)
	}

	// Create decoded demodulator if decoding is requested
	decodedDemod, decodingName, err := createDecodedDemodulator(pskDecoder, decoding, rsN, rsK)
	if err != nil {
		return fmt.Errorf("failed to create decoded demodulator: %w", err)
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
		fmt.Printf("Reading PSK signal from WAV file: %s\n", input)
		unmarshaller = audio.NewUnmarshaller(audio.WithWAVPath(input))
	} else {
		fmt.Printf("Listening for PSK signals on audio device...\n")
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
		if rx, ok := any(pskDecoder).(modulation.Receiver); ok {
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
		Channels:      2,
		OnDecoded:     onDecoded,
		OnFrame:       onFrame,
	}

	if err := demodulateAudio(unmarshaller, decodedDemod, ctx, cfgStream); err != nil {
		return err
	}

	// Try to get receiver stats from the underlying demodulator
	if rx, ok := any(pskDecoder).(modulation.Receiver); ok {
		quality := rx.Quality()
		stats := rx.Transmission()
		fmt.Printf("Final SNR: %.2f dB, Signal Strength: %.2f dB\n", quality.SNR, quality.SignalStrength)
		fmt.Printf("Total locks: %d, unlocks: %d\n", stats.LockCount, stats.UnlockCount)
	}
	if decodingName != "none" {
		fmt.Printf("Decoding: %s\n", decodingName)
	}
	return nil
}
