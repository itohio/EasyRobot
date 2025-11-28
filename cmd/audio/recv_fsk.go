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
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvFSKCmd() *cobra.Command {
	var (
		deviceID        string
		input           string
		frequencies     string
		sampleRate      int
		symbolRate      float64
		modulationOrder int
		decoding        string
		rsN             int
		rsK             int
	)

	cmd := &cobra.Command{
		Use:   "recv-fsk",
		Short: "Receive FSK modulated data from audio device or WAV file",
		Long:  "Listens for FSK modulated signals and decodes them, printing received data. Supports Hamming/Reed-Solomon/Gray decoding.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvFSK(deviceID, input, frequencies, sampleRate, symbolRate, modulationOrder, decoding, rsN, rsK)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().StringVarP(&frequencies, "frequencies", "f", "1000,2000", "Frequencies in Hz (comma-separated, 2-16 frequencies, default: 1000,2000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (2, 4, 8, 16, default: 4)")
	cmd.Flags().StringVar(&decoding, "decoding", "none", "Data decoding: \"none\", \"gray\", \"hamming\", or \"rs\" (default: none)")
	cmd.Flags().IntVar(&rsN, "rs-n", 255, "Reed-Solomon total codeword length (n, default: 255)")
	cmd.Flags().IntVar(&rsK, "rs-k", 223, "Reed-Solomon data symbols per codeword (k, default: 223)")

	return cmd
}

func runRecvFSK(deviceID, input, frequencies string, sampleRate int, symbolRate float64, modulationOrder int, decoding string, rsN, rsK int) error {
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

	// Convert to float32
	freqs32 := make([]float32, len(freqs))
	for i, f := range freqs {
		freqs32[i] = float32(f)
	}

	spacing := freqs32[1] - freqs32[0]
	for i := 2; i < len(freqs32); i++ {
		if math.Abs(float64((freqs32[i]-freqs32[i-1])-spacing)) > 1e-3 {
			return fmt.Errorf("frequencies must be evenly spaced for the current demodulator, got %v", freqs)
		}
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
			spacing := freqs32[1] - freqs32[0]
			freqs32 = make([]float32, modulationOrder)
			for i := range modulationOrder {
				freqs32[i] = freqs32[0] + float32(i)*spacing
			}
		} else {
			return fmt.Errorf("number of frequencies (%d) must match modulation order (%d)", len(freqs), modulationOrder)
		}
	}
	
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	if bitsPerSymbol <= 0 {
		return fmt.Errorf("modulation order %d is not a power of two", modulationOrder)
	}

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

	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid FSK demod config: %w", err)
	}

	fskDecoder, err := mfsk.NewDemodulator(cfg)
	if err != nil {
		return fmt.Errorf("failed to create FSK demodulator: %w", err)
	}

	// Create decoded demodulator if decoding is requested
	decodedDemod, decodingName, err := createDecodedDemodulator(fskDecoder, decoding, rsN, rsK)
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
		// Read from WAV file
		fmt.Printf("Reading FSK signal from WAV file: %s\n", input)
		unmarshaller = audio.NewUnmarshaller(audio.WithWAVPath(input))
	} else {
		// Read from audio device
		fmt.Printf("Listening for FSK signals on audio device...\n")
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
		if rx, ok := any(fskDecoder).(modulation.Receiver); ok {
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

	if err := demodulateAudio(unmarshaller, decodedDemod, ctx, cfgStream); err != nil {
		return err
	}

	if rx, ok := any(fskDecoder).(modulation.Receiver); ok {
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
