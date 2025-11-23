package main

import (
	"context"
	"encoding/hex"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvFSKCmd() *cobra.Command {
	var (
		deviceID    string
		input       string
		frequencies string
		sampleRate  int
		symbolRate  float64
	)

	cmd := &cobra.Command{
		Use:   "recv-fsk",
		Short: "Receive FSK modulated data from audio device or WAV file",
		Long:  "Listens for FSK modulated signals and decodes them, printing received data.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvFSK(deviceID, input, frequencies, sampleRate, symbolRate)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().StringVarP(&frequencies, "frequencies", "f", "1000,2000", "Frequencies in Hz (comma-separated, 2-16 frequencies, default: 1000,2000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")

	return cmd
}

func runRecvFSK(deviceID, input, frequencies string, sampleRate int, symbolRate float64) error {
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

	// Create FSK decoder
	fskDecoder := mfsk.NewMDemodulator(float32(sampleRate), float32(symbolRate), freqs32...)

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

	// Use generic readAndProcess function - work directly with decoded user data
	return readAndProcess(unmarshaller, fskDecoder, ctx, func(decodedData []byte) error {
		if len(decodedData) == 0 {
			return nil
		}

		// Print decoded user data directly
		fmt.Printf("  Detector lock: %v\n", fskDecoder.IsLocked())
		if isASCII(decodedData) {
			fmt.Printf("  Decoded: %s\n", string(decodedData))
		} else {
			fmt.Printf("  Decoded (hex): %s\n", hex.EncodeToString(decodedData))
		}
		return nil
	}, func(floatData []float32) error {
		// Calculate RMS statistics per frame
		if len(floatData) > 0 {
			signalVec := vec.NewFrom(floatData...)
			rmsStats := dsp.CalculateRMSStatistics(signalVec)
			fmt.Printf("  Frame RMS: %.2f dB\n", rmsStats.AvgRMSDb)
		}
		return nil
	})
}
