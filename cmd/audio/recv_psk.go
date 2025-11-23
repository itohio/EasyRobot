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
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mpsk"
	"github.com/itohio/EasyRobot/x/math/filter/fir"
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
	)

	cmd := &cobra.Command{
		Use:   "recv-psk",
		Short: "Receive PSK modulated data from audio device or WAV file",
		Long:  "Listens for PSK modulated signals and decodes them, printing received data.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvPSK(deviceID, input, frequency, sampleRate, symbolRate, modulationOrder)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVar(&modulationOrder, "modulation-order", 2, "Modulation order (2=BPSK, 4=QPSK, default: 2)")

	return cmd
}

func runRecvPSK(deviceID, input string, frequency float64, sampleRate int, symbolRate float64, modulationOrder int) error {
	// Validate modulation order
	if modulationOrder != 2 && modulationOrder != 4 && modulationOrder != 8 && modulationOrder != 16 {
		return fmt.Errorf("modulation order must be 2, 4, 8, or 16, got %d", modulationOrder)
	}

	// Create PSK decoder
	pskDecoder := mpsk.NewDemodulator(float32(sampleRate), float32(frequency), float32(symbolRate), modulationOrder)

	// Set up low-pass filter for baseband filtering
	cutoffFreq := float32(symbolRate) / 2.0
	lowPass := fir.NewLowPass(32, cutoffFreq, float32(sampleRate))
	pskDecoder.SetFilter(lowPass)

	// Note: PSKDecoder doesn't implement ModulatorDecoder interface yet
	// We'll use Demodulate() and convert symbols to bytes manually

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

	// Use generic readAndProcess function - work directly with decoded user data
	return readAndProcess(unmarshaller, pskDecoder, ctx, func(decodedData []byte) error {
		if len(decodedData) == 0 {
			return nil
		}

		// Print decoded user data directly
		fmt.Printf("  Detector lock: %v\n", pskDecoder.IsLocked())
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
