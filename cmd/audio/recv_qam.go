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
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/qam"
	"github.com/itohio/EasyRobot/x/math/filter/fir"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvQAMCmd() *cobra.Command {
	var (
		deviceID   string
		input      string
		frequency  float64
		sampleRate int
		symbolRate float64
	)

	cmd := &cobra.Command{
		Use:   "recv-qam",
		Short: "Receive QAM modulated data from audio device or WAV file",
		Long:  "Listens for QAM modulated signals and decodes them, printing received data.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvQAM(deviceID, input, frequency, sampleRate, symbolRate)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")

	return cmd
}

func runRecvQAM(deviceID, input string, frequency float64, sampleRate int, symbolRate float64) error {
	// Create QAM decoder
	qamDecoder := qam.NewDecoder(float32(sampleRate), float32(frequency), float32(symbolRate))

	// Set up low-pass filter for baseband filtering
	cutoffFreq := float32(symbolRate) / 2.0
	lowPass := fir.NewLowPass(32, cutoffFreq, float32(sampleRate))
	qamDecoder.SetFilter(lowPass)

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
		fmt.Printf("Reading QAM signal from WAV file: %s\n", input)
		unmarshaller = audio.NewUnmarshaller(audio.WithWAVPath(input))
	} else {
		fmt.Printf("Listening for QAM signals on audio device...\n")
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
	return readAndProcess(unmarshaller, qamDecoder, ctx, func(decodedData []byte) error {
		if len(decodedData) == 0 {
			return nil
		}

		// Print decoded user data directly
		fmt.Printf("  Detector lock: %v\n", qamDecoder.IsLocked())
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
