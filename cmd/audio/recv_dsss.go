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
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/dsss"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvDSSSCmd() *cobra.Command {
	var (
		deviceID        string
		input           string
		frequency       float64
		sampleRate      int
		symbolRate      float64
		spreadingFactor int
		chips           int
		codes           []string
	)

	cmd := &cobra.Command{
		Use:   "recv-dsss",
		Short: "Receive DSSS modulated data from audio device or WAV file",
		Long:  "Listens for DSSS modulated signals and decodes them, printing received data.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvDSSS(deviceID, input, frequency, sampleRate, symbolRate, spreadingFactor, chips, codes)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 1000, "Carrier frequency in Hz (default: 1000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVar(&spreadingFactor, "spreading-factor", 31, "Spreading factor (chips per symbol, default: 31)")
	cmd.Flags().IntVar(&chips, "chips", 0, "Code length in chips (required when using --code, inferred from hex string if not specified)")
	cmd.Flags().StringSliceVarP(&codes, "code", "c", []string{}, "Spreading code as hex string (can be specified multiple times or comma-separated, e.g., --code A5,F3 or -c A5 -c F3 for 2-ary DSSS, must match encoder)")

	return cmd
}

func runRecvDSSS(deviceID, input string, frequency float64, sampleRate int, symbolRate float64, spreadingFactor, chips int, codes []string) error {
	// Parse spreading codes from hex strings (must match encoder)
	if len(codes) == 0 {
		return fmt.Errorf("--code is required (can specify multiple for M-ary DSSS, e.g., --code A5 --code F3)")
	}

	// Determine code length: use --chips if provided, otherwise infer from hex string
	codeLength := chips
	if codeLength == 0 {
		// Infer code length from first hex string (all codes should have same length)
		// Hex string length: 2 hex chars = 1 byte = 8 bits
		// Default: use all bits from decoded hex bytes
		hexBytes, err := hex.DecodeString(codes[0])
		if err != nil {
			return fmt.Errorf("invalid hex code: %s: %w", codes[0], err)
		}
		codeLength = len(hexBytes) * 8
		if codeLength == 0 {
			return fmt.Errorf("invalid hex code: %s (must have at least 2 hex characters). Use --chips to specify code length explicitly", codes[0])
		}
	}

	if codeLength <= 0 {
		return fmt.Errorf("code length (--chips) must be positive, got %d", codeLength)
	}

	// Parse all hex codes
	dsssCodes := make([]dsss.Code, len(codes))
	var err error
	for i, hexStr := range codes {
		dsssCodes[i], err = dsss.CodeFromHex(hexStr, codeLength)
		if err != nil {
			return fmt.Errorf("failed to parse code %d from hex %s: %w", i, hexStr, err)
		}
		if dsssCodes[i].Length() != codeLength {
			return fmt.Errorf("code %d has length %d, expected %d", i, dsssCodes[i].Length(), codeLength)
		}
	}

	// Validation will be done by demodulator constructor
	fmt.Printf("  Code length: %d chips\n", codeLength)
	fmt.Printf("  Modulation order: %d\n", len(dsssCodes))

	// Create DSSS decoder with varargs codes
	dsssDecoder := dsss.NewDemodulator(float32(sampleRate), float32(frequency), float32(symbolRate), dsssCodes...)

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
		fmt.Printf("Reading DSSS signal from WAV file: %s\n", input)
		unmarshaller = audio.NewUnmarshaller(audio.WithWAVPath(input))
	} else {
		fmt.Printf("Listening for DSSS signals on audio device...\n")
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
	return readAndProcess(unmarshaller, dsssDecoder, ctx, func(decodedData []byte) error {
		if len(decodedData) == 0 {
			return nil
		}

		// Print decoded user data directly
		fmt.Printf("  Detector lock: %v\n", dsssDecoder.IsLocked())
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
