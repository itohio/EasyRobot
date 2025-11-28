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
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/dsss"
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
		modulationOrder int
	)

	cmd := &cobra.Command{
		Use:   "recv-dsss",
		Short: "Receive DSSS modulated data from audio device or WAV file",
		Long:  "Listens for DSSS modulated signals and decodes them, printing received data.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvDSSS(deviceID, input, frequency, sampleRate, symbolRate, spreadingFactor, chips, codes, modulationOrder)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 10000, "Carrier frequency in Hz (default: 10000)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 96000, "Sample rate (Hz)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (default: 4)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 1000, "Symbol rate in baud (default: 1000)")
	cmd.Flags().IntVar(&spreadingFactor, "spreading-factor", 31, "Spreading factor (chips per symbol, default: 31)")
	cmd.Flags().IntVar(&chips, "chips", 0, "Code length in chips (required when using --code, inferred from hex string if not specified)")
	cmd.Flags().StringSliceVarP(&codes, "code", "c", []string{}, "Spreading code as hex string (can be specified multiple times or comma-separated, e.g., --code A5,F3 or -c A5 -c F3 for 2-ary DSSS, must match encoder)")

	return cmd
}

func runRecvDSSS(deviceID, input string, frequency float64, sampleRate int, symbolRate float64, spreadingFactor, chips int, codes []string, modulationOrder int) error {
	// Parse spreading codes from hex strings (must match encoder)
	if len(codes) == 0 {
		return fmt.Errorf("--code is required (can specify multiple for M-ary DSSS, e.g., --code A5 --code F3)")
	}
	
	// Ensure number of codes matches modulation order
	if len(codes) != modulationOrder {
		return fmt.Errorf("number of codes (%d) must match modulation order (%d)", len(codes), modulationOrder)
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
			codeLength = 16 // Default key size is 16 bits
		}
	}

	if codeLength <= 0 {
		return fmt.Errorf("code length (--chips) must be positive, got %d", codeLength)
	}

	// Parse all hex codes to [][]int8
	dsssCodes := make([][]int8, len(codes))
	for i, hexStr := range codes {
		hexBytes, err := hex.DecodeString(hexStr)
		if err != nil {
			return fmt.Errorf("invalid hex code %d: %s: %w", i, hexStr, err)
		}
		// Convert bytes to bipolar int8 codes (-1, +1)
		code := make([]int8, codeLength)
		for j := 0; j < codeLength && j < len(hexBytes)*8; j++ {
			byteIdx := j / 8
			bitIdx := j % 8
			if (hexBytes[byteIdx]>>bitIdx)&1 == 1 {
				code[j] = 1
			} else {
				code[j] = -1
			}
		}
		dsssCodes[i] = code
	}

	// Calculate bits per symbol from modulation order
	bitsPerSymbol := int(math.Log2(float64(len(dsssCodes))))
	if bitsPerSymbol == 0 {
		bitsPerSymbol = 1
	}

	// Create DSSS config
	config := dsss.Config{
		BitsPerSymbol:   bitsPerSymbol,
		ModulationOrder: len(dsssCodes),
		Codes:           dsssCodes,
		SpreadingFactor: spreadingFactor,
		SampleRate:      float32(sampleRate),
		SymbolRate:      float32(symbolRate),
		CarrierFreq:     float32(frequency),
		PulseShape:      "raised_cosine",
		Rolloff:         0.35,
		PulseFilterSpan: 4,
		PreambleLength:  0,
		Amplitude:       1.0,
	}

	// Validation will be done by demodulator constructor
	fmt.Printf("  Code length: %d chips\n", codeLength)
	fmt.Printf("  Modulation order: %d\n", len(dsssCodes))

	// Create DSSS decoder
	dsssDecoder, err := dsss.NewDemodulator(config)
	if err != nil {
		return fmt.Errorf("failed to create DSSS demodulator: %w", err)
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

	// Use generic demodulateAudio function - work directly with decoded user data
	return demodulateAudio(unmarshaller, dsssDecoder, ctx, demodProcessConfig{
		BitsPerSymbol: bitsPerSymbol,
		Channels:      1,
		ChunkParts:    defaultChunkParts,
		OnDecoded: func(decodedData []byte) error {
			if len(decodedData) == 0 {
				return nil
			}

			// Print decoded user data directly
			if rx, ok := any(dsssDecoder).(modulation.Receiver); ok {
				stats := rx.Transmission()
				fmt.Printf("  Detector lock: %v (locks=%d unlocks=%d)\n", stats.IsLocked, stats.LockCount, stats.UnlockCount)
			}
			if isASCII(decodedData) {
				fmt.Printf("  Decoded: %s\n", string(decodedData))
			} else {
				fmt.Printf("  Decoded (hex): %s\n", hex.EncodeToString(decodedData))
			}
			return nil
		},
	})
}
