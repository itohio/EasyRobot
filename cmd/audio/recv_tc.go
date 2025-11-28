package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/tc"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvTCCmd() *cobra.Command {
	var (
		deviceID   string
		input      string
		frameRate  string
		sampleRate int
	)

	cmd := &cobra.Command{
		Use:   "recv-tc",
		Short: "Receive timecode from audio device or WAV file",
		Long:  "Listens for SMPTE timecode signals (LTC) and decodes them, printing received timecode.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvTC(deviceID, input, frameRate, sampleRate)
		},
	}

	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().StringVar(&frameRate, "frame-rate", "30", "Frame rate: \"24\", \"25\", \"29.97\", or \"30\" (default: 30)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")

	return cmd
}

func runRecvTC(deviceID, input, frameRate string, sampleRate int) error {
	// Parse frame rate
	var tcFrameRate tc.FrameRate
	switch frameRate {
	case "24":
		tcFrameRate = tc.FrameRate24
	case "25":
		tcFrameRate = tc.FrameRate25
	case "29.97":
		tcFrameRate = tc.FrameRate2997
	case "30":
		tcFrameRate = tc.FrameRate30
	default:
		return fmt.Errorf("invalid frame rate: %s (use 24, 25, 29.97, or 30)", frameRate)
	}

	// Create LTC demodulator
	ltcDemodulator := tc.NewLTCDemodulator(float32(sampleRate), tcFrameRate)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		fmt.Println("\nStopping...")
		cancel()
	}()

	if input != "" {
		// Read from WAV file
		return recvTCFromWAV(input, ltcDemodulator, ctx)
	} else {
		// Read from audio device
		return recvTCFromDevice(deviceID, sampleRate, ltcDemodulator, ctx)
	}
}

func recvTCFromWAV(input string, decoder *tc.LTCDemodulator, ctx context.Context) error {
	fmt.Printf("Reading timecode signal from WAV file: %s\n", input)
	fmt.Println("Press Ctrl+C to stop")

	// Open WAV file
	unmarshaller := audio.NewUnmarshaller(audio.WithWAVPath(input))
	var tensor types.Tensor
	err := unmarshaller.Unmarshal(nil, &tensor)
	if err != nil {
		return fmt.Errorf("failed to read WAV file: %w", err)
	}

	// Convert tensor to float32 vector for decoding
	tensorData := tensor.Data()
	convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, tensorData)
	if convertedData == nil {
		return fmt.Errorf("failed to convert tensor data to float32")
	}
	floatData, ok := convertedData.([]float32)
	if !ok {
		return fmt.Errorf("converted data is not []float32: %T", convertedData)
	}

	// Decode signal to timecode
	signalVec := vec.NewFrom(floatData...)
	timecodes := make([]*tc.SMPTETimeCode, 1)
	timecodes[0] = &tc.SMPTETimeCode{}
	framesDecoded := decoder.Decode(timecodes, signalVec)

	if framesDecoded == 0 {
		fmt.Println("\nNo timecode decoded from WAV file")
		return nil
	}

	tc := timecodes[0]
	fmt.Printf("\nReceived timecode:\n")
	fmt.Printf("  Timecode: %02d:%02d:%02d:%02d\n", tc.Hours, tc.Minutes, tc.Seconds, tc.Frames)
	fmt.Printf("  Frame rate: %.2f fps\n", float32(tc.FrameRate))
	fmt.Printf("  Drop frame: %v\n", tc.DropFrame)
	fmt.Printf("  Color frame: %v\n", tc.ColorFrame)

	return nil
}

func recvTCFromDevice(deviceID string, sampleRate int, decoder *tc.LTCDemodulator, ctx context.Context) error {
	fmt.Printf("Listening for timecode signals on audio device...\n")
	fmt.Println("Press Ctrl+C to stop")

	// Configure unmarshaller for device capture
	opts := []types.Option{
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(1),
	}
	if deviceID != "" {
		opts = append(opts, audio.WithCaptureDevice(deviceID))
	}

	unmarshaller := audio.NewUnmarshaller(opts...)

	// Create frame stream
	var stream types.FrameStream
	err := unmarshaller.Unmarshal(nil, &stream)
	if err != nil {
		return fmt.Errorf("failed to start audio capture: %w", err)
	}
	defer stream.Close()

	// Buffer for timecodes
	timecodes := make([]*tc.SMPTETimeCode, 1)
	timecodes[0] = &tc.SMPTETimeCode{}
	frameCount := 0

	// Process frames from channel
	for {
		select {
		case <-ctx.Done():
			return nil
		case frame, ok := <-stream.C:
			if !ok {
				return nil // Channel closed
			}

			// Convert frame to float32 vector
			if len(frame.Tensors) == 0 {
				continue
			}
			frameTensor := frame.Tensors[0]
			frameData := frameTensor.Data()

			// Convert frame tensor data to float32
			convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, frameData)
			if convertedData == nil {
				frameTensor.Release()
				continue
			}
			floatData, ok := convertedData.([]float32)
			if !ok {
				frameTensor.Release()
				continue
			}
			frameTensor.Release()

			// Decode frame to timecode
			signalVec := vec.NewFrom(floatData...)
			framesDecoded := decoder.DecodeStream(timecodes, signalVec)

			if framesDecoded > 0 {
				tc := timecodes[0]
				frameCount++
				fmt.Printf("\n[Frame %d]\n", frameCount)
				fmt.Printf("  Timecode: %02d:%02d:%02d:%02d\n", tc.Hours, tc.Minutes, tc.Seconds, tc.Frames)
				fmt.Printf("  Frame rate: %.2f fps\n", float32(tc.FrameRate))
				fmt.Printf("  Drop frame: %v\n", tc.DropFrame)
				fmt.Printf("  Color frame: %v\n", tc.ColorFrame)
				fmt.Println()
			}
		}
	}
}
