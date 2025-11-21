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
	"github.com/itohio/EasyRobot/x/math/filter/fir"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvPSKCmd() *cobra.Command {
	var (
		deviceID       string
		input          string // WAV file input (alternative to device)
		frequency      float64
		sampleRate     int
		symbolRate     float64
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

	// Create protocol unmarshaller
	protocol := audio.NewProtocolUnmarshaller()

	// Create PSK decoder
	pskDecoder := dsp.NewPSKDecoder(float32(sampleRate), float32(frequency), float32(symbolRate), modulationOrder)

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

	if input != "" {
		// Read from WAV file
		return recvPSKFromWAV(input, pskDecoder, protocol, ctx, modulationOrder)
	} else {
		// Read from audio device
		return recvPSKFromDevice(deviceID, sampleRate, pskDecoder, protocol, ctx, modulationOrder)
	}
}

func recvPSKFromWAV(input string, decoder *dsp.PSKDecoder, protocol *audio.ProtocolUnmarshaller, ctx context.Context, modulationOrder int) error {
	fmt.Printf("Reading PSK signal from WAV file: %s\n", input)
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

	var floatData []float32
	switch data := tensorData.(type) {
	case []int16:
		floatData = make([]float32, len(data))
		for i, v := range data {
			floatData[i] = float32(v) / 32768.0
		}
	case []int32:
		floatData = make([]float32, len(data))
		for i, v := range data {
			floatData[i] = float32(v) / 2147483648.0
		}
	case []float32:
		floatData = data
	case []uint8:
		floatData = make([]float32, len(data))
		for i, v := range data {
			floatData[i] = (float32(v) - 128.0) / 128.0
		}
	default:
		return fmt.Errorf("unsupported tensor data type: %T", tensorData)
	}

	// Demodulate signal to symbols
	signalVec := vec.NewFrom(floatData...)
	symbols := decoder.Demodulate(signalVec)

	// Convert symbols to bytes
	// For BPSK: each symbol is 0 or 1 (1 bit)
	// For QPSK: each symbol is 0, 1, 2, or 3 (2 bits)
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	symbolData := symbols.View().(vec.Vector)

	// Convert symbols to bits, then to bytes
	bitBuffer := make([]byte, 0, len(symbolData)*bitsPerSymbol)
	for i := 0; i < len(symbolData); i++ {
		symbol := int(symbolData[i])
		for j := bitsPerSymbol - 1; j >= 0; j-- {
			bit := (symbol >> j) & 1
			bitBuffer = append(bitBuffer, byte(bit))
		}
	}

	// Convert bits to bytes
	decodedBytes := make([]byte, 0, len(bitBuffer)/8)
	for i := 0; i < len(bitBuffer)-7; i += 8 {
		byteVal := byte(0)
		for j := 0; j < 8; j++ {
			byteVal |= bitBuffer[i+j] << uint(7-j)
		}
		decodedBytes = append(decodedBytes, byteVal)
	}

	// Try to extract protocol packet
	packet, _, err := protocol.UnmarshalData(decodedBytes)
	if err != nil {
		return fmt.Errorf("failed to unmarshal protocol: %w", err)
	}

	if packet != nil {
		fmt.Printf("\nReceived packet:\n")
		fmt.Printf("  Length: %d bytes\n", len(packet))
		fmt.Printf("  Hex: %s\n", hex.EncodeToString(packet))
		fmt.Printf("  Text: %s\n", string(packet))
	} else {
		fmt.Println("\nNo complete packet found in WAV file")
		fmt.Printf("  Decoded %d bytes (may be incomplete)\n", len(decodedBytes))
	}

	return nil
}

func recvPSKFromDevice(deviceID string, sampleRate int, decoder *dsp.PSKDecoder, protocol *audio.ProtocolUnmarshaller, ctx context.Context, modulationOrder int) error {
	fmt.Printf("Listening for PSK signals on audio device...\n")
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

	// Buffer for decoded bytes
	bitBuffer := make([]byte, 0, 4096*8)
	packetCount := 0

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

			var floatData []float32
			switch data := frameData.(type) {
			case []int16:
				floatData = make([]float32, len(data))
				for i, v := range data {
					floatData[i] = float32(v) / 32768.0
				}
			case []float32:
				floatData = data
			default:
				continue // Skip unsupported types
			}

			// Demodulate frame to symbols
			signalVec := vec.NewFrom(floatData...)
			symbols := decoder.Demodulate(signalVec)
			symbolData := symbols.View().(vec.Vector)

			// Convert symbols to bits
			bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
			for i := 0; i < len(symbolData); i++ {
				symbol := int(symbolData[i])
				for j := bitsPerSymbol - 1; j >= 0; j-- {
					bit := (symbol >> j) & 1
					bitBuffer = append(bitBuffer, byte(bit))
				}
			}

			// Convert bits to bytes when we have enough
			if len(bitBuffer) >= 8 {
				decodedBytes := make([]byte, 0, len(bitBuffer)/8)
				for i := 0; i < len(bitBuffer)-7; i += 8 {
					byteVal := byte(0)
					for j := 0; j < 8; j++ {
						byteVal |= bitBuffer[i+j] << uint(7-j)
					}
					decodedBytes = append(decodedBytes, byteVal)
				}

				// Try to extract protocol packet
				packet, _, err := protocol.UnmarshalData(decodedBytes)
				if err == nil && packet != nil {
					packetCount++
					fmt.Printf("\n[Packet %d]\n", packetCount)
					fmt.Printf("  Length: %d bytes\n", len(packet))
					fmt.Printf("  Hex: %s\n", hex.EncodeToString(packet))
					fmt.Printf("  Text: %s\n", string(packet))
					fmt.Println()

					// Reset for next packet
					bitBuffer = bitBuffer[:0]
					protocol.Reset()
				} else if len(decodedBytes) > 0 {
					// Keep remaining bits for next iteration
					remainingBits := len(bitBuffer) % 8
					if remainingBits > 0 {
						bitBuffer = bitBuffer[len(bitBuffer)-remainingBits:]
					} else {
						bitBuffer = bitBuffer[:0]
					}
				}
			}
		}
	}
}

