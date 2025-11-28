package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	marshalleraudio "github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/spf13/cobra"
)

func newPlayCmd() *cobra.Command {
	var (
		input     string
		deviceIDs []string
		channels  int
	)

	cmd := &cobra.Command{
		Use:   "play",
		Short: "Play WAV file to audio device",
		Long:  "Read a WAV file and play it to a playback device(s).",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runPlay(input, deviceIDs, channels)
		},
	}

	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file path (required)")
	cmd.Flags().StringSliceVarP(&deviceIDs, "device-id", "o", nil, "Playback device ID (repeatable/comma-separated; uses default if not specified)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 0, "Number of channels per device (0 = use WAV channels or device max)")

	cmd.MarkFlagRequired("input")

	return cmd
}

func runPlay(input string, deviceIDs []string, channelsPerDevice int) error {
	if input == "" {
		return fmt.Errorf("input file path is required")
	}
	if channelsPerDevice < 0 {
		return fmt.Errorf("channels must be non-negative (0 = use WAV channels or device max)")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	go func() {
		select {
		case <-sigChan:
			fmt.Println("\nStopping playback...")
			cancel()
		case <-ctx.Done():
		}
	}()

	devices, err := fetchAudioDevices()
	if err != nil {
		return err
	}

	// Resolve playback devices
	requestedIDs := collectDeviceIDs(deviceIDs)
	var playbackDevices []types.AudioDeviceInfo
	if len(requestedIDs) == 0 {
		// Use default
		dev, err := selectDevice(devices, "", "playback")
		if err != nil {
			return err
		}
		playbackDevices = []types.AudioDeviceInfo{dev}
	} else {
		// Use requested devices
		for _, id := range requestedIDs {
			dev, err := selectDevice(devices, id, "playback")
			if err != nil {
				return fmt.Errorf("failed to find playback device %s: %w", id, err)
			}
			playbackDevices = append(playbackDevices, dev)
		}
	}

	// Read WAV file
	unmarshaller := marshalleraudio.NewUnmarshaller(marshalleraudio.WithWAVPath(input))
	var tensor types.Tensor
	if err := unmarshaller.Unmarshal(nil, &tensor); err != nil {
		return fmt.Errorf("failed to read WAV file: %w", err)
	}
	defer tensor.Release()

	wavSampleRate := unmarshaller.SampleRate()
	if wavSampleRate <= 0 {
		wavSampleRate = 48000
	}
	wavChannels := unmarshaller.Channels()
	if wavChannels <= 0 {
		if shape := tensor.Shape(); len(shape) >= 2 {
			wavChannels = shape[1]
		} else {
			wavChannels = 1
		}
	}

	frames := 0
	if shape := tensor.Shape(); len(shape) >= 1 {
		frames = shape[0]
	}
	duration := float64(frames) / float64(wavSampleRate)

	// Print playback info
	fmt.Printf("Playing %s", input)
	for i, dev := range playbackDevices {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%s [ID=%s]", dev.Name, dev.ID)
	}
	fmt.Printf(" (%d channels @ %d Hz)\n", wavChannels, wavSampleRate)
	if duration > 0 {
		fmt.Printf("Duration: %.2f seconds (%d frames)\n", duration, frames)
	}
	fmt.Println("Press Ctrl+C to stop playback")

	// Configure marshaller with multiple devices
	marshallerOpts := []types.Option{
		marshalleraudio.WithSampleRate(wavSampleRate),
	}

	// Add devices with per-device channel configuration
	for _, dev := range playbackDevices {
		marshallerOpts = append(marshallerOpts, marshalleraudio.WithPlaybackDevice(dev.ID))
		if channelsPerDevice > 0 {
			// Per-device channels will be handled in marshaller
			marshallerOpts = append(marshallerOpts, marshalleraudio.WithChannels(channelsPerDevice))
		}
	}

	marshaller := marshalleraudio.NewMarshaller(marshallerOpts...)
	start := time.Now()

	// Marshal tensor directly (marshaller supports Tensor)
	// Marshal returns immediately after writing all data to the device buffer
	if err := marshaller.Marshal(nil, tensor, types.WithContext(ctx)); err != nil && !errors.Is(err, context.Canceled) {
		return fmt.Errorf("playback failed: %w", err)
	}

	// Wait for playback to complete: calculate expected duration and wait
	expectedDuration := time.Duration(frames) * time.Second / time.Duration(wavSampleRate)
	// Add small buffer for latency
	waitDuration := expectedDuration + 100*time.Millisecond

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(waitDuration):
		// Playback should be complete
	}

	fmt.Printf("Playback finished in %s\n", time.Since(start).Truncate(time.Millisecond))
	return nil
}

func formatFromTensor(dtype tensortypes.DataType) string {
	switch dtype {
	case tensortypes.UINT8:
		return "U8"
	case tensortypes.INT8:
		return "U8"
	case tensortypes.INT16:
		return "S16"
	case tensortypes.INT32:
		return "S32"
	case tensortypes.FP32:
		return "F32"
	default:
		return ""
	}
}
