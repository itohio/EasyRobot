package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	marshalleraudio "github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/spf13/cobra"
)

func newRecordCmd() *cobra.Command {
	var (
		output       string
		deviceIDs    []string
		outputDevice string
		channels     int
		sampleRate   int
		format       string
	)

	cmd := &cobra.Command{
		Use:   "record",
		Short: "Record audio from device to WAV file",
		Long:  "Record audio from a capture device and save it to a WAV file.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecord(output, deviceIDs, outputDevice, channels, sampleRate, format)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringSliceVarP(&deviceIDs, "device-id", "i", nil, "Capture device ID (repeatable/comma-separated; uses default if not specified)")
	cmd.Flags().StringVar(&outputDevice, "output-device", "", "Optional playback device ID to monitor the recording")
	cmd.Flags().IntVarP(&channels, "channels", "c", 0, "Number of channels per device (0 = use device default)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().StringVarP(&format, "format", "f", "", "Format (S16, S32, F32, etc.) (optional)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runRecord(output string, inputDevices []string, monitorDevice string, channels, sampleRate int, format string) error {
	if output == "" {
		return fmt.Errorf("output file path is required")
	}
	if channels < 0 {
		return fmt.Errorf("channels must be non-negative (0 = use device default)")
	}
	if sampleRate <= 0 {
		return fmt.Errorf("sample rate must be positive")
	}

	format = normalizeFormat(format)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	go func() {
		select {
		case <-sigChan:
			fmt.Println("\nStopping recording...")
			cancel()
		case <-ctx.Done():
		}
	}()

	devices, err := fetchAudioDevices()
	if err != nil {
		return err
	}

	requestedIDs := collectDeviceIDs(inputDevices)
	fallbackSelector := ""
	if len(requestedIDs) == 1 {
		fallbackSelector = requestedIDs[0]
	}

	captureDevice, err := selectDevice(devices, fallbackSelector, "capture")
	if err != nil {
		return err
	}

	captureDevices, err := resolveCaptureDevices(devices, requestedIDs, captureDevice)
	if err != nil {
		return err
	}

	var playbackDevice *types.AudioDeviceInfo
	if monitorDevice != "" {
		dev, err := selectDevice(devices, monitorDevice, "playback")
		if err != nil {
			return err
		}
		playbackDevice = &dev
	}

	printCaptureBanner(captureDevices, output)
	if playbackDevice != nil {
		fmt.Printf("Monitoring on %s [ID=%s]\n", playbackDevice.Name, playbackDevice.ID)
	}
	fmt.Println("Press Ctrl+C to stop recording")

	unmarshaller := marshalleraudio.NewUnmarshaller()
	recordOpts := []types.Option{
		marshalleraudio.WithSampleRate(sampleRate),
		types.WithContext(ctx),
	}
	// Only set channels if explicitly requested (channels > 0)
	if channels > 0 {
		recordOpts = append(recordOpts, marshalleraudio.WithChannels(channels))
	}
	for _, dev := range captureDevices {
		recordOpts = append(recordOpts, marshalleraudio.WithCaptureDevice(dev.ID))
	}
	if format != "" {
		recordOpts = append(recordOpts, marshalleraudio.WithFormat(format))
	}

	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream, recordOpts...); err != nil {
		return fmt.Errorf("failed to start capture: %w", err)
	}
	defer stream.Close()

	var (
		outputCh                chan types.Frame
		outputMarshaller        *marshalleraudio.Marshaller
		outputWG                sync.WaitGroup
		outputErr               error
		actualChannels          int
		actualSampleRate        = sampleRate
		outputUnmarshallerDtype types.DataType
	)

	var playbackCh chan types.Frame
	var playbackWG sync.WaitGroup
	var playbackErr error
	var playbackCleanup func()
	startPlayback := func() {
		if playbackDevice == nil || playbackCh != nil || actualChannels == 0 {
			return
		}
		playbackCh = make(chan types.Frame, 16)
		playbackStream := types.NewFrameStream(playbackCh, nil)
		playbackMarshaller := marshalleraudio.NewMarshaller(
			marshalleraudio.WithPlaybackDevice(playbackDevice.ID),
			marshalleraudio.WithSampleRate(actualSampleRate),
			marshalleraudio.WithChannels(actualChannels),
			marshalleraudio.WithFormat(format),
		)

		playbackWG.Add(1)
		go func() {
			defer playbackWG.Done()
			if err := playbackMarshaller.Marshal(nil, playbackStream, types.WithContext(ctx)); err != nil && !errors.Is(err, context.Canceled) {
				playbackErr = fmt.Errorf("playback failed: %w", err)
			}
		}()
		playbackCleanup = func() {
			close(playbackCh)
			playbackWG.Wait()
		}
	}
	defer func() {
		if playbackCleanup != nil {
			playbackCleanup()
		}
	}()

	start := time.Now()
	totalFrames := 0

	for {
		select {
		case <-ctx.Done():
			goto finish
		case frame, ok := <-stream.C:
			if !ok {
				goto finish
			}
			if len(frame.Tensors) == 0 {
				continue
			}

			// Combine all tensors from the frame into a single tensor with all channels
			var combinedTensor types.Tensor
			if len(frame.Tensors) == 1 {
				combinedTensor = frame.Tensors[0]
			} else {
				// Multiple tensors: combine channels from all tensors
				var err error
				combinedTensor, err = combineTensors(frame.Tensors)
				if err != nil {
					return fmt.Errorf("failed to combine tensors: %w", err)
				}
			}

			if combinedTensor.Empty() {
				continue
			}

			if outputMarshaller == nil {
				shape := combinedTensor.Shape()
				if len(shape) != 2 || shape[1] == 0 {
					continue
				}
				actualChannels = shape[1]
				if actualSampleRate <= 0 {
					if sr := unmarshaller.SampleRate(); sr > 0 {
						actualSampleRate = sr
					} else {
						actualSampleRate = sampleRate
					}
				}
				outputUnmarshallerDtype = unmarshaller.DataType()
				if outputUnmarshallerDtype == types.DT_UNKNOWN {
					outputUnmarshallerDtype = types.INT16 // default
				}

				// Create output FrameStream and marshaller
				outputCh = make(chan types.Frame, 16)
				outputStream := types.NewFrameStream(outputCh, nil)

				outputMarshaller = marshalleraudio.NewMarshaller(
					marshalleraudio.WithWAVPath(output),
					marshalleraudio.WithSampleRate(actualSampleRate),
					marshalleraudio.WithChannels(actualChannels),
					marshalleraudio.WithFormat(format),
					types.WithDestinationType(outputUnmarshallerDtype),
				)

				outputWG.Add(1)
				go func() {
					defer outputWG.Done()
					if err := outputMarshaller.Marshal(nil, outputStream, types.WithContext(ctx)); err != nil && !errors.Is(err, context.Canceled) {
						outputErr = fmt.Errorf("failed to write WAV file: %w", err)
					}
				}()
				startPlayback()
			}
			if outputMarshaller == nil {
				continue
			}

			// Count frames from tensor shape
			shape := combinedTensor.Shape()
			if len(shape) >= 1 && shape[0] > 0 {
				totalFrames += shape[0]
			}

			// Send frame to output marshaller
			outputFrame := types.Frame{Tensors: []types.Tensor{combinedTensor}}
			select {
			case outputCh <- outputFrame:
			case <-ctx.Done():
				goto finish
			}

			if playbackCh != nil {
				// Create a copy of the tensor for playback since marshaller may consume/release it
				playbackTensor := cloneTensor(combinedTensor)
				if playbackTensor == nil {
					// If cloning fails, skip playback for this frame
					continue
				}
				playbackFrame := types.Frame{Tensors: []types.Tensor{playbackTensor}}
				select {
				case playbackCh <- playbackFrame:
				case <-ctx.Done():
					playbackTensor.Release()
					goto finish
				}
			}
		}
	}

finish:
	if playbackCleanup != nil {
		playbackCleanup()
		playbackCleanup = nil
	}
	if outputCh != nil {
		close(outputCh)
		outputWG.Wait()
	}

	if outputErr != nil {
		return outputErr
	}
	if playbackErr != nil {
		return playbackErr
	}

	if totalFrames == 0 {
		return fmt.Errorf("no audio frames captured")
	}

	durationSeconds := float64(totalFrames) / float64(actualSampleRate)
	fmt.Printf("Saved %.2f seconds (%d frames) to %s\n", durationSeconds, totalFrames, output)
	fmt.Printf("Recording duration: %s\n", time.Since(start).Truncate(time.Millisecond))

	return nil
}

func fetchAudioDevices() ([]types.AudioDeviceInfo, error) {
	unmarshaller := marshalleraudio.NewUnmarshaller()
	var devices []types.AudioDeviceInfo
	if err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices); err != nil {
		return nil, fmt.Errorf("failed to enumerate audio devices: %w", err)
	}
	return devices, nil
}

func selectDevice(devices []types.AudioDeviceInfo, requested, deviceType string) (types.AudioDeviceInfo, error) {
	var filtered []types.AudioDeviceInfo
	for _, device := range devices {
		if device.DeviceType == deviceType {
			filtered = append(filtered, device)
		}
	}

	if len(filtered) == 0 {
		return types.AudioDeviceInfo{}, fmt.Errorf("no %s devices found", deviceType)
	}

	if requested != "" {
		for _, device := range filtered {
			if device.ID == requested {
				return device, nil
			}
		}
		return types.AudioDeviceInfo{}, fmt.Errorf("%s device %s not found", deviceType, requested)
	}

	for _, device := range filtered {
		if device.IsDefault {
			return device, nil
		}
	}

	return filtered[0], nil
}

func resolveCaptureDevices(devices []types.AudioDeviceInfo, requested []string, fallback types.AudioDeviceInfo) ([]types.AudioDeviceInfo, error) {
	if len(requested) == 0 {
		return []types.AudioDeviceInfo{fallback}, nil
	}

	var result []types.AudioDeviceInfo
	for _, id := range requested {
		if id == "" {
			continue
		}
		dev, err := lookupDevice(devices, id, "capture")
		if err != nil {
			return nil, err
		}
		result = append(result, dev)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("no valid capture devices resolved from %v", requested)
	}

	return result, nil
}

func lookupDevice(devices []types.AudioDeviceInfo, id, deviceType string) (types.AudioDeviceInfo, error) {
	for _, device := range devices {
		if device.DeviceType == deviceType && device.ID == id {
			return device, nil
		}
	}
	return types.AudioDeviceInfo{}, fmt.Errorf("%s device %s not found", deviceType, id)
}

func parseDeviceIDs(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			result = append(result, part)
		}
	}
	return result
}

func collectDeviceIDs(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	var result []string
	for _, v := range values {
		result = append(result, parseDeviceIDs(v)...)
	}
	return result
}

func printCaptureBanner(devices []types.AudioDeviceInfo, output string) {
	if len(devices) == 1 {
		dev := devices[0]
		fmt.Printf("Recording from %s [ID=%s] -> %s\n", dev.Name, dev.ID, output)
		if dev.MinChannels > 0 && dev.MaxChannels > 0 {
			fmt.Printf("Capture capabilities: %d-%d channels, %d-%d Hz\n", dev.MinChannels, dev.MaxChannels, dev.MinSampleRate, dev.MaxSampleRate)
		}
		return
	}

	fmt.Printf("Recording from %d capture devices -> %s\n", len(devices), output)
	for idx, dev := range devices {
		fmt.Printf("  %d. %s [ID=%s]\n", idx+1, dev.Name, dev.ID)
		if dev.MinChannels > 0 && dev.MaxChannels > 0 {
			fmt.Printf("     Channels: %d-%d, Sample Rate: %d-%d Hz\n", dev.MinChannels, dev.MaxChannels, dev.MinSampleRate, dev.MaxSampleRate)
		}
	}
	fmt.Println("Captured channels will be appended sequentially in the output WAV.")
}

func normalizeFormat(format string) string {
	format = strings.ToUpper(strings.TrimSpace(format))
	switch format {
	case "U8", "S16", "S24", "S32", "F32":
		return format
	case "":
		return "S16"
	default:
		return "S16"
	}
}

// combineTensors combines multiple tensors into a single tensor with all channels appended.
// All tensors must have the same number of frames, sample rate, and data type.
func combineTensors(tensors []types.Tensor) (types.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors to combine")
	}
	if len(tensors) == 1 {
		return tensors[0], nil
	}

	// Validate all tensors have compatible shapes and types
	firstShape := tensors[0].Shape()
	if len(firstShape) != 2 {
		return nil, fmt.Errorf("tensor shape must be [frames, channels]")
	}
	firstFrames := firstShape[0]
	firstDtype := tensors[0].DataType()
	totalChannels := firstShape[1]

	for i := 1; i < len(tensors); i++ {
		shape := tensors[i].Shape()
		if len(shape) != 2 {
			return nil, fmt.Errorf("tensor %d: shape must be [frames, channels]", i)
		}
		if shape[0] != firstFrames {
			return nil, fmt.Errorf("tensor %d: frame count mismatch (%d != %d)", i, shape[0], firstFrames)
		}
		if tensors[i].DataType() != firstDtype {
			return nil, fmt.Errorf("tensor %d: data type mismatch (%v != %v)", i, tensors[i].DataType(), firstDtype)
		}
		totalChannels += shape[1]
	}

	// Create combined tensor
	combinedShape := tensor.NewShape(firstFrames, totalChannels)
	combined := tensor.New(firstDtype, combinedShape)
	if combined.Empty() {
		return nil, fmt.Errorf("failed to create combined tensor")
	}

	// Copy data from all tensors
	channelOffset := 0
	for _, t := range tensors {
		shape := t.Shape()
		channels := shape[1]
		srcData := t.Data()
		if srcData == nil {
			channelOffset += channels
			continue
		}

		dstData := combined.Data()
		if dstData == nil {
			return nil, fmt.Errorf("combined tensor data is nil")
		}

		// Copy channels from source to combined tensor
		switch firstDtype {
		case tensortypes.INT16:
			dst := dstData.([]int16)
			src := srcData.([]int16)
			for frame := 0; frame < firstFrames; frame++ {
				for ch := 0; ch < channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*channels+ch]
				}
			}
		case tensortypes.INT32:
			dst := dstData.([]int32)
			src := srcData.([]int32)
			for frame := 0; frame < firstFrames; frame++ {
				for ch := 0; ch < channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*channels+ch]
				}
			}
		case tensortypes.FP32:
			dst := dstData.([]float32)
			src := srcData.([]float32)
			for frame := 0; frame < firstFrames; frame++ {
				for ch := 0; ch < channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*channels+ch]
				}
			}
		case tensortypes.UINT8:
			dst := dstData.([]uint8)
			src := srcData.([]uint8)
			for frame := 0; frame < firstFrames; frame++ {
				for ch := 0; ch < channels; ch++ {
					dst[frame*totalChannels+channelOffset+ch] = src[frame*channels+ch]
				}
			}
		default:
			return nil, fmt.Errorf("unsupported dtype: %v", firstDtype)
		}

		channelOffset += channels
	}

	return combined, nil
}

// cloneTensor creates a copy of the tensor with the same data using Clone().
// Returns nil if cloning fails.
func cloneTensor(t types.Tensor) types.Tensor {
	if t.Empty() {
		return nil
	}
	cloned := t.Clone()
	if cloned.Empty() {
		return nil
	}
	return cloned
}
