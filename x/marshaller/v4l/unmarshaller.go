// go:build linux
package v4l

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Unmarshaller implements the unmarshaller interface for V4L devices
type Unmarshaller struct {
	opts               types.Options
	cfg                Options
	activeControllers  map[string]types.CameraController // device path -> controller
	activeStreams      map[string]types.CameraStream     // device path -> stream
}

// New creates a new V4L unmarshaller
func New(opts ...types.Option) *Unmarshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, Options{}, opts...)
	return &Unmarshaller{
		opts:              baseOpts,
		cfg:               baseCfg,
		activeControllers: make(map[string]types.CameraController),
		activeStreams:     make(map[string]types.CameraStream),
	}
}

// Format returns the unmarshaller format identifier
func (u *Unmarshaller) Format() string {
	return "v4l"
}

// CameraController returns the camera controller for the specified device path
func (u *Unmarshaller) CameraController(devicePath string) types.CameraController {
	return u.activeControllers[devicePath]
}

// CameraStream returns the active stream for the specified device path
func (u *Unmarshaller) CameraStream(devicePath string) types.CameraStream {
	return u.activeStreams[devicePath]
}

// Cameras returns a list of available cameras
func (u *Unmarshaller) Cameras() []CameraDevice {
	devices, err := GetAllDevices()
	if err != nil {
		return nil
	}
	return devices
}

// Unmarshal handles unmarshalling of V4L-related types
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	localOpts, localCfg := applyOptions(u.opts, u.cfg, opts...)

	switch out := dst.(type) {
	case *[]types.CameraInfo:
		return u.unmarshalDeviceInfoList(r, out)
	case *types.CameraInfo:
		var info types.CameraInfo
		if err := u.unmarshalDeviceInfo(r, &info); err != nil {
			return err
		}
		*out = info
		return nil
	case *types.FrameStream:
		stream, err := u.unmarshalFrameStream(r, localCfg)
		if err != nil {
			return err
		}
		*out = stream
		return nil
	default:
		return types.NewError("unmarshal", "v4l", fmt.Sprintf("unsupported destination type %T", dst), nil)
	}
}

// unmarshalDeviceInfo unmarshals device information
func (u *Unmarshaller) unmarshalDeviceInfo(r io.Reader, dst *types.CameraInfo) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "v4l", "failed to read device info", err)
	}

	if err := json.Unmarshal(data, dst); err != nil {
		return types.NewError("unmarshal", "v4l", "failed to unmarshal device info", err)
	}

	return nil
}

// unmarshalDeviceInfoList unmarshals a list of device information or enumerates devices
func (u *Unmarshaller) unmarshalDeviceInfoList(r io.Reader, dst *[]types.CameraInfo) error {
	// First try to read as JSON array
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "v4l", "failed to read device list", err)
	}

	// Check if it's a command to list devices
	content := strings.TrimSpace(string(data))
	if content == "list" || content == "" {
		// Enumerate all devices
		infos, err := GetAllDeviceInfos()
		if err != nil {
			return types.NewError("unmarshal", "v4l", "failed to enumerate devices", err)
		}
		*dst = infos
		return nil
	}

	// Try to parse as JSON array
	var infos []types.CameraInfo
	if err := json.Unmarshal(data, &infos); err != nil {
		return types.NewError("unmarshal", "v4l", "failed to unmarshal device list", err)
	}

	*dst = infos
	return nil
}


// unmarshalFrameStream unmarshals a FrameStream (compatible with marshaller ecosystem)
func (u *Unmarshaller) unmarshalFrameStream(r io.Reader, cfg Options) (types.FrameStream, error) {
	// Read configuration from reader if provided
	if r != nil {
		data, err := io.ReadAll(r)
		if err != nil {
			return types.FrameStream{}, types.NewError("unmarshal", "v4l", "read config", err)
		}

		content := strings.TrimSpace(string(data))
		if content != "" && !strings.Contains(content, "{") {
			// Treat as device path list (one per line)
			paths := strings.Split(content, "\n")
			for _, path := range paths {
				path = strings.TrimSpace(path)
				if path != "" {
					cfg.DevicePaths = append(cfg.DevicePaths, path)
				}
			}
		}
	}

	// If no device paths configured, check if we have any from options
	if len(cfg.DevicePaths) == 0 {
		return types.FrameStream{}, types.NewError("unmarshal", "v4l", "no device paths configured", nil)
	}

	// Create streams for each device path
	var streams []types.CameraStream
	var controllers []types.CameraController

	for _, devicePath := range cfg.DevicePaths {
		// Create device
		device, err := NewDevice(devicePath)
		if err != nil {
			return types.FrameStream{}, types.NewError("unmarshal", "v4l", fmt.Sprintf("failed to open device %s", devicePath), err)
		}

		// Open stream with shared options
		var streamOpts []types.CameraOption
		if cfg.Width > 0 && cfg.Height > 0 {
			streamOpts = append(streamOpts, types.WithCameraResolution(cfg.Width, cfg.Height))
		}
		if cfg.PixelFormat != "" {
			streamOpts = append(streamOpts, types.WithCameraPixelFormat(cfg.PixelFormat))
		}
		if cfg.BufferCount > 0 {
			streamOpts = append(streamOpts, types.WithCameraBufferCount(cfg.BufferCount))
		}
		if len(cfg.Controls) > 0 {
			streamOpts = append(streamOpts, types.WithCameraControls(cfg.Controls))
		}

		stream, err := device.Open(streamOpts...)
		if err != nil {
			return types.FrameStream{}, types.NewError("unmarshal", "v4l", fmt.Sprintf("failed to open stream for device %s", devicePath), err)
		}

		streams = append(streams, stream)
		if controller := stream.Controller(); controller != nil {
			controllers = append(controllers, controller)
			u.activeControllers[devicePath] = controller
			u.activeStreams[devicePath] = stream
		}
	}

	// Create multi-stream for synchronization
	multiStream := NewMultiStream(streams,
		WithBestEffort(cfg.AllowBestEffort),
		WithSequential(cfg.Sequential),
		WithContext(cfg.Context),
	)

	// Convert to FrameStream
	return u.multiStreamToFrameStream(multiStream, cfg)
}

// streamToFrameStream converts a single Stream to FrameStream
func (u *Unmarshaller) streamToFrameStream(stream Stream, cfg Options) (types.FrameStream, error) {
	output := make(chan types.Frame, cfg.BufferCount*2)

	go func() {
		defer close(output)

		if err := stream.Start(cfg.Context); err != nil {
			output <- errorFrame(err, 0)
			return
		}
		defer stream.Stop()

		frameChan := stream.FrameChannel()
		for {
			select {
			case frame, ok := <-frameChan:
				if !ok {
					return
				}

				// Convert V4L Frame to types.Frame
				typesFrame := types.Frame{
					Index:     frame.Index,
					Timestamp: frame.Timestamp,
					Metadata:  frame.Metadata,
				}

				// Add tensor(s)
				if frame.Tensor != nil {
					typesFrame.Tensors = []types.Tensor{frame.Tensor}
				} else if len(frame.Tensors) > 0 {
					typesFrame.Tensors = frame.Tensors
				}

				select {
				case output <- typesFrame:
				case <-cfg.Context.Done():
					return
				}

			case <-cfg.Context.Done():
				return
			}
		}
	}()

	return types.NewFrameStream(output, func() {
		stream.Stop()
		stream.Close()
	}), nil
}

// multiStreamToFrameStream converts a MultiStream to FrameStream
func (u *Unmarshaller) multiStreamToFrameStream(multiStream *MultiStream, cfg Options) (types.FrameStream, error) {
	output := make(chan types.Frame, len(multiStream.streams)*cfg.BufferCount*2)

	go func() {
		defer close(output)

		if err := multiStream.Start(cfg.Context); err != nil {
			output <- errorFrame(err, 0)
			return
		}
		defer multiStream.Stop()

		frameChan := multiStream.FrameChannel()
		for {
			select {
			case frame, ok := <-frameChan:
				if !ok {
					return
				}

				// Convert V4L Frame to types.Frame
				typesFrame := types.Frame{
					Index:     frame.Index,
					Timestamp: frame.Timestamp,
					Metadata:  frame.Metadata,
					Tensors:   frame.Tensors,
				}

				select {
				case output <- typesFrame:
				case <-cfg.Context.Done():
					return
				}

			case <-cfg.Context.Done():
				return
			}
		}
	}()

	return types.NewFrameStream(output, func() {
		multiStream.Stop()
		multiStream.Close()
	}), nil
}

// errorFrame creates an error frame
func errorFrame(err error, index int) types.Frame {
	return types.Frame{
		Index:     index,
		Timestamp: time.Now().UnixNano(),
		Metadata: map[string]any{
			"error": err.Error(),
		},
	}
}

// parsePixelFormat parses a pixel format string
func parsePixelFormat(s string) PixelFormat {
	switch strings.ToUpper(s) {
	case "RGB24":
		return PixelFmtRGB24
	case "BGR24":
		return PixelFmtBGR24
	case "YUYV":
		return PixelFmtYUYV
	case "UYVY":
		return PixelFmtUYVY
	case "YU12", "YUV420":
		return PixelFmtYUV420
	case "422P":
		return PixelFmtYUV422P
	case "MJPEG", "MJPG":
		return PixelFmtMJPEG
	case "H264":
		return PixelFmtH264
	case "NV12":
		return PixelFmtNV12
	case "GREY", "GRAY":
		return PixelFmtGREY
	default:
		return PixelFmtMJPEG // Default fallback
	}
}
