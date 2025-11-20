package v4l

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vladimirvivien/go4vl/v4l2"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

// v4lDevice implements the CameraDevice interface using go4vl
type v4lDevice struct {
	path   string
	device *v4l2.Device
	info   types.CameraInfo
}

// NewDevice creates a new V4L device wrapper
func NewDevice(path string) (CameraDevice, error) {
	device, err := v4l2.Open(path)
	if err != nil {
		return nil, fmt.Errorf("v4l: failed to open device %s: %w", path, err)
	}

	info, err := queryDeviceInfo(device, path)
	if err != nil {
		device.Close()
		return nil, fmt.Errorf("v4l: failed to query device info for %s: %w", path, err)
	}

	return &v4lDevice{
		path:   path,
		device: device,
		info:   info,
	}, nil
}

// Info returns device information and capabilities
func (d *v4lDevice) Info() types.CameraInfo {
	return d.info
}

// Open opens the device with specified options
func (d *v4lDevice) Open(opts ...types.CameraOption) (types.CameraStream, error) {
	// Apply camera options to our internal options
	var cameraOpts types.CameraOptions
	for _, opt := range opts {
		opt.Apply(&cameraOpts)
	}

	// Convert shared camera options to our internal options
	var options Options
	options.Width = cameraOpts.Width
	options.Height = cameraOpts.Height
	if cameraOpts.FrameRate > 0 {
		options.FrameRate = Fraction{Numerator: cameraOpts.FrameRate, Denominator: 1}
	}
	if cameraOpts.PixelFormat != nil {
		// Try to convert pixel format
		if pf, ok := cameraOpts.PixelFormat.(string); ok {
			options.PixelFormat = parsePixelFormat(pf)
		}
	}
	// Convert named controls to internal format
	options.Controls = make(map[ControlID]int32)
	for name, value := range cameraOpts.Controls {
		if ctrlID, exists := controlNameToID[name]; exists {
			options.Controls[ctrlID] = value
		}
	}
	options.BufferCount = cameraOpts.BufferCount

	// Set defaults
	if options.BufferCount == 0 {
		options.BufferCount = 4
	}
	if options.Width == 0 {
		options.Width = 640
	}
	if options.Height == 0 {
		options.Height = 480
	}
	if options.FrameRate.Numerator == 0 {
		options.FrameRate = Fraction{Numerator: 30, Denominator: 1}
	}
	if options.PixelFormat == 0 {
		options.PixelFormat = PixelFmtMJPEG
	}
	if options.BufferPool == nil {
		options.BufferPool = createDefaultBufferPool()
	}
	if options.TensorFactory == nil {
		options.TensorFactory = defaultTensorFactory(options.BufferPool)
	}
	if options.Context == nil {
		options.Context = context.Background()
	}
	if options.Controls == nil {
		options.Controls = make(map[ControlID]int32)
	}

	stream, err := newV4LStream(d.device, options)
	if err != nil {
		return nil, fmt.Errorf("v4l: failed to create stream for device %s: %w", d.path, err)
	}

	return stream, nil
}

// Close closes the device
func (d *v4lDevice) Close() error {
	if d.device != nil {
		err := d.device.Close()
		d.device = nil
		return err
	}
	return nil
}

// queryDeviceInfo retrieves device information and capabilities
func queryDeviceInfo(device *v4l2.Device, path string) (types.CameraInfo, error) {
	caps, err := device.Capabilities()
	if err != nil {
		return types.CameraInfo{}, err
	}

	info := types.CameraInfo{
		Path:         path,
		Name:         string(caps.Card[:]),
		Driver:       string(caps.Driver[:]),
		BusInfo:      string(caps.BusInfo[:]),
		Capabilities: CapabilityFlags(caps.Capabilities),
		Metadata: map[string]any{
			"version": fmt.Sprintf("%d.%d.%d", caps.Version>>16, (caps.Version>>8)&0xff, caps.Version&0xff),
		},
	}

	// Query supported formats
	formats, err := device.Formats()
	if err == nil {
		for _, fmt := range formats {
			v4lFmt := Format{
				Width:       int(fmt.Width),
				Height:      int(fmt.Height),
				PixelFormat: PixelFormat(fmt.PixelFormat),
				Field:       Field(fmt.Field),
			}
			info.SupportedFormats = append(info.SupportedFormats, v4lFmt.ToVideoFormat())
		}
	}

	// Query controls
	controls, err := device.Controls()
	if err == nil {
		for _, ctrl := range controls {
			// Map V4L2 control to shared name if possible
			ctrlName := string(ctrl.Name[:])
			if name, exists := controlIDToName[ControlID(ctrl.ID)]; exists {
				ctrlName = name
			}

			var menuItems []string
			// Query menu items if applicable
			if ctrl.Type == v4l2.CtrlTypeMenu {
				menu, err := device.ControlMenu(ctrl.ID)
				if err == nil {
					for _, item := range menu {
						menuItems = append(menuItems, string(item.Name[:]))
					}
				}
			}

			ctrlInfo := ConvertControlToShared(
				ControlID(ctrl.ID),
				ctrlName,
				ControlType(ctrl.Type),
				ctrl.Min,
				ctrl.Max,
				ctrl.Default,
				ctrl.Step,
				menuItems,
			)

			info.Controls = append(info.Controls, ctrlInfo)
		}
	}

	return info, nil
}

// GetAllDevices returns all available V4L devices
func GetAllDevices() ([]CameraDevice, error) {
	paths, err := v4l2.DevicePaths()
	if err != nil {
		return nil, fmt.Errorf("v4l: failed to enumerate devices: %w", err)
	}

	var devices []CameraDevice
	for _, path := range paths {
		device, err := NewDevice(path)
		if err != nil {
			// Skip devices that can't be opened
			continue
		}
		devices = append(devices, device)
	}

	return devices, nil
}

// GetDeviceInfo returns device info without opening the device
func GetDeviceInfo(path string) (types.CameraInfo, error) {
	device, err := v4l2.Open(path)
	if err != nil {
		return types.CameraInfo{}, fmt.Errorf("v4l: failed to open device %s: %w", path, err)
	}
	defer device.Close()

	return queryDeviceInfo(device, path)
}

// GetAllDeviceInfos returns info for all available devices
func GetAllDeviceInfos() ([]types.CameraInfo, error) {
	paths, err := v4l2.DevicePaths()
	if err != nil {
		return nil, fmt.Errorf("v4l: failed to enumerate devices: %w", err)
	}

	var infos []types.CameraInfo
	for _, path := range paths {
		info, err := GetDeviceInfo(path)
		if err != nil {
			// Skip devices that can't be queried
			continue
		}
		infos = append(infos, info)
	}

	return infos, nil
}

// parsePixelFormat parses a pixel format string (internal helper)
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
