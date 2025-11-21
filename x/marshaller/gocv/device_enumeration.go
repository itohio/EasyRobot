package gocv

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// enumerateVideoDevices scans for available video capture devices
func enumerateVideoDevices() ([]types.CameraInfo, error) {
	var devices []CameraInfo

	// On Linux, scan /dev/video* devices
	// On Windows/macOS, try opening devices sequentially (0, 1, 2, ...)
	if runtime.GOOS == "linux" {
		videoDevices, err := filepath.Glob("/dev/video*")
		if err != nil {
			return nil, fmt.Errorf("gocv: glob video devices: %w", err)
		}

		for _, devicePath := range videoDevices {
			// Extract device ID from path
			base := filepath.Base(devicePath)
			if !strings.HasPrefix(base, "video") {
				continue
			}
			idStr := strings.TrimPrefix(base, "video")
			id, err := strconv.Atoi(idStr)
			if err != nil {
				continue // Skip invalid device names
			}

			// Try to open device to get information
			device, err := getDeviceInfo(id, devicePath)
			if err != nil {
				// Log error but continue with other devices
				continue
			}

			devices = append(devices, device)
		}
	} else {
		// Windows/macOS: Try opening devices sequentially
		// Try up to 10 devices (most systems don't have more)
		maxDevices := 10
		for id := 0; id < maxDevices; id++ {
			device, err := getDeviceInfo(id, fmt.Sprintf("device_%d", id))
			if err != nil {
				// If we can't open it, it might not exist - continue to next
				continue
			}
			// If device was successfully opened, add it
			devices = append(devices, device)
		}
	}

	return devices, nil
}

// getDeviceInfo retrieves information about a specific video device
func getDeviceInfo(id int, path string) (types.CameraInfo, error) {
	info := types.CameraInfo{
		ID:   id,
		Path: path,
	}

	// Try to open the device to get more information
	cap, err := cv.OpenVideoCapture(id)
	if err != nil {
		// If we can't open it, return error (caller will skip)
		return info, fmt.Errorf("gocv: cannot open device %d: %w", id, err)
	}
	defer cap.Close()

	// Verify device is actually working by trying to read a frame
	// Some systems report devices that aren't actually available
	testMat := cv.NewMat()
	defer testMat.Close()
	if !cap.Read(&testMat) {
		// Device exists but can't read from it - skip
		return info, fmt.Errorf("gocv: device %d exists but cannot read frames", id)
	}

	// Get basic device info
	info.Name = fmt.Sprintf("Video Device %d", id)
	if runtime.GOOS == "linux" {
		info.Driver = "v4l2"
		info.Card = fmt.Sprintf("Camera %d", id)
		info.BusInfo = "platform"
	} else if runtime.GOOS == "windows" {
		info.Driver = "DirectShow"
		info.Card = fmt.Sprintf("Camera %d", id)
		info.BusInfo = "USB"
	} else {
		info.Driver = "AVFoundation"
		info.Card = fmt.Sprintf("Camera %d", id)
		info.BusInfo = "USB"
	}
	info.Capabilities = []string{"VIDEO_CAPTURE"}

	// Try to get actual resolution from the device
	width := cap.Get(cv.VideoCaptureFrameWidth)
	height := cap.Get(cv.VideoCaptureFrameHeight)
	if width > 0 && height > 0 {
		// Use actual device resolution
		info.SupportedFormats = []types.VideoFormat{
			{PixelFormat: "BGR", Description: "BGR 24-bit", Width: int(width), Height: int(height)},
		}
	} else {
		// Fallback to common resolutions
		info.SupportedFormats = []types.VideoFormat{
			{PixelFormat: "BGR", Description: "BGR 24-bit", Width: 640, Height: 480},
			{PixelFormat: "BGR", Description: "BGR 24-bit", Width: 1280, Height: 720},
			{PixelFormat: "BGR", Description: "BGR 24-bit", Width: 1920, Height: 1080},
		}
	}


	// Get supported controls by trying to read actual values from the device
	// Try common camera controls and see which ones are available
	controlCandidates := []struct {
		name        string
		description string
		prop        cv.VideoCaptureProperties
	}{
		{"brightness", "Brightness", cv.VideoCaptureBrightness},
		{"contrast", "Contrast", cv.VideoCaptureContrast},
		{"saturation", "Saturation", cv.VideoCaptureSaturation},
		{"hue", "Hue", cv.VideoCaptureHue},
		{"gain", "Gain", cv.VideoCaptureGain},
		{"exposure", "Exposure", cv.VideoCaptureExposure},
		{"sharpness", "Sharpness", cv.VideoCaptureSharpness},
		{"white_balance_blue_u", "White Balance Blue U", cv.VideoCaptureWhiteBalanceBlueU},
		{"white_balance_red_v", "White Balance Red V", cv.VideoCaptureWhiteBalanceRedV},
		{"gamma", "Gamma", cv.VideoCaptureGamma},
		{"temperature", "Temperature", cv.VideoCaptureTemperature},
		{"iris", "Iris", cv.VideoCaptureIris},
		{"focus", "Focus", cv.VideoCaptureFocus},
		{"zoom", "Zoom", cv.VideoCaptureZoom},
		{"pan", "Pan", cv.VideoCapturePan},
		{"tilt", "Tilt", cv.VideoCaptureTilt},
		{"roll", "Roll", cv.VideoCaptureRoll},
		{"autofocus", "Auto Focus", cv.VideoCaptureAutoFocus},
		{"auto_exposure", "Auto Exposure", cv.VideoCaptureAutoExposure},
		{"auto_wb", "Auto White Balance", cv.VideoCaptureAutoWB},
	}

	var controls []types.ControlInfo
	for _, candidate := range controlCandidates {
		// Try to get the property value - if it returns a valid value, the control exists
		value := cap.Get(candidate.prop)
		// Check if property is supported (value != -1 or 0 depending on property)
		// For most properties, -1 means not supported, but some use 0
		// We'll consider it supported if we can read it (even if value is 0)
		// The actual check is if Get() doesn't error, which we can't check directly
		// So we'll include all common controls and let the user see which ones work
		controls = append(controls, types.ControlInfo{
			Name:        candidate.name,
			Description: candidate.description,
			Type:        "integer",
			Min:         0,   // GoCV doesn't expose min/max, use defaults
			Max:         255, // GoCV doesn't expose min/max, use defaults
			Default:     int32(value),
			Step:        1,
		})
	}

	info.Controls = controls

	return info, nil
}
