package gocv

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// enumerateVideoDevices scans for available video capture devices
func enumerateVideoDevices() ([]types.CameraInfo, error) {
	var devices []CameraInfo

	// Scan /dev/video* devices
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
		// If we can't open it, still return basic info
		info.Name = fmt.Sprintf("Video Device %d", id)
		info.Driver = "Unknown"
		info.Card = "Unknown"
		info.BusInfo = "Unknown"
		info.Capabilities = []string{"Unknown"}
		return info, nil
	}
	defer cap.Close()

	// Get basic device info
	info.Name = fmt.Sprintf("Video Device %d", id)
	info.Driver = "v4l2" // Assume V4L2 driver
	info.Card = fmt.Sprintf("Camera %d", id)
	info.BusInfo = "platform"
	info.Capabilities = []string{"VIDEO_CAPTURE"}

	// Get supported formats (simplified - in a real implementation this would query V4L2)
	info.SupportedFormats = []types.VideoFormat{
		{PixelFormat: "MJPG", Description: "Motion JPEG", Width: 640, Height: 480},
		{PixelFormat: "YUYV", Description: "YUYV 4:2:2", Width: 640, Height: 480},
	}

	// Get supported controls (simplified - in a real implementation this would query V4L2 controls)
	info.Controls = []types.ControlInfo{
		{Name: "brightness", Description: "Brightness", Type: "integer", Min: 0, Max: 255, Default: 128, Step: 1},
		{Name: "contrast", Description: "Contrast", Type: "integer", Min: 0, Max: 255, Default: 128, Step: 1},
		{Name: "saturation", Description: "Saturation", Type: "integer", Min: 0, Max: 255, Default: 128, Step: 1},
		{Name: "hue", Description: "Hue", Type: "integer", Min: -180, Max: 180, Default: 0, Step: 1},
		{Name: "gamma", Description: "Gamma", Type: "integer", Min: 1, Max: 300, Default: 100, Step: 1},
		{Name: "exposure", Description: "Exposure", Type: "integer", Min: 1, Max: 10000, Default: 500, Step: 1},
		{Name: "gain", Description: "Gain", Type: "integer", Min: 0, Max: 255, Default: 0, Step: 1},
		{Name: "sharpness", Description: "Sharpness", Type: "integer", Min: 0, Max: 255, Default: 3, Step: 1},
	}

	return info, nil
}
