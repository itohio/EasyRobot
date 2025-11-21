package gocv

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type videoDeviceLoader struct {
	spec    deviceSpec
	cfg     config
	capture *cv.VideoCapture
	index   int
}

// cameraController implements types.CameraController for GoCV VideoCapture
type cameraController struct {
	capture *cv.VideoCapture
}

// Controls returns available camera controls (simplified for GoCV limitations)
func (c *cameraController) Controls() []types.ControlInfo {
	return []types.ControlInfo{
		{Name: "brightness", Description: "Brightness", Type: "integer", Min: 0, Max: 255, Default: 128, Step: 1},
		{Name: "contrast", Description: "Contrast", Type: "integer", Min: 0, Max: 255, Default: 128, Step: 1},
		{Name: "saturation", Description: "Saturation", Type: "integer", Min: 0, Max: 255, Default: 128, Step: 1},
		{Name: "hue", Description: "Hue", Type: "integer", Min: -180, Max: 180, Default: 0, Step: 1},
		{Name: "gain", Description: "Gain", Type: "integer", Min: 0, Max: 255, Default: 0, Step: 1},
		{Name: "exposure", Description: "Exposure", Type: "integer", Min: -1, Max: -1, Default: -1, Step: 1}, // Auto
	}
}

// GetControl gets a camera control value
func (c *cameraController) GetControl(name string) (int32, error) {
	if c.capture == nil {
		return 0, fmt.Errorf("gocv: camera not available")
	}

	var prop cv.VideoCaptureProperties
	switch name {
	case "brightness":
		prop = cv.VideoCaptureBrightness
	case "contrast":
		prop = cv.VideoCaptureContrast
	case "saturation":
		prop = cv.VideoCaptureSaturation
	case "hue":
		prop = cv.VideoCaptureHue
	case "gain":
		prop = cv.VideoCaptureGain
	case "exposure":
		prop = cv.VideoCaptureExposure
	default:
		return 0, fmt.Errorf("gocv: unknown control %s", name)
	}

	value := c.capture.Get(prop)
	return int32(value), nil
}

// SetControl sets a camera control value
func (c *cameraController) SetControl(name string, value int32) error {
	if c.capture == nil {
		return fmt.Errorf("gocv: camera not available")
	}

	var prop cv.VideoCaptureProperties
	switch name {
	case "brightness":
		prop = cv.VideoCaptureBrightness
	case "contrast":
		prop = cv.VideoCaptureContrast
	case "saturation":
		prop = cv.VideoCaptureSaturation
	case "hue":
		prop = cv.VideoCaptureHue
	case "gain":
		prop = cv.VideoCaptureGain
	case "exposure":
		prop = cv.VideoCaptureExposure
	default:
		return fmt.Errorf("gocv: unknown control %s", name)
	}

	c.capture.Set(prop, float64(value))
	return nil
}

// GetControls gets multiple control values
func (c *cameraController) GetControls() (map[string]int32, error) {
	controls := make(map[string]int32)
	controlInfos := c.Controls()

	for _, info := range controlInfos {
		value, err := c.GetControl(info.Name)
		if err != nil {
			continue // Skip controls that can't be read
		}
		controls[info.Name] = value
	}

	return controls, nil
}

// SetControls sets multiple control values
func (c *cameraController) SetControls(controls map[string]int32) error {
	for name, value := range controls {
		if err := c.SetControl(name, value); err != nil {
			return fmt.Errorf("gocv: set control %s: %w", name, err)
		}
	}
	return nil
}

// CameraController returns the camera controller for runtime control
func (l *videoDeviceLoader) CameraController() types.CameraController {
	if l.capture == nil {
		return nil
	}
	return &cameraController{capture: l.capture}
}

// fourccCode creates a FOURCC code from four characters.
// FOURCC is a 32-bit integer where each byte represents a character.
func fourccCode(c1, c2, c3, c4 byte) int {
	return int(c1) | (int(c2) << 8) | (int(c3) << 16) | (int(c4) << 24)
}

// pixelFormatToFOURCC converts a pixel format string to a FOURCC code.
// Supports common formats like: mjpeg, yuyv, yuv2, rgb24, bgr24, etc.
// Returns the FOURCC code as an integer that can be used with VideoCaptureFOURCC.
func pixelFormatToFOURCC(format string) (int, error) {
	format = strings.ToUpper(strings.TrimSpace(format))
	
	// Handle common pixel format aliases
	switch format {
	case "MJPEG", "MJPG", "JPEG":
		return fourccCode('M', 'J', 'P', 'G'), nil
	case "YUYV", "YUY2":
		return fourccCode('Y', 'U', 'Y', 'V'), nil
	case "YUV2", "UYVY":
		return fourccCode('U', 'Y', 'V', 'Y'), nil
	case "RGB24", "RGB3":
		return fourccCode('R', 'G', 'B', '3'), nil
	case "BGR24", "BGR3":
		return fourccCode('B', 'G', 'R', '3'), nil
	case "NV12":
		return fourccCode('N', 'V', '1', '2'), nil
	case "NV21":
		return fourccCode('N', 'V', '2', '1'), nil
	case "I420", "YV12":
		return fourccCode('I', '4', '2', '0'), nil
	case "H264":
		return fourccCode('H', '2', '6', '4'), nil
	case "H265", "HEVC":
		return fourccCode('H', 'E', 'V', 'C'), nil
	default:
		// If format is exactly 4 characters, treat as FOURCC code directly
		if len(format) == 4 {
			return fourccCode(
				byte(format[0]),
				byte(format[1]),
				byte(format[2]),
				byte(format[3]),
			), nil
		}
		return 0, fmt.Errorf("unknown pixel format: %s (supported: mjpeg, yuyv, yuv2, rgb24, bgr24, nv12, nv21, i420, h264, h265, or 4-char FOURCC)", format)
	}
}

func newVideoDeviceLoader(spec deviceSpec, cfg config) (sourceStream, error) {
	cap, err := cv.OpenVideoCapture(spec.ID)
	if err != nil {
		return nil, fmt.Errorf("gocv: open video device %d: %w", spec.ID, err)
	}

	// Set pixel format FIRST (if specified) - some backends require format before resolution
	if spec.PixelFormat != "" {
		fourcc, err := pixelFormatToFOURCC(spec.PixelFormat)
		if err != nil {
			slog.Warn("Invalid pixel format, ignoring", "device_id", spec.ID, "format", spec.PixelFormat, "err", err)
		} else {
			// Set pixel format using FOURCC code
			// This should be set before width/height for best compatibility
			cap.Set(cv.VideoCaptureFOURCC, float64(fourcc))
			// Verify the format was set by reading it back
			actualFourcc := int(cap.Get(cv.VideoCaptureFOURCC))
			if actualFourcc == fourcc {
				slog.Info("Pixel format set successfully", "device_id", spec.ID, "format", spec.PixelFormat, "fourcc", fourcc)
			} else {
				slog.Warn("Pixel format may not have been set correctly", 
					"device_id", spec.ID, "format", spec.PixelFormat, "requested_fourcc", fourcc, "actual_fourcc", actualFourcc)
			}
		}
	}

	// Configure capture properties (resolution and frame rate)
	if spec.Width > 0 {
		cap.Set(cv.VideoCaptureFrameWidth, float64(spec.Width))
	}
	if spec.Height > 0 {
		cap.Set(cv.VideoCaptureFrameHeight, float64(spec.Height))
	}
	if spec.FrameRate > 0 {
		cap.Set(cv.VideoCaptureFPS, float64(spec.FrameRate))
	}

	return &videoDeviceLoader{
		spec:    spec,
		cfg:     cfg,
		capture: cap,
	}, nil
}

func (l *videoDeviceLoader) Next(ctx context.Context) (frameItem, bool, error) {
	if l.capture == nil {
		return frameItem{}, false, fmt.Errorf("gocv: device capture closed")
	}

	for {
		select {
		case <-ctx.Done():
			return frameItem{}, false, ctx.Err()
		default:
		}

		frame := cv.NewMat()
		if ok := l.capture.Read(&frame); !ok {
			frame.Close()
			if l.cfg.stream.allowBestEffort {
				time.Sleep(10 * time.Millisecond)
				continue
			}
			return frameItem{}, false, nil
		}
		if frame.Empty() {
			frame.Close()
			if l.cfg.stream.allowBestEffort {
				time.Sleep(5 * time.Millisecond)
				continue
			}
			return frameItem{}, false, nil
		}

		tensor, err := matToTensor(frame, l.cfg, types.DT_UNKNOWN)
		if err != nil {
			frame.Close()
			return frameItem{}, false, err
		}

		filename := fmt.Sprintf("device_%d_%06d.png", l.spec.ID, l.index)
		meta := map[string]any{
			"device":      l.spec.ID,
			"timestamp":   time.Now().UnixNano(),
			"source":      "video_device",
			"frame_index": l.index,
			"index":       l.index,
			"filename":    filename,
			"name":        []string{filename},
		}
		l.index++

		return frameItem{
			tensors:  []types.Tensor{tensor},
			metadata: meta,
		}, true, nil
	}
}

func (l *videoDeviceLoader) Close() error {
	if l.capture != nil {
		l.capture.Close()
		l.capture = nil
	}
	return nil
}
