package gocv

import (
	"context"
	"fmt"
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

func newVideoDeviceLoader(spec deviceSpec, cfg config) (sourceStream, error) {
	cap, err := cv.OpenVideoCapture(spec.ID)
	if err != nil {
		return nil, fmt.Errorf("gocv: open video device %d: %w", spec.ID, err)
	}

	// Configure capture properties
	if spec.Width > 0 {
		cap.Set(cv.VideoCaptureFrameWidth, float64(spec.Width))
	}
	if spec.Height > 0 {
		cap.Set(cv.VideoCaptureFrameHeight, float64(spec.Height))
	}
	if spec.FrameRate > 0 {
		cap.Set(cv.VideoCaptureFPS, float64(spec.FrameRate))
	}

	// Note: GoCV doesn't directly expose pixel format setting via VideoCapture
	// The pixel format is typically set through the resolution/frame rate configuration
	// or through lower-level V4L2 APIs if needed

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
