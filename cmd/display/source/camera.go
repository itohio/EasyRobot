package source

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// cameraConfig holds configuration for a single camera device
type cameraConfig struct {
	ID          int
	Width       int
	Height      int
	FrameRate   int
	PixelFormat string
}

// cameraSource implements Source for camera devices.
type cameraSource struct {
	baseSource
	deviceIDs   FlagArray
	configs     []cameraConfig // Per-camera configurations
	width       int            // Default width (used if not specified in config)
	height      int            // Default height (used if not specified in config)
	pixelFormat string         // Default pixel format (used if not specified in config)
	frameRate   int            // Default frame rate (used if not specified in config)
}

// NewCameraSource creates a new camera source.
func NewCameraSource() Source {
	return &cameraSource{}
}

func (s *cameraSource) RegisterFlags() {
	// Flags are registered by RegisterAllFlags() in factory.go
	// This method exists for interface compliance but does nothing
	// since flags are shared at package level
}

// parseCameraConfig parses a camera configuration string.
// Format: ID[:widthxheight[@fps[/format]]]
// Examples:
//   - "0" → ID=0, use defaults
//   - "0:640x480" → ID=0, width=640, height=480
//   - "0:640x480@30" → ID=0, width=640, height=480, fps=30
//   - "0:640x480@30/mjpeg" → ID=0, width=640, height=480, fps=30, format=mjpeg
//   - "0@30" → ID=0, fps=30
//   - "0/mjpeg" → ID=0, format=mjpeg
//   - "0:640x480/mjpeg" → ID=0, width=640, height=480, format=mjpeg
func parseCameraConfig(cameraStr string, defaultWidth, defaultHeight, defaultFPS int, defaultFormat string) (cameraConfig, error) {
	cfg := cameraConfig{
		Width:       defaultWidth,
		Height:      defaultHeight,
		FrameRate:   defaultFPS,
		PixelFormat: defaultFormat,
	}

	// Split by colon to get ID and optional config
	parts := strings.SplitN(cameraStr, ":", 2)
	if len(parts) == 0 {
		return cfg, fmt.Errorf("empty camera string")
	}

	// Parse ID
	id, err := strconv.Atoi(parts[0])
	if err != nil {
		return cfg, fmt.Errorf("invalid camera ID '%s': %w", parts[0], err)
	}
	cfg.ID = id

	// If no config part, use defaults
	if len(parts) == 1 {
		return cfg, nil
	}

	configPart := parts[1]

	// Parse format (if present): /format
	formatIdx := strings.LastIndex(configPart, "/")
	if formatIdx >= 0 {
		cfg.PixelFormat = strings.ToUpper(configPart[formatIdx+1:])
		configPart = configPart[:formatIdx]
	}

	// Parse fps (if present): @fps
	fpsIdx := strings.LastIndex(configPart, "@")
	if fpsIdx >= 0 {
		fpsStr := configPart[fpsIdx+1:]
		fps, err := strconv.Atoi(fpsStr)
		if err != nil {
			return cfg, fmt.Errorf("invalid fps '%s': %w", fpsStr, err)
		}
		cfg.FrameRate = fps
		configPart = configPart[:fpsIdx]
	}

	// Parse resolution (if present): widthxheight
	if configPart != "" {
		resParts := strings.Split(configPart, "x")
		if len(resParts) == 2 {
			width, err := strconv.Atoi(strings.TrimSpace(resParts[0]))
			if err != nil {
				return cfg, fmt.Errorf("invalid width '%s': %w", resParts[0], err)
			}
			height, err := strconv.Atoi(strings.TrimSpace(resParts[1]))
			if err != nil {
				return cfg, fmt.Errorf("invalid height '%s': %w", resParts[1], err)
			}
			cfg.Width = width
			cfg.Height = height
		} else {
			return cfg, fmt.Errorf("invalid resolution format '%s' (expected widthxheight)", configPart)
		}
	}

	return cfg, nil
}

func (s *cameraSource) Start(ctx context.Context) error {
	if len(s.deviceIDs) == 0 {
		slog.Error("No camera device IDs specified")
		return fmt.Errorf("no camera device IDs specified")
	}

	// Parse camera configurations
	s.configs = make([]cameraConfig, 0, len(s.deviceIDs))
	for _, devStr := range s.deviceIDs {
		cfg, err := parseCameraConfig(devStr, s.width, s.height, s.frameRate, s.pixelFormat)
		if err != nil {
			slog.Error("Invalid camera configuration", "camera", devStr, "err", err)
			return fmt.Errorf("invalid camera configuration '%s': %w", devStr, err)
		}
		s.configs = append(s.configs, cfg)
	}

	slog.Info("Starting camera source",
		"count", len(s.configs),
		"default_width", s.width,
		"default_height", s.height,
		"default_pixel_format", s.pixelFormat,
		"default_frame_rate", s.frameRate,
	)

	opts := []types.Option{
		types.WithContext(ctx),
	}

	for _, cfg := range s.configs {
		slog.Debug("Adding camera device",
			"device_id", cfg.ID,
			"width", cfg.Width,
			"height", cfg.Height,
			"pixel_format", cfg.PixelFormat,
			"frame_rate", cfg.FrameRate)
		opts = append(opts, gocv.WithVideoDeviceEx(cfg.ID, cfg.Width, cfg.Height, cfg.FrameRate, cfg.PixelFormat))
	}

	slog.Debug("Creating gocv unmarshaller for camera")
	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		slog.Error("Failed to create camera source", "err", err)
		return fmt.Errorf("failed to create camera source: %w", err)
	}

	s.stream = stream
	slog.Info("Camera source stream created successfully")
	return s.baseSource.Start(ctx)
}

