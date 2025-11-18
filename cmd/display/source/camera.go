package source

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// cameraSource implements Source for camera devices.
type cameraSource struct {
	baseSource
	deviceIDs FlagArray
	width     int
	height    int
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

func (s *cameraSource) Start(ctx context.Context) error {
	if len(s.deviceIDs) == 0 {
		slog.Error("No camera device IDs specified")
		return fmt.Errorf("no camera device IDs specified")
	}

	slog.Info("Starting camera source",
		"device_ids", s.deviceIDs,
		"count", len(s.deviceIDs),
		"width", s.width,
		"height", s.height,
	)

	opts := []types.Option{
		types.WithContext(ctx),
	}

	for _, devStr := range s.deviceIDs {
		deviceID, err := strconv.Atoi(devStr)
		if err != nil {
			slog.Error("Invalid camera device ID", "device_id", devStr, "err", err)
			return fmt.Errorf("invalid camera device ID '%s': %w", devStr, err)
		}
		slog.Debug("Adding camera device", "device_id", deviceID, "width", s.width, "height", s.height)
		opts = append(opts, gocv.WithVideoDevice(deviceID, s.width, s.height))
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

