package source

import (
	"context"
	"fmt"
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
		return fmt.Errorf("no camera device IDs specified")
	}

	opts := []types.Option{
		types.WithContext(ctx),
	}

	for _, devStr := range s.deviceIDs {
		deviceID, err := strconv.Atoi(devStr)
		if err != nil {
			return fmt.Errorf("invalid camera device ID '%s': %w", devStr, err)
		}
		opts = append(opts, gocv.WithVideoDevice(deviceID, s.width, s.height))
	}

	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		return fmt.Errorf("failed to create camera source: %w", err)
	}

	s.stream = stream
	return s.baseSource.Start(ctx)
}

