package source

import (
	"context"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// videoSource implements Source for video files.
type videoSource struct {
	baseSource
	paths FlagArray
}

// NewVideoSource creates a new video source.
func NewVideoSource() Source {
	return &videoSource{}
}

func (s *videoSource) RegisterFlags() {
	// Flags are registered by RegisterAllFlags() in factory.go
	// This method exists for interface compliance but does nothing
	// since flags are shared at package level
}

func (s *videoSource) Start(ctx context.Context) error {
	if len(s.paths) == 0 {
		return fmt.Errorf("no video paths specified")
	}

	opts := []types.Option{
		types.WithContext(ctx),
	}
	for _, path := range s.paths {
		opts = append(opts, gocv.WithPath(path))
	}

	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		return fmt.Errorf("failed to create video source: %w", err)
	}

	s.stream = stream
	return s.baseSource.Start(ctx)
}

