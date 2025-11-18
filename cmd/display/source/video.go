package source

import (
	"context"
	"fmt"
	"log/slog"

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
		slog.Error("No video paths specified")
		return fmt.Errorf("no video paths specified")
	}

	slog.Info("Starting video source", "paths", s.paths, "count", len(s.paths))

	opts := []types.Option{
		types.WithContext(ctx),
	}
	for _, path := range s.paths {
		opts = append(opts, gocv.WithPath(path))
		slog.Debug("Adding video path", "path", path)
	}

	slog.Debug("Creating gocv unmarshaller for video")
	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		slog.Error("Failed to create video source", "err", err)
		return fmt.Errorf("failed to create video source: %w", err)
	}

	s.stream = stream
	slog.Info("Video source stream created successfully")
	return s.baseSource.Start(ctx)
}

