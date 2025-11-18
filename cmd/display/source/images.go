package source

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// FlagArray is a custom flag type that supports multiple values.
type FlagArray []string

// String returns a string representation of the flag array.
func (f *FlagArray) String() string {
	if f == nil || len(*f) == 0 {
		return ""
	}
	return strings.Join(*f, ",")
}

// Set adds a value to the flag array.
func (f *FlagArray) Set(value string) error {
	*f = append(*f, value)
	return nil
}

// imageSource implements Source for image files and directories.
type imageSource struct {
	baseSource
	paths FlagArray
}

// NewImageSource creates a new image source.
func NewImageSource() Source {
	return &imageSource{}
}

func (s *imageSource) RegisterFlags() {
	// Flags are registered by RegisterAllFlags() in factory.go
	// This method exists for interface compliance but does nothing
	// since flags are shared at package level
}

func (s *imageSource) Start(ctx context.Context) error {
	if len(s.paths) == 0 {
		slog.Error("No image paths specified")
		return fmt.Errorf("no image paths specified")
	}

	slog.Info("Starting image source", "paths", s.paths, "count", len(s.paths))

	opts := []types.Option{
		types.WithContext(ctx),
	}
	for _, path := range s.paths {
		opts = append(opts, gocv.WithPath(path))
		slog.Debug("Adding image path", "path", path)
	}

	slog.Debug("Creating gocv unmarshaller for images")
	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		slog.Error("Failed to create image source", "err", err)
		return fmt.Errorf("failed to create image source: %w", err)
	}

	s.stream = stream
	slog.Info("Image source stream created successfully")
	return s.baseSource.Start(ctx)
}

