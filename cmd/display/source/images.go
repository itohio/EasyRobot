package source

import (
	"context"
	"fmt"
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
		return fmt.Errorf("no image paths specified")
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
		return fmt.Errorf("failed to create image source: %w", err)
	}

	s.stream = stream
	return s.baseSource.Start(ctx)
}

