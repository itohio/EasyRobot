package destination

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	cv "gocv.io/x/gocv"
)

var (
	outputPath string
)

// videoDestination implements Destination for writing frames to a video file.
type videoDestination struct {
	ctx     context.Context
	writer  *cv.VideoWriter
	started bool
	path    string
	fps     float64
	width   int
	height  int
}

// NewVideo creates a new video file destination.
func NewVideo() Destination {
	return &videoDestination{
		fps: 30.0, // Default FPS
	}
}

func (v *videoDestination) RegisterFlags() {
	// Flags are registered by RegisterAllFlags() in factory.go
	// This method exists for interface compliance but does nothing
}

func (v *videoDestination) Start(ctx context.Context) error {
	if v.started {
		return fmt.Errorf("video destination already started")
	}
	if outputPath == "" {
		return nil // Video output is disabled
	}

	v.ctx = ctx
	v.path = outputPath
	v.started = true
	return nil
}

func (v *videoDestination) AddFrame(frame types.Frame) error {
	if !v.started || v.path == "" {
		return nil // Video output is disabled or not started
	}

	if len(frame.Tensors) == 0 {
		return nil
	}

	// Convert tensor to Mat
	mat, err := tensorToMat(frame.Tensors[0])
	if err != nil {
		return fmt.Errorf("failed to convert tensor to mat: %w", err)
	}
	defer mat.Close()

	// Initialize video writer on first frame if not already initialized
	if v.writer == nil {
		// Get frame dimensions
		size := mat.Size()
		if len(size) < 2 {
			return fmt.Errorf("invalid mat size")
		}
		v.height = size[0]
		v.width = size[1]

		// Determine codec from file extension
		ext := strings.ToLower(filepath.Ext(v.path))
		codec := v.getCodec(ext)

		// Create video writer
		// GoCV VideoWriter API
		writer, err := cv.VideoWriterFile(v.path, codec, v.fps, v.width, v.height, true)
		if err != nil {
			return fmt.Errorf("failed to create video writer: %w", err)
		}
		if writer == nil {
			return fmt.Errorf("video writer is nil")
		}
		v.writer = writer
	}

	// Write frame
	if err := v.writer.Write(mat); err != nil {
		return fmt.Errorf("failed to write frame: %w", err)
	}

	return nil
}

func (v *videoDestination) getCodec(ext string) string {
	switch ext {
	case ".mp4":
		return "mp4v"
	case ".avi":
		return "XVID"
	case ".mov":
		return "mp4v"
	default:
		return "mp4v" // Default to mp4v
	}
}

func (v *videoDestination) Close() error {
	if v.writer != nil {
		v.writer.Close()
		v.writer = nil
	}
	v.started = false
	return nil
}

