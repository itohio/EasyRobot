package destination

import (
	"context"
	"fmt"
	"log/slog"
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
		slog.Warn("Video destination already started")
		return fmt.Errorf("video destination already started")
	}
	if outputPath == "" {
		slog.Info("Video destination disabled (no output path)")
		return nil // Video output is disabled
	}

	slog.Info("Starting video destination", "path", outputPath, "fps", v.fps)
	v.ctx = ctx
	v.path = outputPath
	v.started = true
	slog.Info("Video destination started (writer will be created on first frame)")
	return nil
}

func (v *videoDestination) AddFrame(frame types.Frame) error {
	if !v.started || v.path == "" {
		return nil // Video output is disabled or not started
	}

	if len(frame.Tensors) == 0 {
		slog.Debug("Frame has no tensors, skipping video write", "frame_index", frame.Index)
		return nil
	}

	// Convert tensor to Mat
	slog.Debug("Converting tensor to Mat for video", "frame_index", frame.Index)
	mat, err := tensorToMat(frame.Tensors[0])
	if err != nil {
		slog.Error("Failed to convert tensor to mat", "frame_index", frame.Index, "err", err)
		return fmt.Errorf("failed to convert tensor to mat: %w", err)
	}
	defer mat.Close()
	
	// Release tensor after converting to Mat (tensor is no longer needed)
	// The Mat clone is independent, so we can release the tensor
	// Note: If using smart tensors, this will decrement the ref count
	defer frame.Tensors[0].Release()

	// Initialize video writer on first frame if not already initialized
	if v.writer == nil {
		slog.Info("Initializing video writer on first frame", "frame_index", frame.Index)
		// Get frame dimensions
		size := mat.Size()
		if len(size) < 2 {
			slog.Error("Invalid mat size", "size", size)
			return fmt.Errorf("invalid mat size")
		}
		v.height = size[0]
		v.width = size[1]

		// Determine codec from file extension
		ext := strings.ToLower(filepath.Ext(v.path))
		codec := v.getCodec(ext)

		slog.Info("Creating video writer",
			"path", v.path,
			"codec", codec,
			"fps", v.fps,
			"width", v.width,
			"height", v.height,
		)

		// Create video writer
		// GoCV VideoWriter API
		writer, err := cv.VideoWriterFile(v.path, codec, v.fps, v.width, v.height, true)
		if err != nil {
			slog.Error("Failed to create video writer", "err", err)
			return fmt.Errorf("failed to create video writer: %w", err)
		}
		if writer == nil {
			slog.Error("Video writer is nil")
			return fmt.Errorf("video writer is nil")
		}
		v.writer = writer
		slog.Info("Video writer created successfully")
	}

	// Write frame
	slog.Debug("Writing frame to video", "frame_index", frame.Index)
	if err := v.writer.Write(mat); err != nil {
		slog.Error("Failed to write frame to video", "frame_index", frame.Index, "err", err)
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
	slog.Info("Closing video destination", "path", v.path)
	if v.writer != nil {
		slog.Debug("Closing video writer")
		v.writer.Close()
		v.writer = nil
		slog.Info("Video file written", "path", v.path)
	}
	v.started = false
	slog.Info("Video destination closed")
	return nil
}

