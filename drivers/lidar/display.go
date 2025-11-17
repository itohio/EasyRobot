//go:build linux || windows

package main

import (
	"context"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

// DisplaySetup holds display-related resources.
type DisplaySetup struct {
	Stream      types.FrameStream
	FrameChan   chan types.Frame
	Tensor      types.Tensor
	Mat         *cv.Mat
	ImageWidth  int
	ImageHeight int
	ScaleFactor float64
}

// SetupDisplay sets up the display if enabled in config.
func SetupDisplay(ctx context.Context, cfg *Config) (*DisplaySetup, error) {
	if !cfg.Display {
		return nil, nil
	}

	frameChan := make(chan types.Frame, 1)
	stream := types.NewFrameStream(frameChan, func() {
		close(frameChan)
	})

	winW, winH := cfg.WindowSize()

	marshaller := gocv.NewMarshaller(
		gocv.WithDisplay(ctx),
		gocv.WithTitle("LiDAR Scan Visualization (Press ESC to exit)"),
		gocv.WithWindowSize(winW, winH),
	)

	go func() {
		if err := marshaller.Marshal(nil, stream); err != nil {
			// Error already logged by marshaller
		}
	}()

	tensor, err := tensorgocv.NewImage(cfg.ImageHeight, cfg.ImageWidth, 3)
	if err != nil {
		stream.Close()
		return nil, fmt.Errorf("failed to create display tensor: %w", err)
	}

	accessor, ok := tensor.(tensorgocv.Accessor)
	if !ok {
		tensor.Release()
		stream.Close()
		return nil, fmt.Errorf("display tensor does not implement Accessor")
	}

	mat, err := accessor.MatRef()
	if err != nil {
		tensor.Release()
		stream.Close()
		return nil, fmt.Errorf("failed to get Mat reference: %w", err)
	}

	return &DisplaySetup{
		Stream:      stream,
		FrameChan:   frameChan,
		Tensor:      tensor,
		Mat:         mat,
		ImageWidth:  cfg.ImageWidth,
		ImageHeight: cfg.ImageHeight,
		ScaleFactor: cfg.ScaleFactor,
	}, nil
}

// Close releases display resources.
func (d *DisplaySetup) Close() {
	if d == nil {
		return
	}
	if d.Tensor != nil {
		d.Tensor.Release()
	}
	if d.Stream.C != nil {
		d.Stream.Close()
	}
}

