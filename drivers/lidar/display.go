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
// Follows the gocv marshaller pattern for display setup.
// The cancelFunc is called when ESC is pressed to cancel the parent context.
func SetupDisplay(ctx context.Context, cfg *Config, cancelFunc context.CancelFunc) (*DisplaySetup, error) {
	if !cfg.Display {
		return nil, nil
	}

	frameChan := make(chan types.Frame, 1)
	stream := types.NewFrameStream(frameChan, func() {
		close(frameChan)
	})

	winW, winH := cfg.WindowSize()

	// Create marshaller with display options following gocv pattern
	// Pass context for cancellation, key handler for ESC, and cancel function
	marshaller := gocv.NewMarshaller(
		gocv.WithDisplay(ctx),
		gocv.WithTitle("LiDAR Scan Visualization (Press ESC to exit)"),
		gocv.WithWindowSize(winW, winH),
		gocv.WithOnKey(func(event types.KeyEvent) bool {
			// ESC key (27) cancels parent context and stops event loop
			if event.Key == types.KeyEscape {
				if cancelFunc != nil {
					cancelFunc()
				}
				return false // Stop event loop, marshaller will handle cleanup
			}
			return true // Continue processing
		}),
		gocv.WithCancel(cancelFunc),
	)

	// Start marshaller in goroutine (nil writer = side-effect operation, display only)
	go func() {
		if err := marshaller.Marshal(nil, stream); err != nil {
			// Error already logged by marshaller
			// Context cancellation or window close will cause normal exit
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
