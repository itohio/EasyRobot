package destination

import (
	"context"
	"fmt"
	"sync"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	cv "gocv.io/x/gocv"
)

var (
	noDisplay bool
	title     string
	width     int
	height    int
)

// displayDestination implements Destination for displaying frames in a window.
type displayDestination struct {
	ctx     context.Context
	window  *cv.Window
	started bool
	once    sync.Once
	stopped bool
}

// NewDisplay creates a new display destination.
func NewDisplay() Destination {
	return &displayDestination{}
}

func (d *displayDestination) RegisterFlags() {
	// Flags are registered by RegisterAllFlags() in factory.go
	// This method exists for interface compliance but does nothing
}

func (d *displayDestination) Start(ctx context.Context) error {
	if d.started {
		return fmt.Errorf("display destination already started")
	}
	if noDisplay {
		d.started = true
		return nil // Display is disabled
	}

	d.ctx = ctx
	d.started = true
	return nil
}

func (d *displayDestination) ensureWindow() {
	if noDisplay {
		return
	}
	d.once.Do(func() {
		winTitle := title
		if winTitle == "" {
			winTitle = "Display"
		}
		d.window = cv.NewWindow(winTitle)
		if width > 0 && height > 0 {
			d.window.ResizeWindow(width, height)
		}
	})
}

func (d *displayDestination) AddFrame(frame types.Frame) error {
	if !d.started || noDisplay {
		return nil // Display is disabled or not started
	}

	if d.stopped {
		return nil
	}

	if len(frame.Tensors) == 0 {
		return nil
	}

	d.ensureWindow()
	if d.window == nil || !d.window.IsOpen() {
		d.stopped = true
		return nil
	}

	// Convert tensor to Mat
	mat, err := tensorToMat(frame.Tensors[0])
	if err != nil {
		return fmt.Errorf("failed to convert tensor to mat: %w", err)
	}
	defer mat.Close()

	if err := d.window.IMShow(mat); err != nil {
		return err
	}

	// Check for keyboard input
	key := d.window.WaitKey(1)
	if key == 27 { // ESC
		d.stopped = true
		// Cancel context to signal stop
		if d.ctx != nil {
			// Context cancellation should be handled by caller
		}
	}

	return nil
}

func (d *displayDestination) Close() error {
	if d.window != nil {
		d.window.Close()
		d.window = nil
	}
	d.started = false
	return nil
}

