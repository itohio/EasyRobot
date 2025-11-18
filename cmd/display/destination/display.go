package destination

import (
	"context"
	"fmt"
	"log/slog"
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
	cancel  context.CancelFunc
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
		slog.Warn("Display destination already started")
		return fmt.Errorf("display destination already started")
	}
	if noDisplay {
		slog.Info("Display destination disabled (--no-display)")
		d.started = true
		return nil // Display is disabled
	}

	slog.Info("Starting display destination", "title", title, "width", width, "height", height)
	d.ctx = ctx
	// Store cancel function if available (from context.WithCancel)
	if cancel, ok := ctx.Value("cancel").(context.CancelFunc); ok {
		d.cancel = cancel
	}
	d.started = true
	slog.Info("Display destination started (window will be created on first frame)")
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
		slog.Info("Creating display window", "title", winTitle, "width", width, "height", height)
		d.window = cv.NewWindow(winTitle)
		if width > 0 && height > 0 {
			slog.Debug("Resizing window", "width", width, "height", height)
			d.window.ResizeWindow(width, height)
		}
		slog.Info("Display window created", "is_open", d.window.IsOpen())
	})
}

func (d *displayDestination) AddFrame(frame types.Frame) error {
	if !d.started || noDisplay {
		return nil // Display is disabled or not started
	}

	if d.stopped {
		slog.Debug("Display destination stopped, ignoring frame", "frame_index", frame.Index)
		return nil
	}

	if len(frame.Tensors) == 0 {
		slog.Debug("Frame has no tensors, skipping display", "frame_index", frame.Index)
		return nil
	}

	d.ensureWindow()
	if d.window == nil || !d.window.IsOpen() {
		slog.Warn("Display window is not open, stopping", "frame_index", frame.Index)
		d.stopped = true
		// Cancel context to signal stop to main loop
		if d.cancel != nil {
			slog.Info("Cancelling context due to window closed")
			d.cancel()
		}
		return nil
	}

	// Convert tensor to Mat
	slog.Debug("Converting tensor to Mat for display", "frame_index", frame.Index)
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

	slog.Debug("Displaying frame", "frame_index", frame.Index, "mat_size", mat.Size())
	if err := d.window.IMShow(mat); err != nil {
		slog.Error("Failed to display frame", "frame_index", frame.Index, "err", err)
		return err
	}

	// Check for keyboard input
	key := d.window.WaitKey(1)
	if key == 27 { // ESC
		slog.Info("ESC key pressed, stopping display", "frame_index", frame.Index)
		d.stopped = true
		// Cancel context to signal stop to main loop
		if d.cancel != nil {
			slog.Info("Cancelling context due to ESC key")
			d.cancel()
		}
	}

	return nil
}

func (d *displayDestination) Close() error {
	slog.Info("Closing display destination")
	if d.window != nil {
		slog.Debug("Closing display window")
		d.window.Close()
		d.window = nil
	}
	d.started = false
	slog.Info("Display destination closed")
	return nil
}

