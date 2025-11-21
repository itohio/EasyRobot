package destination

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	cv "gocv.io/x/gocv"
)

var (
	noDisplay bool
	title     string
	width     int
	height    int
)

// CancelSetter is an interface for destinations that can accept a cancel function
// to be called when certain events occur (e.g., window close).
type CancelSetter interface {
	SetCancelFunc(cancel context.CancelFunc)
}

// displayDestination implements Destination for displaying frames in a window.
// It uses the marshaller's display writer to handle window management and event processing.
type displayDestination struct {
	ctx          context.Context
	cancel       context.CancelFunc
	parentCtx    context.Context
	parentCancel context.CancelFunc // Cancel function from parent context - called when main window closes
	displaySink  *gocv.DisplaySink
	started      bool
}

// NewDisplay creates a new display destination.
func NewDisplay() Destination {
	return &displayDestination{}
}

// SetCancelFunc sets the cancel function to be called when the main window is closed.
// This allows the application to be notified and cleanly shut down when the user closes
// the main display window. Call this before Start() if you want window close to cancel
// a parent context.
func (d *displayDestination) SetCancelFunc(cancel context.CancelFunc) {
	d.parentCancel = cancel
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

	// Store parent context and create a cancellable context for the display destination
	d.parentCtx = ctx
	d.ctx, d.cancel = context.WithCancel(ctx)

	// Try to extract parent cancel function - if the parent context was created with WithCancel,
	// we need to get that cancel function to cancel it when main window closes.
	// Since we can't use context.Value, we'll use a type assertion or interface.
	// For now, we'll derive a context from parent and store its cancel.
	// The caller (main.go) should pass cancel via a proper mechanism.
	// TODO: Consider adding a WithParentCancel option to Start() method or create a new interface

	// Create display sink using marshaller's display writer
	// Pass cancel function and window close callback so user can decide when to terminate
	var cancelFunc context.CancelFunc
	if d.parentCancel != nil {
		// Use parent cancel function if available - it will cancel the parent context
		cancelFunc = d.parentCancel
	} else {
		// Fallback: cancel our derived context if parent cancel not set
		cancelFunc = d.cancel
	}

	// Close callback - user decides when to terminate based on window info
	onClose := func(window *cv.Window, remainingWindows int) bool {
		// Query window information to make proper decision
		// window can be queried for properties if needed

		slog.Info("Window closed",
			"remaining_windows", remainingWindows)

		// Terminate when last window closes
		if remainingWindows == 0 {
			slog.Info("Last window closed, terminating application")
			return true
		}
		// Keep running if other windows are still open
		return false
	}

	// Key handler - ESC key is treated as a regular keyboard event (not special)
	// Users can decide what to do with ESC by setting their own handler.
	// This is a convenience handler for cmd/display that cancels on ESC.
	// To customize ESC behavior, users can create their own DisplaySink with WithOnKey().
	onKey := func(event types.KeyEvent) bool {
		if event.Key == 27 { // ESC key
			slog.Info("ESC key pressed, cancelling context")
			if cancelFunc != nil {
				cancelFunc()
			}
			return false // Stop event processing
		}
		// Other keys are ignored by default - return true to continue processing
		return true
	}

	displaySink, err := gocv.NewDisplaySinkFromOptions(
		d.ctx,
		title,
		width,
		height,
		gocv.WithCancel(cancelFunc), // Cancel function called if callback returns true
		gocv.WithOnClose(onClose),   // Callback decides when to terminate
		gocv.WithOnKey(onKey),       // Key handler - ESC cancels context, other keys ignored
		types.WithRelease(),         // Release tensors after displaying
	)
	if err != nil {
		return fmt.Errorf("failed to create display sink: %w", err)
	}
	d.displaySink = displaySink

	// The global event loop is started automatically when DisplaySink is created
	// Window close detection is handled by the marshaller's event loop which will
	// call the cancel function (if set via SetCancelFunc) when the main window closes
	slog.Info("Display sink created, global event loop will handle window events and cancellation")

	d.started = true
	slog.Info("Display destination started (window will be created on first frame)")
	return nil
}

func (d *displayDestination) AddFrame(frame types.Frame) error {
	if !d.started || noDisplay {
		return nil // Display is disabled or not started
	}

	if d.displaySink == nil {
		return fmt.Errorf("display sink not initialized")
	}

	// Use marshaller's display sink to write the frame
	// The event loop is already running in a goroutine
	return d.displaySink.WriteFrame(frame)
}

func (d *displayDestination) Close() error {
	slog.Info("Closing display destination")

	// Cancel our derived context to signal shutdown
	if d.cancel != nil {
		d.cancel()
	}

	// Close display sink (which closes the window)
	// Window close detection is handled by the marshaller's event loop,
	// so we don't need a separate monitoring goroutine
	if d.displaySink != nil {
		if err := d.displaySink.Close(); err != nil {
			slog.Error("Error closing display sink", "err", err)
		}
		d.displaySink = nil
	}

	d.started = false
	slog.Info("Display destination closed")
	return nil
}
