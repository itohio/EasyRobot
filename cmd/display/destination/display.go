package destination

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

var (
	noDisplay bool
	title     string
	width     int
	height    int
)

// displayDestination implements Destination for displaying frames in a window.
// It uses the marshaller's display writer to handle window management and event processing.
type displayDestination struct {
	ctx          context.Context
	cancel       context.CancelFunc
	parentCtx    context.Context
	parentCancel context.CancelFunc
	displaySink  *gocv.DisplaySink
	started      bool
	eventLoopWg  sync.WaitGroup
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

	// Store parent context and create a cancellable context for the display destination
	d.parentCtx = ctx
	// Try to get parent cancel function if available
	if parentCancel, ok := ctx.Value("cancel").(context.CancelFunc); ok {
		d.parentCancel = parentCancel
	}
	d.ctx, d.cancel = context.WithCancel(ctx)

	// Create display sink using marshaller's display writer
	// The window is created immediately so the event loop can start right away
	displaySink, err := gocv.NewDisplaySinkFromOptions(d.ctx, title, width, height)
	if err != nil {
		return fmt.Errorf("failed to create display sink: %w", err)
	}
	d.displaySink = displaySink

	// The global event loop is started automatically when DisplaySink is created
	// We just need to monitor for window close events
	slog.Info("Display sink created, global event loop will handle window events")

	// Monitor for window close (poll window state)
	d.eventLoopWg.Add(1)
	go func() {
		defer d.eventLoopWg.Done()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-d.ctx.Done():
				return
			case <-ticker.C:
				window := d.displaySink.Window()
				if window == nil || !window.IsOpen() {
					slog.Info("Display window closed, cancelling parent context")
					if d.parentCancel != nil {
						d.parentCancel()
					}
					return
				}
			}
		}
	}()

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

	// Stop monitoring goroutine first
	if d.cancel != nil {
		d.cancel()
	}

	// Wait for monitoring goroutine to finish (with timeout)
	done := make(chan struct{})
	go func() {
		d.eventLoopWg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Monitoring goroutine finished
	case <-time.After(200 * time.Millisecond):
		slog.Debug("Monitoring goroutine did not finish within timeout")
	}

	// Close display sink (which closes the window)
	// This is non-blocking and won't hang even if event loop is shutting down
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
