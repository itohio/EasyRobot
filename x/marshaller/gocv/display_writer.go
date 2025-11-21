package gocv

import (
	"context"
	"fmt"
	"log/slog"
	"runtime"
	"sync"
	"time"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Window event types (legacy - use types.WindowEvent instead)
type windowEventType int

const (
	eventKey windowEventType = iota
	eventMouse
	eventClose
)

// Internal window event for legacy gocv event handling
type internalWindowEvent struct {
	WindowID string
	Type     windowEventType
	Key      int
	Mouse    struct {
		Event int
		X     int
		Y     int
		Flags int
	}
}

// GUI command types for the event loop
type guiCommandType int

const (
	cmdCreateWindow guiCommandType = iota
	cmdShowImage
	cmdCloseWindow
	cmdStop
)

// GUICommand is sent to the event loop thread
type GUICommand struct {
	Type     guiCommandType
	WindowID string
	Config   *windowConfig
	Image    cv.Mat
	Result   chan error
}

type windowConfig struct {
	Title        string
	Width        int
	Height       int
	OnKey        func(int) bool // Legacy handlers (for backward compatibility)
	OnMouse      func(event, x, y, flags int) bool
	KeyHandler   func(types.KeyEvent) bool // New shared interface handlers
	MouseHandler func(types.MouseEvent) bool
	CancelFunc   context.CancelFunc // Cancel function to call if window close callback returns true
	OnClose      CloseCallback      // Callback called when this window closes
}

// WindowInfo tracks a window in the event loop
type windowInfo struct {
	id           string
	window       *cv.Window
	config       windowConfig
	onKey        func(int) bool                    // Legacy handler
	onMouse      func(event, x, y, flags int) bool // Legacy handler
	keyHandler   func(types.KeyEvent) bool         // New shared interface handler
	mouseHandler func(types.MouseEvent) bool       // New shared interface handler
	cancelFunc   context.CancelFunc                // Cancel function if callback returns true
	onClose      CloseCallback                     // Callback called when this window closes
}

// Global event loop - single thread for all GUI operations
var (
	eventLoopOnce   sync.Once
	guiCommands     chan GUICommand
	windowEvents    chan internalWindowEvent
	windowsMap      map[string]*windowInfo
	windowsMapMu    sync.RWMutex
	eventLoopCtx    context.Context
	eventLoopCancel context.CancelFunc
	eventLoopWg     sync.WaitGroup
)

type displayWriter struct {
	cfg      config
	windowID string
	stopped  bool
	mu       sync.Mutex
	ready    chan struct{} // Signals when window is ready
}

func newDisplayWriter(cfg config) (frameWriter, error) {
	// Start the global event loop if not already started
	eventLoopOnce.Do(func() {
		eventLoopCtx, eventLoopCancel = context.WithCancel(cfg.ctx)
		if eventLoopCtx == nil {
			eventLoopCtx = context.Background()
		}
		guiCommands = make(chan GUICommand, 100)
		windowEvents = make(chan internalWindowEvent, 100)
		windowsMap = make(map[string]*windowInfo)
		eventLoopWg.Add(1)
		go runGUIEventLoop()
	})

	// Generate unique window ID
	windowID := generateWindowID()

	dw := &displayWriter{
		cfg:      cfg,
		windowID: windowID,
		ready:    make(chan struct{}),
	}

	// Send create window command to event loop
	title := cfg.display.title
	if title == "" {
		title = "GoCV Display"
	}

	result := make(chan error, 1)
	// Create adapter functions to convert between shared types and legacy gocv callbacks
	var onKey func(int) bool
	if cfg.display.onKey != nil {
		onKey = func(key int) bool {
			return cfg.display.onKey(types.KeyEvent{Key: key})
		}
	}
	var onMouse func(event, x, y, flags int) bool
	if cfg.display.onMouse != nil {
		onMouse = func(event, x, y, flags int) bool {
			// Convert gocv mouse event to shared MouseEvent
			var action types.MouseAction
			var button types.MouseButton

			// Map gocv event types to shared actions (using OpenCV standard values)
			// 0 = mouse move, 1 = left down, 2 = left up, 3 = right down, 4 = right up, etc.
			switch event {
			case 0: // CV_EVENT_MOUSEMOVE
				action = types.MouseMove
				button = types.MouseButtonNone
			case 1: // CV_EVENT_LBUTTONDOWN
				action = types.MousePress
				button = types.MouseButtonLeft
			case 2: // CV_EVENT_LBUTTONUP
				action = types.MouseRelease
				button = types.MouseButtonLeft
			case 3: // CV_EVENT_RBUTTONDOWN
				action = types.MousePress
				button = types.MouseButtonRight
			case 4: // CV_EVENT_RBUTTONUP
				action = types.MouseRelease
				button = types.MouseButtonRight
			case 5: // CV_EVENT_MBUTTONDOWN
				action = types.MousePress
				button = types.MouseButtonMiddle
			case 6: // CV_EVENT_MBUTTONUP
				action = types.MouseRelease
				button = types.MouseButtonMiddle
			default:
				action = types.MouseMove
				button = types.MouseButtonNone
			}

			return cfg.display.onMouse(types.MouseEvent{
				X:      x,
				Y:      y,
				Button: button,
				Action: action,
			})
		}
	}

	cmd := GUICommand{
		Type:     cmdCreateWindow,
		WindowID: windowID,
		Config: &windowConfig{
			Title:        title,
			Width:        cfg.display.width,
			Height:       cfg.display.height,
			OnKey:        onKey,
			OnMouse:      onMouse,
			KeyHandler:   cfg.display.onKey,
			MouseHandler: cfg.display.onMouse,
			CancelFunc:   cfg.display.cancelFunc,
			OnClose:      cfg.display.onClose,
		},
		Result: result,
	}

	select {
	case guiCommands <- cmd:
		// Command sent
	case <-time.After(1 * time.Second):
		return nil, fmt.Errorf("failed to send create window command")
	}

	// Wait for window creation result
	select {
	case err := <-result:
		if err != nil {
			return nil, fmt.Errorf("failed to create window: %w", err)
		}
		close(dw.ready)
	case <-time.After(2 * time.Second):
		return nil, fmt.Errorf("window creation timeout")
	}

	return dw, nil
}

func generateWindowID() string {
	// Simple ID generation - in production might want something more robust
	return fmt.Sprintf("window_%d", time.Now().UnixNano())
}

// runGUIEventLoop runs the single GUI event loop on a locked OS thread
// All window creation and drawing happens in this thread
func runGUIEventLoop() {
	defer eventLoopWg.Done()

	// Lock this goroutine to the OS thread - REQUIRED for OpenCV GUI operations
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	slog.Debug("GUI event loop started on locked OS thread")

	ticker := time.NewTicker(4 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-eventLoopCtx.Done():
			slog.Debug("GUI event loop cancelled")
			// Close all windows
			windowsMapMu.Lock()
			for _, win := range windowsMap {
				if win.window != nil && win.window.IsOpen() {
					win.window.Close()
				}
			}
			windowsMap = nil
			windowsMapMu.Unlock()
			// Drain any remaining commands to prevent senders from blocking
			for {
				select {
				case cmd := <-guiCommands:
					// Close any images in commands
					if !cmd.Image.Empty() {
						cmd.Image.Close()
					}
					if cmd.Result != nil {
						select {
						case cmd.Result <- fmt.Errorf("event loop shutting down"):
						default:
						}
					}
				default:
					return
				}
			}

		case cmd := <-guiCommands:
			// Handle GUI commands
			switch cmd.Type {
			case cmdCreateWindow:
				err := handleCreateWindow(cmd)
				if cmd.Result != nil {
					cmd.Result <- err
				}
			case cmdShowImage:
				err := handleShowImage(cmd)
				if cmd.Result != nil {
					cmd.Result <- err
				}
			case cmdCloseWindow:
				handleCloseWindow(cmd)
				if cmd.Result != nil {
					cmd.Result <- nil
				}
			case cmdStop:
				if eventLoopCancel != nil {
					eventLoopCancel()
				}
				return
			}

		case <-ticker.C:
			// Process window events (WaitKey, mouse events, etc.)
			processWindowEvents()
		}
	}
}

func handleCreateWindow(cmd GUICommand) error {
	if cmd.Config == nil {
		return fmt.Errorf("window config is nil")
	}

	slog.Debug("Creating window in event loop thread", "id", cmd.WindowID, "title", cmd.Config.Title)

	window := cv.NewWindow(cmd.Config.Title)
	if !window.IsOpen() {
		return fmt.Errorf("failed to create window")
	}

	if cmd.Config.Width > 0 && cmd.Config.Height > 0 {
		window.ResizeWindow(cmd.Config.Width, cmd.Config.Height)
	}

	winInfo := &windowInfo{
		id:           cmd.WindowID,
		window:       window,
		config:       *cmd.Config,
		onKey:        cmd.Config.OnKey,
		onMouse:      cmd.Config.OnMouse,
		keyHandler:   cmd.Config.KeyHandler,
		mouseHandler: cmd.Config.MouseHandler,
		cancelFunc:   cmd.Config.CancelFunc,
		onClose:      cmd.Config.OnClose,
	}

	// Set mouse handler if provided
	if cmd.Config.OnMouse != nil {
		window.SetMouseHandler(func(event int, x int, y int, flags int, userData interface{}) {
			winID := userData.(string)
			evt := internalWindowEvent{
				WindowID: winID,
				Type:     eventMouse,
			}
			evt.Mouse.Event = event
			evt.Mouse.X = x
			evt.Mouse.Y = y
			evt.Mouse.Flags = flags

			select {
			case windowEvents <- evt:
			default:
				// Event channel full, drop event
			}
		}, cmd.WindowID)
	}

	// Show blank image to initialize window
	blankMat := cv.NewMatWithSize(100, 100, cv.MatTypeCV8UC3)
	blankMat.SetTo(cv.NewScalar(0, 0, 0, 0))
	window.IMShow(blankMat)
	window.WaitKey(1)
	blankMat.Close()

	windowsMapMu.Lock()
	windowsMap[cmd.WindowID] = winInfo
	windowsMapMu.Unlock()

	slog.Debug("Window created", "id", cmd.WindowID, "is_open", window.IsOpen())
	return nil
}

func handleShowImage(cmd GUICommand) error {
	windowsMapMu.RLock()
	winInfo, exists := windowsMap[cmd.WindowID]
	windowsMapMu.RUnlock()

	if !exists || winInfo.window == nil {
		cmd.Image.Close() // Close mat if window not found
		return fmt.Errorf("window not found: %s", cmd.WindowID)
	}

	if !winInfo.window.IsOpen() {
		cmd.Image.Close() // Close mat if window is closed
		return fmt.Errorf("window is closed: %s", cmd.WindowID)
	}

	if err := winInfo.window.IMShow(cmd.Image); err != nil {
		cmd.Image.Close() // Close mat on error
		return fmt.Errorf("failed to show image: %w", err)
	}

	// Close mat after IMShow (IMShow copies the data)
	cmd.Image.Close()

	// WaitKey is called in processWindowEvents, but we call it here too
	// to ensure immediate display update
	winInfo.window.WaitKey(1)

	return nil
}

func handleCloseWindow(cmd GUICommand) {
	windowsMapMu.Lock()
	defer windowsMapMu.Unlock()

	winInfo, exists := windowsMap[cmd.WindowID]
	if !exists {
		return
	}

	if winInfo.window != nil && winInfo.window.IsOpen() {
		winInfo.window.Close()
	}

	// Calculate remaining windows before deleting this one
	remainingWindows := len(windowsMap) - 1

	// Call close callback if available, passing actual window and remaining count
	shouldCancel := false
	if winInfo.onClose != nil && winInfo.window != nil {
		shouldCancel = winInfo.onClose(winInfo.window, remainingWindows)
	}

	// Remove window from map
	delete(windowsMap, cmd.WindowID)

	// Cancel context if callback indicated we should terminate
	if shouldCancel && winInfo.cancelFunc != nil {
		slog.Info("Window close callback requested termination, cancelling context",
			"window_id", cmd.WindowID,
			"remaining_windows", remainingWindows)
		winInfo.cancelFunc()
	}
}

func processWindowEvents() {
	windowsMapMu.RLock()
	if len(windowsMap) == 0 {
		windowsMapMu.RUnlock()
		return
	}

	// Get first active window for WaitKey (OpenCV processes all windows with one call)
	var firstWindow *cv.Window
	for _, winInfo := range windowsMap {
		if winInfo.window != nil && winInfo.window.IsOpen() {
			firstWindow = winInfo.window
			break
		}
	}
	windowsMapMu.RUnlock()

	if firstWindow == nil {
		return
	}

	// Process keyboard events
	key := firstWindow.WaitKey(1)
	if key >= 0 {
		// Check all windows for key handlers
		windowsMapMu.RLock()
		for _, winInfo := range windowsMap {
			if winInfo.window != nil && winInfo.window.IsOpen() {
				// Call new key handler if available
				if winInfo.keyHandler != nil {
					keyEvent := types.KeyEvent{
						Key:       key,
						Action:    types.KeyPress,
						Timestamp: time.Now().UnixNano(),
					}
					if !winInfo.keyHandler(keyEvent) {
						// Handler returned false, stop
						windowsMapMu.RUnlock()
						if eventLoopCancel != nil {
							eventLoopCancel()
						}
						return
					}
				}
				// Call legacy key handler for backward compatibility
				if winInfo.onKey != nil {
					if !winInfo.onKey(key) {
						// Handler returned false, stop
						windowsMapMu.RUnlock()
						if eventLoopCancel != nil {
							eventLoopCancel()
						}
						return
					}
				} else if key == 27 {
					// ESC key - default behavior: stop all
					slog.Info("ESC key pressed, stopping all displays")
					windowsMapMu.RUnlock()
					if eventLoopCancel != nil {
						eventLoopCancel()
					}
					return
				}
			}
		}
		windowsMapMu.RUnlock()
	}

	// Process mouse events from channel (non-blocking)
	processedAny := false
	for {
		select {
		case evt := <-windowEvents:
			processedAny = true
			windowsMapMu.RLock()
			winInfo, exists := windowsMap[evt.WindowID]
			windowsMapMu.RUnlock()

			if exists {
				// Call new mouse handler if available
				if winInfo.mouseHandler != nil {
					mouseEvent := types.MouseEvent{
						X:         evt.Mouse.X,
						Y:         evt.Mouse.Y,
						Action:    types.MouseMove, // Default to move, could be enhanced
						Timestamp: time.Now().UnixNano(),
					}
					if !winInfo.mouseHandler(mouseEvent) {
						// Handler returned false, stop
						if eventLoopCancel != nil {
							eventLoopCancel()
						}
						return
					}
				}
				// Call legacy mouse handler for backward compatibility
				if winInfo.onMouse != nil {
					if !winInfo.onMouse(evt.Mouse.Event, evt.Mouse.X, evt.Mouse.Y, evt.Mouse.Flags) {
						// Handler returned false, stop
						if eventLoopCancel != nil {
							eventLoopCancel()
						}
						return
					}
				}
			}
		default:
			// No more events
			if !processedAny {
				// Check for closed windows only if we didn't process any mouse events
				windowsMapMu.Lock()
				// Collect closed windows first to calculate remaining count correctly
				var closedWindows []struct {
					id             string
					winInfo        *windowInfo
					remainingCount int
				}

				for id, winInfo := range windowsMap {
					if winInfo.window != nil && !winInfo.window.IsOpen() {
						remainingCount := len(windowsMap) - 1 - len(closedWindows)
						closedWindows = append(closedWindows, struct {
							id             string
							winInfo        *windowInfo
							remainingCount int
						}{id, winInfo, remainingCount})
					}
				}
				windowsMapMu.Unlock()

				// Process closed windows outside the lock
				for _, closed := range closedWindows {
					shouldCancel := false
					if closed.winInfo.onClose != nil && closed.winInfo.window != nil {
						shouldCancel = closed.winInfo.onClose(closed.winInfo.window, closed.remainingCount)
					}

					// Remove from map
					windowsMapMu.Lock()
					delete(windowsMap, closed.id)
					windowsMapMu.Unlock()

					// Cancel context if callback indicated we should terminate
					if shouldCancel && closed.winInfo.cancelFunc != nil {
						slog.Info("Window close callback requested termination, cancelling context",
							"window_id", closed.id,
							"remaining_windows", closed.remainingCount)
						closed.winInfo.cancelFunc()
					}
				}
			}
			return
		}
	}
}

// WriteFrame implements StreamSink interface.
func (dw *displayWriter) WriteFrame(frame types.Frame) error {
	return dw.Write(frame)
}

// Write displays a frame (legacy method name, kept for backward compatibility).
func (dw *displayWriter) Write(frame types.Frame) error {
	dw.mu.Lock()
	stopped := dw.stopped
	dw.mu.Unlock()

	if stopped {
		return errStopLoop
	}
	if len(frame.Tensors) == 0 {
		return nil
	}

	// Wait for window to be ready
	select {
	case <-dw.ready:
		// Window is ready
	case <-time.After(2 * time.Second):
		return fmt.Errorf("window not ready")
	}

	// Convert tensor to Mat
	mat, err := tensorToMat(frame.Tensors[0])
	if err != nil {
		return err
	}
	// Don't close mat here - it will be closed in the event loop after IMShow

	// Release objects that implement Releaser after displaying if WithRelease is enabled
	if dw.cfg.ReleaseAfterProcessing {
		defer func() {
			for _, tensor := range frame.Tensors {
				if releaser, ok := tensor.(types.Releaser); ok {
					releaser.Release()
				}
			}
		}()
	}

	// Send show image command to event loop
	result := make(chan error, 1)
	cmd := GUICommand{
		Type:     cmdShowImage,
		WindowID: dw.windowID,
		Image:    mat,
		Result:   result,
	}

	select {
	case guiCommands <- cmd:
		// Command sent, wait for result
		// Mat will be closed in handleShowImage after IMShow
		select {
		case err := <-result:
			if err != nil {
				return fmt.Errorf("failed to show image: %w", err)
			}
		case <-time.After(1 * time.Second):
			mat.Close() // Close on timeout
			return fmt.Errorf("show image command timeout")
		}
	case <-time.After(1 * time.Second):
		mat.Close() // Close if we can't send command
		return fmt.Errorf("failed to send show image command")
	}

	return nil
}

func (dw *displayWriter) Close() error {
	dw.mu.Lock()
	alreadyStopped := dw.stopped
	dw.stopped = true
	dw.mu.Unlock()

	if alreadyStopped {
		return nil // Already closed
	}

	// Try to send close window command to event loop (non-blocking with timeout)
	result := make(chan error, 1)
	cmd := GUICommand{
		Type:     cmdCloseWindow,
		WindowID: dw.windowID,
		Result:   result,
	}

	select {
	case guiCommands <- cmd:
		// Command sent, wait for result with timeout
		select {
		case <-result:
			// Close completed
		case <-time.After(500 * time.Millisecond):
			// Timeout - event loop might be shutting down
			slog.Debug("Close window command timeout, event loop may be shutting down")
		}
	case <-time.After(100 * time.Millisecond):
		// Can't send command - event loop might be shutting down or channel full
		slog.Debug("Failed to send close window command, event loop may be shutting down")
	}

	return nil
}

// RunEventLoop is a no-op - the global event loop handles all windows
// This method exists for API compatibility but does nothing
func (dw *displayWriter) RunEventLoop(ctx context.Context) {
	// The global event loop handles all windows, so this is a no-op
	// Just wait for the context to be cancelled
	<-ctx.Done()
}

// Window returns the underlying gocv window. This allows external code
// to access the window for additional operations if needed.
// Note: Accessing the window from outside the event loop thread is not thread-safe.
// This method should only be used for read-only operations or with proper synchronization.
func (dw *displayWriter) Window() *cv.Window {
	windowsMapMu.RLock()
	defer windowsMapMu.RUnlock()
	winInfo, exists := windowsMap[dw.windowID]
	if !exists {
		return nil
	}
	return winInfo.window
}
