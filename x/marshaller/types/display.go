package types

import "context"

// DisplayWindow represents a display window for showing visual content
type DisplayWindow interface {
	// ID returns a unique identifier for this window
	ID() string

	// Title returns the window title
	Title() string

	// Size returns the window dimensions
	Size() (width, height int)

	// Show makes the window visible
	Show()

	// Hide makes the window invisible
	Hide()

	// Close closes the window and releases resources
	Close() error

	// IsOpen returns true if the window is still open
	IsOpen() bool
}

// DisplayManager manages display windows and rendering
type DisplayManager interface {
	// CreateWindow creates a new display window
	CreateWindow(title string, width, height int) (DisplayWindow, error)

	// GetWindow returns an existing window by ID
	GetWindow(id string) (DisplayWindow, bool)

	// CloseAll closes all managed windows
	CloseAll() error

	// WaitForEvents waits for and processes display events
	WaitForEvents(timeout int) bool

	// PollEvents polls for and processes pending display events
	PollEvents()
}

// KeyEvent represents a keyboard event
type KeyEvent struct {
	// Key is the key code (implementation-specific)
	Key int

	// Action indicates press, release, or repeat
	Action KeyAction

	// Modifiers indicates modifier keys (shift, ctrl, alt, etc.)
	Modifiers KeyModifier

	// Timestamp is when the event occurred (nanoseconds)
	Timestamp int64
}

// KeyAction represents the type of key event
type KeyAction int

const (
	KeyPress KeyAction = iota
	KeyRelease
	KeyRepeat
)

// KeyModifier represents modifier key states
type KeyModifier uint32

const (
	ModShift KeyModifier = 1 << iota
	ModCtrl
	ModAlt
	ModSuper // Windows/Cmd key
)

// MouseEvent represents a mouse event
type MouseEvent struct {
	// X, Y are the mouse coordinates
	X, Y int

	// Button indicates which mouse button (if any)
	Button MouseButton

	// Action indicates press, release, click, etc.
	Action MouseAction

	// Modifiers indicates modifier keys
	Modifiers KeyModifier

	// ScrollDelta indicates scroll wheel movement (X, Y)
	ScrollDeltaX, ScrollDeltaY int

	// Timestamp is when the event occurred (nanoseconds)
	Timestamp int64
}

// MouseButton represents mouse button identifiers
type MouseButton int

const (
	MouseButtonLeft MouseButton = iota
	MouseButtonRight
	MouseButtonMiddle
	MouseButton4 // Extra buttons
	MouseButton5
	MouseButtonNone // For motion events without button press
)

// MouseAction represents the type of mouse event
type MouseAction int

const (
	MousePress MouseAction = iota
	MouseRelease
	MouseMove
	MouseScroll
	MouseEnter
	MouseLeave
)

// InputEvent represents a generic input event
type InputEvent struct {
	// Type indicates the event category
	Type InputEventType

	// Key event data (valid when Type == InputEventKey)
	Key KeyEvent

	// Mouse event data (valid when Type == InputEventMouse)
	Mouse MouseEvent

	// WindowID identifies which window this event is for
	WindowID string

	// Timestamp is when the event occurred (nanoseconds)
	Timestamp int64
}

// InputEventType categorizes input events
type InputEventType int

const (
	InputEventKey InputEventType = iota
	InputEventMouse
	InputEventWindow
	InputEventUnknown
)

// InputHandler processes input events
type InputHandler interface {
	// HandleKey processes a keyboard event
	// Return false to stop event processing
	HandleKey(event KeyEvent) bool

	// HandleMouse processes a mouse event
	// Return false to stop event processing
	HandleMouse(event MouseEvent) bool

	// HandleWindow processes a window event
	// Return false to stop event processing
	HandleWindow(event WindowEvent) bool
}

// WindowEvent represents a window management event
type WindowEvent struct {
	// Action indicates what happened to the window
	Action WindowAction

	// Width, Height are the new window dimensions (for resize events)
	Width, Height int

	// Timestamp is when the event occurred (nanoseconds)
	Timestamp int64
}

// WindowAction represents window management actions
type WindowAction int

const (
	WindowClose WindowAction = iota
	WindowResize
	WindowMove
	WindowFocus
	WindowUnfocus
	WindowMinimize
	WindowMaximize
	WindowRestore
)

// EventLoop represents a custom event processing loop
type EventLoop func(ctx context.Context, shouldContinue func() bool)

// DisplayOption configures display behavior
type DisplayOption interface {
	Apply(*DisplayOptions)
}

// DisplayOptions holds display configuration
type DisplayOptions struct {
	// Enabled indicates if display output is enabled
	Enabled bool

	// Title is the window title
	Title string

	// Width, Height are the window dimensions
	Width, Height int

	// KeyHandler processes keyboard events
	KeyHandler func(KeyEvent) bool

	// MouseHandler processes mouse events
	MouseHandler func(MouseEvent) bool

	// WindowHandler processes window events
	WindowHandler func(WindowEvent) bool

	// EventLoop provides custom event processing (optional)
	EventLoop EventLoop

	// Context for cancellation
	Context context.Context
}

// Common key codes (implementation may map these differently)
const (
	KeyUnknown = 0
	KeySpace   = 32
	KeyEscape  = 27
	KeyEnter   = 13
	KeyTab     = 9
	KeyBackspace = 8
	KeyDelete  = 127

	// Arrow keys
	KeyArrowLeft  = 81  // Usually left arrow
	KeyArrowUp    = 82  // Usually up arrow
	KeyArrowRight = 83  // Usually right arrow
	KeyArrowDown  = 84  // Usually down arrow

	// Function keys
	KeyF1  = 112
	KeyF2  = 113
	KeyF3  = 114
	KeyF4  = 115
	KeyF5  = 116
	KeyF6  = 117
	KeyF7  = 118
	KeyF8  = 119
	KeyF9  = 120
	KeyF10 = 121
	KeyF11 = 122
	KeyF12 = 123

	// Number keys (0-9)
	Key0 = 48
	Key1 = 49
	Key2 = 50
	Key3 = 51
	Key4 = 52
	Key5 = 53
	Key6 = 54
	Key7 = 55
	Key8 = 56
	Key9 = 57

	// Letter keys (A-Z)
	KeyA = 65
	KeyB = 66
	KeyC = 67
	KeyD = 68
	KeyE = 69
	KeyF = 70
	KeyG = 71
	KeyH = 72
	KeyI = 73
	KeyJ = 74
	KeyK = 75
	KeyL = 76
	KeyM = 77
	KeyN = 78
	KeyO = 79
	KeyP = 80
	KeyQ = 81
	KeyR = 82
	KeyS = 83
	KeyT = 84
	KeyU = 85
	KeyV = 86
	KeyW = 87
	KeyX = 88
	KeyY = 89
	KeyZ = 90
)

// Display option implementations

type withDisplayEnabled struct{}

func (opt withDisplayEnabled) Apply(opts *DisplayOptions) {
	opts.Enabled = true
}

// WithDisplayEnabled enables display output
func WithDisplayEnabled() DisplayOption {
	return withDisplayEnabled{}
}

type withDisplayTitle string

func (opt withDisplayTitle) Apply(opts *DisplayOptions) {
	opts.Enabled = true
	opts.Title = string(opt)
}

// WithDisplayTitle sets the display window title and enables display
func WithDisplayTitle(title string) DisplayOption {
	return withDisplayTitle(title)
}

type withDisplaySize struct{ width, height int }

func (opt withDisplaySize) Apply(opts *DisplayOptions) {
	opts.Enabled = true
	opts.Width = opt.width
	opts.Height = opt.height
}

// WithDisplaySize sets the display window size and enables display
func WithDisplaySize(width, height int) DisplayOption {
	return withDisplaySize{width: width, height: height}
}

type withKeyHandler struct{ handler func(KeyEvent) bool }

func (opt withKeyHandler) Apply(opts *DisplayOptions) {
	opts.Enabled = true
	opts.KeyHandler = opt.handler
}

// WithKeyHandler installs a keyboard event handler
func WithKeyHandler(handler func(KeyEvent) bool) DisplayOption {
	return withKeyHandler{handler: handler}
}

type withMouseHandler struct{ handler func(MouseEvent) bool }

func (opt withMouseHandler) Apply(opts *DisplayOptions) {
	opts.Enabled = true
	opts.MouseHandler = opt.handler
}

// WithMouseHandler installs a mouse event handler
func WithMouseHandler(handler func(MouseEvent) bool) DisplayOption {
	return withMouseHandler{handler: handler}
}

type withWindowHandler struct{ handler func(WindowEvent) bool }

func (opt withWindowHandler) Apply(opts *DisplayOptions) {
	opts.Enabled = true
	opts.WindowHandler = opt.handler
}

// WithWindowHandler installs a window event handler
func WithWindowHandler(handler func(WindowEvent) bool) DisplayOption {
	return withWindowHandler{handler: handler}
}

type withEventLoop struct{ loop EventLoop }

func (opt withEventLoop) Apply(opts *DisplayOptions) {
	opts.EventLoop = opt.loop
}

// WithEventLoop sets a custom event processing loop
func WithEventLoop(loop EventLoop) DisplayOption {
	return withEventLoop{loop: loop}
}

type withDisplayContext struct{ ctx context.Context }

func (opt withDisplayContext) Apply(opts *DisplayOptions) {
	opts.Context = opt.ctx
}

// WithDisplayContext sets the context for display operations
func WithDisplayContext(ctx context.Context) DisplayOption {
	return withDisplayContext{ctx: ctx}
}
