package types

import "context"

// CameraInfo contains information about a video capture device
type CameraInfo struct {
	// ID is the device identifier (implementation-specific)
	ID int

	// Path is the device node path (e.g., "/dev/video0")
	Path string

	// Name is the human-readable device name
	Name string

	// Driver is the kernel driver name
	Driver string

	// Card is the device description/card name
	Card string

	// BusInfo contains bus information
	BusInfo string

	// Version is the driver version (optional)
	Version string

	// Capabilities bitmap (implementation-specific)
	Capabilities any

	// SupportedFormats lists available video formats
	SupportedFormats []VideoFormat

	// Controls lists available camera controls
	Controls []ControlInfo

	// Metadata contains additional implementation-specific information
	Metadata map[string]any
}

// VideoFormat describes a supported video format
type VideoFormat struct {
	// PixelFormat is the pixel format identifier (implementation-specific)
	PixelFormat any

	// Description is a human-readable description
	Description string

	// Width in pixels
	Width int

	// Height in pixels
	Height int

	// Additional format-specific metadata
	Metadata map[string]any
}

// ControlInfo describes a camera control parameter
type ControlInfo struct {
	// ID is the control identifier (implementation-specific, can be string or enum)
	ID any

	// Name is the control name/key for programmatic access
	Name string

	// Description is a human-readable description
	Description string

	// Type indicates the control type ("integer", "boolean", "menu", etc.)
	Type string

	// Min/Max/Default values
	Min, Max, Default int32

	// Step size for integer controls
	Step int32

	// Menu items for menu controls (optional)
	MenuItems []string

	// Additional control-specific metadata
	Metadata map[string]any
}

// CameraController provides runtime control of camera parameters
type CameraController interface {
	// Controls returns information about available camera controls
	Controls() []ControlInfo

	// GetControl gets the current value of a camera control by name
	GetControl(name string) (int32, error)

	// SetControl sets the value of a camera control by name
	SetControl(name string, value int32) error

	// GetControls gets multiple control values atomically
	GetControls() (map[string]int32, error)

	// SetControls sets multiple control values atomically
	SetControls(controls map[string]int32) error
}

// CameraDevice represents a camera device that can be opened for streaming
type CameraDevice interface {
	// Info returns device information and capabilities
	Info() CameraInfo

	// Open opens the device with specified configuration
	Open(opts ...CameraOption) (CameraStream, error)

	// Close closes the device
	Close() error
}

// CameraStream represents an active camera capture stream
type CameraStream interface {
	// Start begins frame capture
	Start(ctx context.Context) error

	// Stop halts frame capture
	Stop() error

	// Controller returns the camera controller for runtime adjustments
	Controller() CameraController

	// Close closes the stream
	Close() error
}

// CameraOption configures camera device opening and streaming
type CameraOption interface {
	Apply(*CameraOptions)
}

// CameraOptions holds camera configuration
type CameraOptions struct {
	// Width/Height specify desired resolution
	Width, Height int

	// FrameRate specifies desired frame rate
	FrameRate int

	// PixelFormat specifies desired pixel format
	PixelFormat any

	// Controls specifies initial control values
	Controls map[string]int32

	// BufferCount specifies number of capture buffers
	BufferCount int

	// Additional implementation-specific options
	Metadata map[string]any
}

// Common camera control names (for consistency across implementations)
const (
	CameraControlBrightness            = "brightness"
	CameraControlContrast              = "contrast"
	CameraControlSaturation            = "saturation"
	CameraControlHue                   = "hue"
	CameraControlGamma                 = "gamma"
	CameraControlExposure              = "exposure"
	CameraControlGain                  = "gain"
	CameraControlSharpness             = "sharpness"
	CameraControlWhiteBalanceTemp      = "white_balance_temperature"
	CameraControlAutoWhiteBalance      = "auto_white_balance"
	CameraControlAutogain              = "autogain"
	CameraControlBacklightCompensation = "backlight_compensation"
	CameraControlPowerLineFrequency    = "power_line_frequency"
	CameraControlHFlip                 = "horizontal_flip"
	CameraControlVFlip                 = "vertical_flip"
)

// Camera option implementations

type withCameraResolution struct{ width, height int }

func (opt withCameraResolution) Apply(opts *CameraOptions) {
	opts.Width = opt.width
	opts.Height = opt.height
}

// WithCameraResolution sets the desired camera resolution
func WithCameraResolution(width, height int) CameraOption {
	return withCameraResolution{width: width, height: height}
}

type withCameraFrameRate int

func (opt withCameraFrameRate) Apply(opts *CameraOptions) {
	opts.FrameRate = int(opt)
}

// WithCameraFrameRate sets the desired camera frame rate
func WithCameraFrameRate(fps int) CameraOption {
	return withCameraFrameRate(fps)
}

type withCameraPixelFormat struct{ format any }

func (opt withCameraPixelFormat) Apply(opts *CameraOptions) {
	opts.PixelFormat = opt.format
}

// WithCameraPixelFormat sets the desired camera pixel format
func WithCameraPixelFormat(format any) CameraOption {
	return withCameraPixelFormat{format: format}
}

type withCameraControls map[string]int32

func (opt withCameraControls) Apply(opts *CameraOptions) {
	if opts.Controls == nil {
		opts.Controls = make(map[string]int32)
	}
	for k, v := range opt {
		opts.Controls[k] = v
	}
}

// WithCameraControls sets initial camera control values
func WithCameraControls(controls map[string]int32) CameraOption {
	return withCameraControls(controls)
}

type withCameraBufferCount int

func (opt withCameraBufferCount) Apply(opts *CameraOptions) {
	opts.BufferCount = int(opt)
}

// WithCameraBufferCount sets the number of camera capture buffers
func WithCameraBufferCount(count int) CameraOption {
	return withCameraBufferCount(count)
}
