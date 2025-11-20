// go:build linux
package v4l

import (
	"context"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

// DeviceOption configures device opening (legacy - use types.CameraOption)
type Option interface {
	Apply(*Options)
}

// Ensure DeviceOption implements types.CameraOption for compatibility
func (opt withBufferCount) ApplyCamera(opts *types.CameraOptions) {
	// Camera options don't directly map to our internal options
	// This would need to be implemented if we want to support shared camera options
}

func (opt withPixelFormat) ApplyCamera(opts *types.CameraOptions) {
	// Implementation would go here
}

func (opt withResolution) ApplyCamera(opts *types.CameraOptions) {
	opts.Width = opt.width
	opts.Height = opt.height
}

func (opt withFrameRate) ApplyCamera(opts *types.CameraOptions) {
	// Frame rate would need conversion
}

func (opt withControls) ApplyCamera(opts *types.CameraOptions) {
	opts.Controls = make(map[string]int32)
	for k, v := range opt {
		opts.Controls[k] = v
	}
}

// Options holds device configuration
type Options struct {
	// DevicePaths specifies device paths to open (e.g., ["/dev/video0", "/dev/video1"])
	DevicePaths []string

	// BufferCount is number of capture buffers (default: 4)
	BufferCount int

	// PixelFormat specifies desired pixel format (as string)
	PixelFormat string

	// Width/Height specify desired resolution
	Width, Height int

	// FrameRate specifies desired frame rate
	FrameRate Fraction

	// Controls specifies initial control values (by name)
	Controls map[string]int32

	// BufferPool provides custom buffer pool (default: internal pool)
	BufferPool *helpers.Pool[uint8]

	// TensorFactory creates tensors from buffer data
	TensorFactory func([]uint8, int, int, int) types.Tensor

	// Context for cancellation
	Context context.Context

	// AllowBestEffort enables best-effort synchronization between devices
	AllowBestEffort bool

	// Sequential disables synchronization and processes devices sequentially
	Sequential bool
}

// DeviceOption implementations

type withBufferCount int

func (opt withBufferCount) Apply(opts *Options) {
	opts.BufferCount = int(opt)
}

// WithBufferCount sets the number of capture buffers
func WithBufferCount(count int) Option {
	return withBufferCount(count)
}

type withPixelFormat PixelFormat

func (opt withPixelFormat) Apply(opts *Options) {
	opts.PixelFormat = PixelFormat(opt)
}

// WithPixelFormat sets the desired pixel format
func WithPixelFormat(format PixelFormat) Option {
	return withPixelFormat(format)
}

type withResolution struct {
	width, height int
}

func (opt withResolution) Apply(opts *Options) {
	opts.Width = opt.width
	opts.Height = opt.height
}

// WithResolution sets the desired resolution
func WithResolution(width, height int) Option {
	return withResolution{width: width, height: height}
}

type withFrameRate Fraction

func (opt withFrameRate) Apply(opts *Options) {
	opts.FrameRate = Fraction(opt)
}

// WithFrameRate sets the desired frame rate
func WithFrameRate(fps Fraction) Option {
	return withFrameRate(fps)
}

type withControls map[ControlID]int32

func (opt withControls) Apply(opts *Options) {
	if opts.Controls == nil {
		opts.Controls = make(map[ControlID]int32)
	}
	for k, v := range opt {
		opts.Controls[k] = v
	}
}

// WithControls sets initial control values
func WithControls(controls map[ControlID]int32) Option {
	return withControls(controls)
}

type withBufferPool struct {
	pool *helpers.Pool[uint8]
}

func (opt withBufferPool) Apply(opts *Options) {
	opts.BufferPool = opt.pool
}

// WithBufferPool sets a custom buffer pool
func WithBufferPool(pool *helpers.Pool[uint8]) Option {
	return withBufferPool{pool: pool}
}

type withTensorFactory struct {
	factory func([]uint8, int, int, int) types.Tensor
}

func (opt withTensorFactory) Apply(opts *Options) {
	opts.TensorFactory = opt.factory
}

// WithTensorFactory sets the tensor constructor
func WithTensorFactory(factory func([]uint8, int, int, int) types.Tensor) Option {
	return withTensorFactory{factory: factory}
}

type withContext struct {
	ctx context.Context
}

func (opt withContext) Apply(opts *Options) {
	opts.Context = opt.ctx
}

// WithContext sets the context for cancellation
func WithContext(ctx context.Context) Option {
	return withContext{ctx: ctx}
}

type withBestEffort bool

func (opt withBestEffort) Apply(opts *Options) {
	opts.AllowBestEffort = bool(opt)
}

// WithBestEffort enables best-effort synchronization
func WithBestEffort(enable bool) Option {
	return withBestEffort(enable)
}

type withSequential bool

func (opt withSequential) Apply(opts *Options) {
	opts.Sequential = bool(opt)
}

// WithSequential disables synchronization and processes sequentially
func WithSequential(enable bool) Option {
	return withSequential(enable)
}

// Device path configuration options

type withDevicePaths []string

func (opt withDevicePaths) Apply(opts *Options) {
	opts.DevicePaths = opt
}

// WithDevicePaths sets the device paths to open
func WithDevicePaths(paths ...string) Option {
	return withDevicePaths(paths)
}

type withDevicePath string

func (opt withDevicePath) Apply(opts *Options) {
	opts.DevicePaths = append(opts.DevicePaths, string(opt))
}

// WithDevicePath adds a device path to open
func WithDevicePath(path string) Option {
	return withDevicePath(path)
}

// WithVideoDevice registers a video capture device by ID
func WithVideoDevice(id int, width, height int) Option {
	return withVideoDevice{
		id:     id,
		width:  width,
		height: height,
	}
}

// WithVideoDeviceEx registers a video capture device with extended configuration
func WithVideoDeviceEx(id int, width, height, fps int, pixelFormat string) types.Option {
	return withVideoDeviceEx{
		id:          id,
		width:       width,
		height:      height,
		fps:         fps,
		pixelFormat: pixelFormat,
	}
}

// Internal option types

type withVideoDevice struct {
	id, width, height int
}

func (opt withVideoDevice) Apply(opts *Options) {
	devicePath := fmt.Sprintf("/dev/video%d", opt.id)
	opts.DevicePaths = append(opts.DevicePaths, devicePath)
	opts.Width = opt.width
	opts.Height = opt.height
}

type withVideoDeviceEx struct {
	id                 int
	width, height, fps int
	pixelFormat        string
}

func (opt withVideoDeviceEx) Apply(opts *types.Options) {
	// This applies to the shared Options, we'll handle device registration in unmarshaller
}

func (opt withVideoDeviceEx) ApplyV4L(opts *Options) {
	// Custom apply method for V4L-specific options
	devicePath := fmt.Sprintf("/dev/video%d", opt.id)
	opts.DevicePaths = append(opts.DevicePaths, devicePath)
	opts.Width = opt.width
	opts.Height = opt.height
	opts.FrameRate = Fraction{Numerator: opt.fps, Denominator: 1}
	opts.PixelFormat = opt.pixelFormat
}

// WithCameraControls sets camera control values (V4L version)
func WithCameraControls(controls map[string]int32) Option {
	return withControls(controls)
}

// applyOptions applies options to create final configuration
func applyOptions(baseOpts types.Options, baseCfg Options, opts ...types.Option) (types.Options, Options) {
	// Apply marshaller options
	finalOpts := baseOpts
	finalCfg := baseCfg

	// First pass: apply all options
	for _, opt := range opts {
		if opt != nil {
			opt.Apply(&finalOpts)
		}

		// Check for V4L-specific options
		if v4lOpt, ok := opt.(Option); ok {
			v4lOpt.Apply(&finalCfg)
		}

		// Check for extended video device options
		if extOpt, ok := opt.(withVideoDeviceEx); ok {
			extOpt.ApplyV4L(&finalCfg)
		}
	}

	// Set defaults
	if finalCfg.BufferCount == 0 {
		finalCfg.BufferCount = 4
	}
	if finalCfg.Width == 0 {
		finalCfg.Width = 640
	}
	if finalCfg.Height == 0 {
		finalCfg.Height = 480
	}
	if finalCfg.FrameRate.Numerator == 0 {
		finalCfg.FrameRate = Fraction{Numerator: 30, Denominator: 1}
	}
	if finalCfg.PixelFormat == 0 {
		finalCfg.PixelFormat = PixelFmtMJPEG
	}
	if finalCfg.BufferPool == nil {
		finalCfg.BufferPool = createDefaultBufferPool()
	}
	if finalCfg.TensorFactory == nil {
		finalCfg.TensorFactory = defaultTensorFactory(finalCfg.BufferPool)
	}
	if finalCfg.Context == nil {
		finalCfg.Context = context.Background()
	}

	return finalOpts, finalCfg
}

// createDefaultBufferPool creates a default buffer pool with reasonable tiers
func createDefaultBufferPool() *helpers.Pool[uint8] {
	pool := &helpers.Pool[uint8]{}
	// Tiers for common frame sizes: 64KB, 256KB, 1MB, 4MB, 16MB
	tiers := []int{65536, 262144, 1048576, 4194304, 16777216}
	if err := pool.Reconfigure(tiers...); err != nil {
		// Fallback to default pool configuration
		pool.Reconfigure()
	}
	return pool
}
