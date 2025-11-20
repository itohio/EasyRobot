package gocv

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
)

// Re-export shared types for convenience
type CameraInfo = types.CameraInfo
type VideoFormat = types.VideoFormat
type ControlInfo = types.ControlInfo
type CameraController = types.CameraController

// Display types (re-exported from shared types)
type DisplayOptions = types.DisplayOptions
type DisplayOption = types.DisplayOption
type KeyEvent = types.KeyEvent
type MouseEvent = types.MouseEvent
type WindowEvent = types.WindowEvent
type EventLoop = types.EventLoop

type sourceKind int

const (
	sourceKindUnknown sourceKind = iota
	sourceKindSingle
	sourceKindVideoFile
	sourceKindVideoDevice
	sourceKindFileList
)

type deviceSpec struct {
	ID           int
	Width        int
	Height       int
	FrameRate    int
	PixelFormat  string
	Controls     map[string]int32
}

type sourceSpec struct {
	Kind   sourceKind
	Path   string
	Device *deviceSpec
	Files  []string
}

type fileSorter func([]string) []string

// Camera types are now defined in the shared types package

type config struct {
	ctx         context.Context
	codec       codecConfig
	stream      streamConfig
	display     displayConfig
	dnn         dnnConfig
	autoRelease bool
}

func defaultConfig() config {
	return config{
		ctx:         context.Background(),
		codec:       defaultCodecConfig(),
		stream:      defaultStreamConfig(),
		display:     defaultDisplayConfig(),
		dnn:         defaultDNNConfig(),
		autoRelease: false,
	}
}

func defaultSorter(list []string) []string {
	if len(list) == 0 {
		return list
	}
	sorted := append([]string(nil), list...)
	sort.Slice(sorted, func(i, j int) bool {
		return strings.ToLower(sorted[i]) < strings.ToLower(sorted[j])
	})
	return sorted
}

type configOption interface {
	types.Option
	applyConfig(*config)
}

type optionFunc struct {
	onTypes  func(*types.Options)
	onConfig func(*config)
}

func (o optionFunc) Apply(opts *types.Options) {
	if o.onTypes != nil {
		o.onTypes(opts)
	}
}

func (o optionFunc) applyConfig(cfg *config) {
	if o.onConfig != nil {
		o.onConfig(cfg)
	}
}

func newOption(onCfg func(*config)) configOption {
	return optionFunc{onConfig: onCfg}
}

// WithPath registers one or more filesystem paths that should be consumed by
// the GoCV unmarshaller when producing a streaming FrameStream. Paths may point
// to image files, custom .mat blobs, directories, or video files.
func WithPath(path string) types.Option {
	path = strings.TrimSpace(path)
	return newOption(func(cfg *config) {
		if path == "" {
			return
		}
		cfg.stream.sources = append(cfg.stream.sources, sourceSpec{Kind: sourceKindUnknown, Path: path})
	})
}

// WithVideoDevice registers a video capture device with basic configuration.
func WithVideoDevice(id int, width, height int) types.Option {
	return newOption(func(cfg *config) {
		cfg.stream.sources = append(cfg.stream.sources, sourceSpec{
			Kind: sourceKindVideoDevice,
			Device: &deviceSpec{
				ID:     id,
				Width:  width,
				Height: height,
			},
		})
	})
}

// WithVideoDeviceEx registers a video capture device with extended configuration.
func WithVideoDeviceEx(id int, width, height, fps int, pixelFormat string) types.Option {
	return newOption(func(cfg *config) {
		cfg.stream.sources = append(cfg.stream.sources, sourceSpec{
			Kind: sourceKindVideoDevice,
			Device: &deviceSpec{
				ID:          id,
				Width:       width,
				Height:      height,
				FrameRate:   fps,
				PixelFormat: pixelFormat,
			},
		})
	})
}

// WithFrameRate sets the desired frame rate for video devices.
func WithFrameRate(fps int) types.Option {
	return newOption(func(cfg *config) {
		// Apply to all video device sources
		for i := range cfg.stream.sources {
			if cfg.stream.sources[i].Kind == sourceKindVideoDevice && cfg.stream.sources[i].Device != nil {
				cfg.stream.sources[i].Device.FrameRate = fps
			}
		}
	})
}

// WithPixelFormat sets the desired pixel format for video devices.
func WithPixelFormat(format string) types.Option {
	return newOption(func(cfg *config) {
		// Apply to all video device sources
		for i := range cfg.stream.sources {
			if cfg.stream.sources[i].Kind == sourceKindVideoDevice && cfg.stream.sources[i].Device != nil {
				cfg.stream.sources[i].Device.PixelFormat = format
			}
		}
	})
}

// WithCameraControls sets initial camera control values.
func WithCameraControls(controls map[string]int32) types.Option {
	return newOption(func(cfg *config) {
		// Apply to all video device sources
		for i := range cfg.stream.sources {
			if cfg.stream.sources[i].Kind == sourceKindVideoDevice && cfg.stream.sources[i].Device != nil {
				if cfg.stream.sources[i].Device.Controls == nil {
					cfg.stream.sources[i].Device.Controls = make(map[string]int32)
				}
				for k, v := range controls {
					cfg.stream.sources[i].Device.Controls[k] = v
				}
			}
		}
	})
}

// WithImageEncoding overrides the image encoding used when marshalling
// image.Image values. Supported values: "png" (default) and "jpeg".
func WithImageEncoding(format string) types.Option {
	format = strings.ToLower(strings.TrimSpace(format))
	return newOption(func(cfg *config) {
		switch format {
		case "", "png":
			cfg.codec.imageEncoding = "png"
		case "jpg", "jpeg":
			cfg.codec.imageEncoding = "jpeg"
		default:
			// unsupported formats fall back to default but we retain hint.
			cfg.codec.imageEncoding = format
		}
	})
}

// WithTensorOptions configures options propagated to gocv tensor helpers.
func WithTensorOptions(opts ...tensorgocv.Option) types.Option {
	return newOption(func(cfg *config) {
		cfg.codec.tensorOpts = append(cfg.codec.tensorOpts, opts...)
	})
}

// WithNetBackend selects preferred backend for gocv.Net instances.
func WithNetBackend(backend cv.NetBackendType) types.Option {
	return newOption(func(cfg *config) {
		cfg.dnn.backend = backend
	})
}

// WithNetTarget selects preferred target for gocv.Net instances.
func WithNetTarget(target cv.NetTargetType) types.Option {
	return newOption(func(cfg *config) {
		cfg.dnn.target = target
	})
}

// WithDNNFormat provides a hint about the DNN payload format (onnx, caffe, etc).
func WithDNNFormat(format string) types.Option {
	return newOption(func(cfg *config) {
		cfg.dnn.format = strings.TrimSpace(strings.ToLower(format))
	})
}

// WithBestEffortDevices toggles best-effort synchronization for video devices.
func WithBestEffortDevices(enable bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.stream.allowBestEffort = enable
	})
}

// WithSequential toggles sequential iteration over sources. When true the
// iterator consumes each source fully before moving to the next. By default
// sources are consumed in lockstep (parallel).
func WithSequential(enable bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.stream.sequential = enable
	})
}

// WithDisplay enables display output using default window parameters.
func WithDisplay(ctx context.Context) types.Option {
	return newOption(func(cfg *config) {
		cfg.display.enabled = true
		cfg.ctx = ctx
		if strings.TrimSpace(cfg.display.title) == "" {
			cfg.display.title = "GoCV Display"
		}
	})
}

// WithTitle sets the display window title and enables display output.
func WithTitle(title string) types.Option {
	return newOption(func(cfg *config) {
		title = strings.TrimSpace(title)
		cfg.display.enabled = true
		cfg.display.title = title
		if cfg.display.title == "" {
			cfg.display.title = "GoCV Display"
		}
	})
}

// WithWindowSize configures the display window size and enables display output.
func WithWindowSize(width, height int) types.Option {
	return newOption(func(cfg *config) {
		cfg.display.enabled = true
		if width > 0 {
			cfg.display.width = width
		}
		if height > 0 {
			cfg.display.height = height
		}
	})
}

// WithOnKey installs a key handler invoked on each WaitKey event. Returning
// false stops the display/event loop.
func WithOnKey(handler func(types.KeyEvent) bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.display.enabled = true
		cfg.display.onKey = handler
	})
}

// WithOnMouse installs a mouse handler; returning false stops the display loop.
func WithOnMouse(handler func(types.MouseEvent) bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.display.enabled = true
		cfg.display.onMouse = handler
	})
}

// WithEventLoop overrides the default event loop used for display rendering.
func WithEventLoop(loop types.EventLoop) types.Option {
	return newOption(func(cfg *config) {
		cfg.display.eventLoop = loop
	})
}

// WithSorter configures filename ordering for globbed image lists.
func WithSorter(sorter func([]string) []string) types.Option {
	return newOption(func(cfg *config) {
		if sorter == nil {
			cfg.stream.sorter = defaultSorter
			return
		}
		cfg.stream.sorter = sorter
	})
}

func applyOptions(base types.Options, cfg config, opts []types.Option) (types.Options, config) {
	local := base
	localCfg := cfg
	// Copy focused configs
	if len(cfg.stream.sources) > 0 {
		localCfg.stream.sources = append([]sourceSpec(nil), cfg.stream.sources...)
	}
	if len(cfg.codec.tensorOpts) > 0 {
		localCfg.codec.tensorOpts = append([]tensorgocv.Option(nil), cfg.codec.tensorOpts...)
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt.Apply(&local)
		if cfgOpt, ok := opt.(configOption); ok {
			cfgOpt.applyConfig(&localCfg)
		}
	}
	if local.Context != nil {
		localCfg.ctx = local.Context
	}
	localCfg.autoRelease = local.ReleaseAfterProcessing
	// Apply defaults
	if localCfg.stream.sorter == nil {
		localCfg.stream.sorter = defaultSorter
	}
	if localCfg.display.enabled && strings.TrimSpace(localCfg.display.title) == "" {
		localCfg.display.title = "GoCV Display"
	}
	return local, localCfg
}

func classifyPath(path string) sourceKind {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff":
		return sourceKindSingle
	case ".mat":
		return sourceKindSingle
	case ".avi", ".mp4", ".mov", ".mkv", ".wmv":
		return sourceKindVideoFile
	default:
		return sourceKindUnknown
	}
}

func resolveSources(cfg config) ([]sourceSpec, error) {
	sources := cfg.stream.sources
	resolved := make([]sourceSpec, 0, len(sources))
	sorter := cfg.stream.sorter
	if sorter == nil {
		sorter = defaultSorter
	}
	for _, spec := range sources {
		if spec.Kind == sourceKindVideoDevice {
			if spec.Device == nil {
				return nil, fmt.Errorf("gocv: video device option missing configuration")
			}
			resolved = append(resolved, spec)
			continue
		}
		if strings.TrimSpace(spec.Path) == "" {
			continue
		}

		kind := spec.Kind
		if kind == sourceKindUnknown {
			if isGlobPattern(spec.Path) {
				matches, err := filepath.Glob(spec.Path)
				if err != nil {
					return nil, fmt.Errorf("gocv: glob %s: %w", spec.Path, err)
				}
				if len(matches) == 0 {
					return nil, fmt.Errorf("gocv: glob %s did not match any files", spec.Path)
				}
				if sorter != nil {
					matches = sorter(matches)
				}
				resolved = append(resolved, sourceSpec{
					Kind:  sourceKindFileList,
					Path:  spec.Path,
					Files: matches,
				})
				continue
			}

			kind = classifyPath(spec.Path)
			if kind == sourceKindUnknown {
				kind = sourceKindSingle
			}
		}
		resolved = append(resolved, sourceSpec{
			Kind: kind,
			Path: spec.Path,
			Files: func() []string {
				if len(spec.Files) == 0 {
					return nil
				}
				cp := append([]string(nil), spec.Files...)
				return cp
			}(),
		})
	}
	return resolved, nil
}

func isGlobPattern(path string) bool {
	return strings.ContainsAny(path, "*?[")
}

func resolveOutputDirs(cfg config) ([]string, error) {
	sources := cfg.stream.sources
	dirs := make([]string, 0, len(sources))
	for _, spec := range sources {
		if spec.Kind != sourceKindUnknown {
			continue
		}
		path := strings.TrimSpace(spec.Path)
		if path == "" {
			continue
		}
		if isGlobPattern(path) {
			continue
		}
		if err := os.MkdirAll(path, 0o755); err != nil {
			return nil, fmt.Errorf("gocv: ensure output dir %s: %w", path, err)
		}
		dirs = append(dirs, path)
	}
	return dirs, nil
}
