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

type sourceKind int

const (
	sourceKindUnknown sourceKind = iota
	sourceKindSingle
	sourceKindVideoFile
	sourceKindVideoDevice
	sourceKindFileList
)

type deviceSpec struct {
	ID     int
	Width  int
	Height int
}

type sourceSpec struct {
	Kind   sourceKind
	Path   string
	Device *deviceSpec
	Files  []string
}

type fileSorter func([]string) []string

type config struct {
	ctx             context.Context
	imageEncoding   string
	tensorOpts      []tensorgocv.Option
	sources         []sourceSpec
	dnnFormat       string
	netBackend      cv.NetBackendType
	netTarget       cv.NetTargetType
	allowBestEffort bool
	sequential      bool
	sorter          fileSorter
	displayEnabled  bool
	displayTitle    string
	displayWidth    int
	displayHeight   int
	onKey           func(int) bool
	onMouse         func(int, int, int, int) bool
	eventLoop       func(context.Context, func() bool)
}

func defaultConfig() config {
	return config{
		ctx:             context.Background(),
		imageEncoding:   "png",
		tensorOpts:      nil,
		sources:         nil,
		dnnFormat:       "",
		netBackend:      cv.NetBackendDefault,
		netTarget:       cv.NetTargetCPU,
		allowBestEffort: true,
		sequential:      false,
		sorter:          defaultSorter,
		displayTitle:    "GoCV Display",
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
		cfg.sources = append(cfg.sources, sourceSpec{Kind: sourceKindUnknown, Path: path})
	})
}

// WithVideoDevice registers a video capture device.
func WithVideoDevice(id int, width, height int) types.Option {
	return newOption(func(cfg *config) {
		cfg.sources = append(cfg.sources, sourceSpec{
			Kind: sourceKindVideoDevice,
			Device: &deviceSpec{
				ID:     id,
				Width:  width,
				Height: height,
			},
		})
	})
}

// WithImageEncoding overrides the image encoding used when marshalling
// image.Image values. Supported values: "png" (default) and "jpeg".
func WithImageEncoding(format string) types.Option {
	format = strings.ToLower(strings.TrimSpace(format))
	return newOption(func(cfg *config) {
		switch format {
		case "", "png":
			cfg.imageEncoding = "png"
		case "jpg", "jpeg":
			cfg.imageEncoding = "jpeg"
		default:
			// unsupported formats fall back to default but we retain hint.
			cfg.imageEncoding = format
		}
	})
}

// WithTensorOptions configures options propagated to gocv tensor helpers.
func WithTensorOptions(opts ...tensorgocv.Option) types.Option {
	return newOption(func(cfg *config) {
		cfg.tensorOpts = append(cfg.tensorOpts, opts...)
	})
}

// WithNetBackend selects preferred backend for gocv.Net instances.
func WithNetBackend(backend cv.NetBackendType) types.Option {
	return newOption(func(cfg *config) {
		cfg.netBackend = backend
	})
}

// WithNetTarget selects preferred target for gocv.Net instances.
func WithNetTarget(target cv.NetTargetType) types.Option {
	return newOption(func(cfg *config) {
		cfg.netTarget = target
	})
}

// WithDNNFormat provides a hint about the DNN payload format (onnx, caffe, etc).
func WithDNNFormat(format string) types.Option {
	return newOption(func(cfg *config) {
		cfg.dnnFormat = strings.TrimSpace(strings.ToLower(format))
	})
}

// WithBestEffortDevices toggles best-effort synchronization for video devices.
func WithBestEffortDevices(enable bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.allowBestEffort = enable
	})
}

// WithSequential toggles sequential iteration over sources. When true the
// iterator consumes each source fully before moving to the next. By default
// sources are consumed in lockstep (parallel).
func WithSequential(enable bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.sequential = enable
	})
}

// WithDisplay enables display output using default window parameters.
func WithDisplay(ctx context.Context) types.Option {
	return newOption(func(cfg *config) {
		cfg.displayEnabled = true
		cfg.ctx = ctx
		if strings.TrimSpace(cfg.displayTitle) == "" {
			cfg.displayTitle = "GoCV Display"
		}
	})
}

// WithTitle sets the display window title and enables display output.
func WithTitle(title string) types.Option {
	return newOption(func(cfg *config) {
		cfg.displayEnabled = true
		cfg.displayTitle = strings.TrimSpace(title)
		if cfg.displayTitle == "" {
			cfg.displayTitle = "GoCV Display"
		}
	})
}

// WithWindowSize configures the display window size and enables display output.
func WithWindowSize(width, height int) types.Option {
	return newOption(func(cfg *config) {
		cfg.displayEnabled = true
		if width > 0 {
			cfg.displayWidth = width
		}
		if height > 0 {
			cfg.displayHeight = height
		}
	})
}

// WithOnKey installs a key handler invoked on each WaitKey event. Returning
// false stops the display/event loop.
func WithOnKey(handler func(int) bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.displayEnabled = true
		cfg.onKey = handler
	})
}

// WithOnMouse installs a mouse handler; returning false stops the display loop.
func WithOnMouse(handler func(event, x, y, flags int) bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.displayEnabled = true
		cfg.onMouse = handler
	})
}

// WithEventLoop overrides the default event loop used for display rendering.
func WithEventLoop(loop func(context.Context, func() bool)) types.Option {
	return newOption(func(cfg *config) {
		cfg.eventLoop = loop
	})
}

// WithSorter configures filename ordering for globbed image lists.
func WithSorter(sorter func([]string) []string) types.Option {
	return newOption(func(cfg *config) {
		if sorter == nil {
			cfg.sorter = defaultSorter
			return
		}
		cfg.sorter = sorter
	})
}

func applyOptions(base types.Options, cfg config, opts []types.Option) (types.Options, config) {
	local := base
	localCfg := cfg
	if len(cfg.sources) > 0 {
		localCfg.sources = append([]sourceSpec(nil), cfg.sources...)
	}
	if len(cfg.tensorOpts) > 0 {
		localCfg.tensorOpts = append([]tensorgocv.Option(nil), cfg.tensorOpts...)
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
	if localCfg.sorter == nil {
		localCfg.sorter = defaultSorter
	}
	if localCfg.displayEnabled && strings.TrimSpace(localCfg.displayTitle) == "" {
		localCfg.displayTitle = "GoCV Display"
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
	resolved := make([]sourceSpec, 0, len(cfg.sources))
	sorter := cfg.sorter
	if sorter == nil {
		sorter = defaultSorter
	}
	for _, spec := range cfg.sources {
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
	dirs := make([]string, 0, len(cfg.sources))
	for _, spec := range cfg.sources {
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
