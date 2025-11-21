package gocv

import (
	"context"

	cv "gocv.io/x/gocv"

	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// codecConfig holds codec-related configuration.
type codecConfig struct {
	imageEncoding string
	tensorOpts    []tensorgocv.Option
}

func defaultCodecConfig() codecConfig {
	return codecConfig{
		imageEncoding: "png",
		tensorOpts:    nil,
	}
}

// streamConfig holds stream-related configuration.
type streamConfig struct {
	sources         []sourceSpec
	sequential      bool
	allowBestEffort bool
	sorter          fileSorter
}

func defaultStreamConfig() streamConfig {
	return streamConfig{
		sources:         nil,
		sequential:      false,
		allowBestEffort: true,
		sorter:          defaultSorter,
	}
}

// CloseCallback is called when a window is closed.
// window is the actual gocv.Window that was closed (for querying information).
// remainingWindows is the number of windows still open (after this one closes).
// Returns true if the application should terminate (cancel context).
type CloseCallback func(window *cv.Window, remainingWindows int) bool

// displayConfig holds display-related configuration.
type displayConfig struct {
	enabled           bool
	title             string
	width             int
	height            int
	onKey             func(types.KeyEvent) bool
	onMouse           func(types.MouseEvent) bool
	eventLoop         types.EventLoop
	cancelFunc        context.CancelFunc      // Cancel function to call if callback returns true
	onClose           CloseCallback           // Callback called when any window closes
}

func defaultDisplayConfig() displayConfig {
	return displayConfig{
		enabled: false,
		title:   "GoCV Display",
	}
}

// dnnConfig holds DNN-related configuration.
type dnnConfig struct {
	format  string
	backend cv.NetBackendType
	target  cv.NetTargetType
}

func defaultDNNConfig() dnnConfig {
	return dnnConfig{
		format:  "",
		backend: cv.NetBackendDefault,
		target:  cv.NetTargetCPU,
	}
}

