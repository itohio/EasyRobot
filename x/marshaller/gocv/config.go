package gocv

import (
	"context"

	cv "gocv.io/x/gocv"

	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
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

// displayConfig holds display-related configuration.
type displayConfig struct {
	enabled   bool
	title     string
	width     int
	height    int
	onKey     func(int) bool
	onMouse   func(int, int, int, int) bool
	eventLoop func(context.Context, func() bool)
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

