package dnn

import (
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"gocv.io/x/gocv"
)

const NAME = "example"

type Options struct {
	model  string
	config string

	target  gocv.NetTargetType
	backend gocv.NetBackendType

	width  int
	height int

	ratio   float64
	mean    [4]float64
	swapRGB bool
}

func WithImageParams(ratio float64, mean [4]float64, swapRGB bool) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.ratio = ratio
			opt.mean = mean
			opt.swapRGB = swapRGB
		}
	}
}

func WithResolution(width, height int) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.width = width
			opt.height = height
		}
	}
}

func WithModel(model, config string) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.model = model
			opt.config = config
		}
	}
}

func WithTarget(target string) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.target = gocv.ParseNetTarget(target)
		}
	}
}

func WithBackend(backend string) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.backend = gocv.ParseNetBackend(backend)
		}
	}
}
