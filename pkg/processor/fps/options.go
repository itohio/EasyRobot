package fps

import (
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

const NAME = "fps"

type Options struct {
	numFrames int
	suffix    store.FQDNType
}

func WithNumFrames(numFrames int) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.numFrames = numFrames
		}
	}
}

func WithSuffix(suffix store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.suffix = suffix
		}
	}
}
