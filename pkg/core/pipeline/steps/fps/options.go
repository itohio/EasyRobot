package fps

import (
	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/store"
)

const NAME = "fps"

type Options struct {
	numFrames int
	suffix    store.FQDNType
}

func WithNumFrames(numFrames int) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.numFrames = numFrames
		}
	}
}

func WithSuffix(suffix store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.suffix = suffix
		}
	}
}
