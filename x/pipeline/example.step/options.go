package example

import (
	"github.com/itohio/EasyRobot/x/options"
	"github.com/itohio/EasyRobot/x/plugin"
)

const NAME = "example"

type Options struct {
	base plugin.Options
	b    bool
}

func WithBool() options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.b = true
		}
	}
}
