package example

import (
	"github.com/foxis/EasyRobot/pkg/plugin"
)

const NAME = "example"

type Options struct {
	base plugin.Options
	b    bool
}

func WithBool() plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.b = true
		}
	}
}
