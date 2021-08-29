package example

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
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
