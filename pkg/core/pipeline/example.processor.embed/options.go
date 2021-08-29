package example

import "github.com/foxis/EasyRobot/pkg/core/options"

const NAME = "example"

type Options struct {
	b bool
}

func WithBool(b bool) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.b = b
		}
	}
}
