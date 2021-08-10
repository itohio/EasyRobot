package example

import "github.com/foxis/EasyRobot/pkg/plugin"

const NAME = "example"

type Options struct {
	b bool
}

func WithBool(b bool) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.b = b
		}
	}
}
