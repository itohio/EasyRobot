package color

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

const NAME = "color"

type Options struct {
	src  store.FQDNType
	dst  store.FQDNType
	code int
}

func WithKey(src, dst store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.src = src
			opt.dst = dst
		}
	}
}

func WithConvertCode(code int) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.code = code
		}
	}
}
