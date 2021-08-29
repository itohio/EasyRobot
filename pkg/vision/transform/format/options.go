package format

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

type Options struct {
	src store.FQDNType
	dst store.FQDNType
}

func WithKey(src, dst store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.src = src
			opt.dst = dst
		}
	}
}
