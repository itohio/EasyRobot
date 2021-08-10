package format

import (
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

type Options struct {
	src store.FQDNType
	dst store.FQDNType
}

func WithKey(src, dst store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.src = src
			opt.dst = dst
		}
	}
}
