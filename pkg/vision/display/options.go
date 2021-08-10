package display

import (
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

const NAME = "show"

type Options struct {
	base plugin.Options
	keys []store.FQDNType
}

func WithKeyFirst(key store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = []store.FQDNType{key}
		}
	}
}

func WithKey(key store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = append(opt.keys, key)
		}
	}
}

func WithKeys(keys []store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = append(opt.keys, keys...)
		}
	}
}
