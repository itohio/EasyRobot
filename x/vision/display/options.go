package display

import (
	"github.com/itohio/EasyRobot/x/options"
	"github.com/itohio/EasyRobot/x/plugin"
	"github.com/itohio/EasyRobot/x/store"
)

const NAME = "show"

type Options struct {
	base plugin.Options
	keys []store.FQDNType
}

func WithKeyFirst(key store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = []store.FQDNType{key}
		}
	}
}

func WithKey(key store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = append(opt.keys, key)
		}
	}
}

func WithKeys(keys []store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = append(opt.keys, keys...)
		}
	}
}
