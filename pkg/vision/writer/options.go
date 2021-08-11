package writer

import (
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

type writerOpts struct {
	base   plugin.Options
	keys   []store.FQDNType
	prefix string
	ext    string
}

func WithKeyFirst(key store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*writerOpts); ok {
			opt.keys = []store.FQDNType{key}
		}
	}
}

func WithKey(key store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*writerOpts); ok {
			opt.keys = append(opt.keys, key)
		}
	}
}

func WithKeys(keys []store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*writerOpts); ok {
			opt.keys = append(opt.keys, keys...)
		}
	}
}

func WithPrefix(prefix string) plugin.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*writerOpts); ok {
			o.prefix = prefix
		}
	}
}
func WithExtension(ext string) plugin.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*writerOpts); ok {
			o.ext = ext
		}
	}
}
