package nats

import (
	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/store"
)

const NAME = "nats"

type Options struct {
	base     plugin.Options
	urls     string
	creds    string
	topicPub string
	topicSub string
	keys     []store.FQDNType
}

func WithUrls(urls string) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.urls = urls
		}
	}
}

func WithCredentials(creds string) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.creds = creds
		}
	}
}

func WithSubscribe(topic string) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.topicSub = topic
		}
	}
}

func WithPublish(topic string) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.topicSub = topic
		}
	}
}

func WithKeys(key ...store.FQDNType) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.keys = append(opt.keys, key...)
		}
	}
}
