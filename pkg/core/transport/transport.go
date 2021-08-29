package transport

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

var (
	transport = plugin.New()
)

type Transport interface {
	Dial(network, address string)
	Listen(network, address string) <-chan store.Store
	Send(store.Store)
	Close()
}

type TransportBuilder func(opts ...options.Option) (Transport, error)

func Register(name string, builder TransportBuilder) error {
	return transport.Register(name, func(opts ...options.Option) (plugin.Plugin, error) { return builder(opts...) })
}

func Unregister(name string) error {
	return transport.Unregister(name)
}

func New(name string) (Transport, error) {
	plg, err := transport.New(name)
	if err != nil {
		return nil, err
	}
	tr, ok := plg.(Transport)
	if !ok {
		return nil, plugin.ErrCorruptedRegistry
	}

	return tr, nil
}
