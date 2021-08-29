package pipeline

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

var (
	pipeline = plugin.New()
)

type StepBuilder func(opts ...options.Option) (Step, error)

func Register(name string, builder StepBuilder) error {
	return pipeline.Register(name, func(opts ...options.Option) (plugin.Plugin, error) { return builder(opts...) })
}

func Unregister(name string) error {
	return pipeline.Unregister(name)
}

func NewStep(name string, opts ...options.Option) (Step, error) {
	plg, err := pipeline.New(name, opts...)
	if err != nil {
		return nil, err
	}
	step, ok := plg.(Step)
	if !ok {
		return nil, plugin.ErrCorruptedRegistry
	}

	return step, nil
}
