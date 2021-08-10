package pipeline

import (
	"github.com/foxis/EasyRobot/pkg/plugin"
)

var (
	pipeline = plugin.New()
)

type StepBuilder func(opts ...plugin.Option) (Step, error)

func Register(name string, builder StepBuilder) error {
	return pipeline.Register(name, func(opts ...plugin.Option) (plugin.Plugin, error) { return builder(opts...) })
}

func Unregister(name string) error {
	return pipeline.Unregister(name)
}

func NewStep(name string, opts ...plugin.Option) (Step, error) {
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
