package example

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

type stepImpl struct {
	options Options
	step    pipeline.Step
}

func init() {
	pipeline.Register(NAME, New)
}

func New(opts ...options.Option) (pipeline.Step, error) {
	algorithm := &stepImpl{
		options: Options{},
	}

	options.ApplyOptions(&algorithm.options, opts...)
	newOpts := append([]options.Option{plugin.WithName(NAME)}, opts...)
	newOpts = append(
		newOpts,
		steps.WithProcessor(algorithm),
	)

	step, err := steps.NewProcessor(newOpts...)
	if err != nil {
		return nil, err
	}
	algorithm.step = step

	return step, nil
}

func (s *stepImpl) Init() error {
	return nil
}

func (s *stepImpl) Reset() {
	return
}

func (s *stepImpl) Close() {
	return
}

func (s *stepImpl) Process(src, dst pipeline.Data) error {
	return nil
}
