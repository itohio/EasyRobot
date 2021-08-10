package example

import (
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/plugin"
)

type stepImpl struct {
	options Options
	step    pipeline.Step
}

func init() {
	pipeline.Register(NAME, New)
}

func New(opts ...plugin.Option) (pipeline.Step, error) {
	algorithm := &stepImpl{
		options: Options{},
	}

	plugin.ApplyOptions(&algorithm.options, opts...)
	newOpts := append([]plugin.Option{plugin.WithName(NAME)}, opts...)
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
