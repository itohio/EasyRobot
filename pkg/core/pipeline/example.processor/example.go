package example

import (
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

const NAME = "example"

func init() {
	pipeline.Register(NAME, NewExample)
}

func NewExample(opts ...plugin.Option) (pipeline.Step, error) {
	newOpts := opts
	newOpts = append(newOpts, WithExample())
	return steps.NewProcessor(newOpts...)
}

func WithExample() plugin.Option {
	return steps.WithProcessor(&processorImpl{})
}

type processorImpl struct {
}

func (s *processorImpl) Process(src, dst store.Store) error {
	return nil
}

func (s *processorImpl) Init() error {
	return nil
}

func (s *processorImpl) Reset() {
}

func (s *processorImpl) Close() {
}

func (s *processorImpl) Name() string {
	return NAME
}
