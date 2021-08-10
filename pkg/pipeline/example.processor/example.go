package example

import (
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
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
