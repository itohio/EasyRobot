package example

import (
	"context"

	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

type stepImpl struct {
	Options
	in  <-chan pipeline.Data
	out chan pipeline.Data
}

func init() {
	pipeline.Register(NAME, New)
}

func New(opts ...plugin.Option) (pipeline.Step, error) {
	step := &stepImpl{
		Options: Options{},
	}
	plugin.ApplyOptions(&step.Options, opts...)
	plugin.ApplyOptions(&step.base, opts...)
	step.Reset()
	return step, nil
}

func (s *stepImpl) In(ch <-chan pipeline.Data) {
	s.in = ch
}

func (s *stepImpl) Out() <-chan pipeline.Data {
	return s.out
}

func (s *stepImpl) Run(ctx context.Context) {
	defer close(s.out)

	for {
		data, err := pipeline.StepReceive(ctx, s.base, s.in)
		if err != nil {
			return
		}

		out := store.NewWithName(NAME)
		out.CopyFrom(data)

		// TODO: process frame
		if err := pipeline.StepSend(ctx, s.base, s.out, out); err != nil {
			return
		}
	}
}

func (s *stepImpl) Reset() {
	s.out = pipeline.StepMakeChan(s.base)
}
