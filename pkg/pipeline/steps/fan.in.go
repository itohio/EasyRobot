package steps

import (
	"context"
	"errors"

	"github.com/foxis/EasyRobot/internal/concurrency"
	. "github.com/foxis/EasyRobot/pkg/logger"
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/plugin"
)

const FANIN_NAME = "fin"

type fanin struct {
	base  plugin.Options
	chArr []<-chan pipeline.Data
	ch    chan pipeline.Data
	out   chan pipeline.Data
}

func init() {
	pipeline.Register(FANIN_NAME, NewFanIn)
}

func NewFanIn(opts ...plugin.Option) (pipeline.Step, error) {
	step := &fanin{base: plugin.DefaultOptions()}
	step.base.Name = FANIN_NAME
	plugin.ApplyOptions(&step.base, opts...)
	step.Reset()
	return step, nil
}

func (s *fanin) In(ch <-chan pipeline.Data) {
	s.chArr = append(s.chArr, ch)
}

func (s *fanin) Out() <-chan pipeline.Data {
	return s.out
}

func (s *fanin) Run(ctx context.Context) {
	defer close(s.out)

	for _, ch := range s.chArr {
		func(ch <-chan pipeline.Data) {
			concurrency.Submit(func() {
				for {
					data, err := pipeline.StepReceive(ctx, s.base, ch)
					if err != nil {
						if !errors.Is(err, pipeline.ErrEOS) {
							Log.Error().Err(err).Str("name", s.base.Name).Msg("Receive")
						}
						return
					}
					err = pipeline.StepSend(ctx, s.base, s.out, data)
					if err != nil && !errors.Is(err, pipeline.ErrDrop) {
						return
					}
				}
			})
		}(ch)
	}

	<-ctx.Done()
}

func (s *fanin) Reset() {
	s.chArr = make([]<-chan pipeline.Data, 0)
	s.out = pipeline.StepMakeChan(s.base)
}
