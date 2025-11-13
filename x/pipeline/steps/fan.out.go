package steps

import (
	"context"
	"errors"

	. "github.com/itohio/EasyRobot/x/logger"
	"github.com/itohio/EasyRobot/x/options"
	"github.com/itohio/EasyRobot/x/pipeline"
	"github.com/itohio/EasyRobot/x/plugin"
	"github.com/itohio/EasyRobot/x/store"
)

const FANOUT_NAME = "fout"

type fanout struct {
	base  plugin.Options
	chArr []chan pipeline.Data
	in    <-chan pipeline.Data
}

func init() {
	pipeline.Register(FANOUT_NAME, NewFanOut)
}

func NewFanOut(opts ...options.Option) (pipeline.Step, error) {
	o := plugin.DefaultOptions()
	o.Name = FANOUT_NAME
	options.ApplyOptions(&o, opts...)
	return &fanout{
		base:  o,
		chArr: make([]chan pipeline.Data, 0),
	}, nil
}

func (s *fanout) In(ch <-chan pipeline.Data) {
	s.in = ch
}

func (s *fanout) Out() <-chan pipeline.Data {
	out := pipeline.StepMakeChan(s.base)
	s.chArr = append(s.chArr, out)

	return out
}

func (s *fanout) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")

	defer func() {
		for _, ch := range s.chArr {
			close(ch)
		}
	}()

	for {
		data, err := pipeline.StepReceive(ctx, s.base, s.in)
		if err != nil {
			if !errors.Is(err, pipeline.ErrEOS) {
				Log.Error().Err(err).Str("name", s.base.Name).Msg("Receive")
			}
			return
		}

		for i, ch := range s.chArr {
			err := pipeline.StepSend(ctx, s.base, ch, data.Clone(store.ANY))
			if err != nil && !errors.Is(err, pipeline.ErrDrop) {
				Log.Error().Err(err).Str("name", s.base.Name).Msg("Send")
				return
			}
			if errors.Is(err, pipeline.ErrDrop) {
				Log.Debug().Int("i", i).Str("name", s.base.Name).Msg("Drop")
			}
		}

		if s.base.Close {
			data.Close(store.ANY)
		}
	}
}

func (s *fanout) Reset() {
	s.chArr = make([]chan pipeline.Data, 0)
}
