package steps

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/foxis/EasyRobot/internal/concurrency"
	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

const JOIN_NAME = "join"

type join struct {
	mutex sync.Mutex
	SyncOptions
	chArr []<-chan pipeline.Data
	inArr []buffer
	ch    chan pipeline.Data
	out   chan pipeline.Data
	index int
}

func init() {
	pipeline.Register(JOIN_NAME, NewJoin)
}

func NewJoin(opts ...options.Option) (pipeline.Step, error) {
	step := &join{
		SyncOptions: SyncOptions{
			tolerance:  time.Millisecond * 10,
			bufferSize: 3,
		},
		chArr: make([]<-chan pipeline.Data, 0),
	}

	options.ApplyOptions(&step.SyncOptions, opts...)
	options.ApplyOptions(&step.base, opts...)
	step.Reset()
	return step, nil
}

func (s *join) In(ch <-chan pipeline.Data) {
	s.chArr = append(s.chArr, ch)
	s.inArr = append(s.inArr, make(buffer, 0, s.bufferSize))
}

func (s *join) Out() <-chan pipeline.Data {
	return s.out
}

func (s *join) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")

	defer close(s.out)

	for idx, ch := range s.chArr {
		func(idx int, ch <-chan pipeline.Data) {
			concurrency.Submit(func() {
				for {
					data, err := pipeline.StepReceive(ctx, s.base, ch)
					if err != nil {
						if !errors.Is(err, pipeline.ErrEOS) {
							Log.Error().Err(err).Str("name", s.base.Name).Msg("Receive")
						}
						return
					}
					err = s.doJoin(ctx, idx, data)
					if err != nil && !errors.Is(err, pipeline.ErrDrop) {
						return
					}
				}
			})
		}(idx, ch)
	}

	<-ctx.Done()
}

func (s *join) Reset() {
	s.chArr = s.chArr[:0]
	s.inArr = s.inArr[:0]
	s.out = pipeline.StepMakeChan(s.SyncOptions.base)
	s.index = 0
}

func (s *join) doJoin(ctx context.Context, idx int, data store.Store) error {
	out := store.NewWithName(s.base.Name)
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.inArr[idx].Add(data)

	for _, buf := range s.inArr {
		if len(buf) == 0 {
			return nil
		}
	}

	for _, buf := range s.inArr {
		out.CopyFrom(buf.Pop())
	}

	out.SetIndex(int64(s.index))
	out.SetTimestamp(int64(time.Now().UnixNano()))
	s.index++

	return pipeline.StepSend(ctx, s.base, s.out, out)
}
