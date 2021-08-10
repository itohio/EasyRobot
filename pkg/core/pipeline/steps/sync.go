package steps

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/foxis/EasyRobot/internal/concurrency"
	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

const SYNC_NAME = "sync"

type buffer []store.Store

func (b buffer) Add(data store.Store) buffer {
	N := len(b)
	M := cap(b)
	if N == M {
		copy(b[:N-1], b[1:])
		b[N-1] = data
	} else {
		b = append(b, data)
	}
	return b
}

func (b *buffer) Pop() (val store.Store) {
	N := len(*b)
	if N > 0 {
		val = (*b)[N-1]
		*b = (*b)[:N-1]
	}
	return
}

func (b buffer) Prune(n int) buffer {
	N := len(b)
	M := N - n
	if M < 0 {
		M = 0
	} else {
		copy(b[:M], b[n:N])
	}

	return b[:M]
}

func (b buffer) FindClosest(timestamp int64) (int, int64) {
	var bestD int64 = int64(time.Hour)
	var bestI = -1

	abs := func(a int64) int64 {
		if a < 0 {
			return -a
		}
		return a
	}

	for i, d := range b {
		ts, err := d.Int64(store.TIMESTAMP)
		if err != nil {
			continue
		}
		d := abs(timestamp - ts)
		if d < bestD {
			bestD = d
			bestI = i
		}
	}

	return bestI, bestD
}

type SyncOptions struct {
	base       plugin.Options
	tolerance  time.Duration
	bufferSize int
}

type syncronize struct {
	mutex sync.Mutex
	SyncOptions
	chArr []<-chan pipeline.Data
	inArr []buffer
	ch    chan pipeline.Data
	out   chan pipeline.Data
	index int
}

func init() {
	pipeline.Register(SYNC_NAME, NewSync)
}

func NewSync(opts ...plugin.Option) (pipeline.Step, error) {
	step := &syncronize{
		SyncOptions: SyncOptions{
			tolerance:  time.Millisecond * 10,
			bufferSize: 3,
		},
		chArr: make([]<-chan pipeline.Data, 0),
	}

	plugin.ApplyOptions(&step.SyncOptions, opts...)
	plugin.ApplyOptions(&step.base, opts...)
	step.Reset()
	return step, nil
}

func (s *syncronize) In(ch <-chan pipeline.Data) {
	s.chArr = append(s.chArr, ch)
	s.inArr = append(s.inArr, make(buffer, 0, s.SyncOptions.bufferSize))
}

func (s *syncronize) Out() <-chan pipeline.Data {
	return s.out
}

func (s *syncronize) Run(ctx context.Context) {
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
					err = s.doSyncronize(ctx, idx, data)
					if err != nil && !errors.Is(err, pipeline.ErrDrop) {
						return
					}
				}
			})
		}(idx, ch)
	}

	<-ctx.Done()
}

func (s *syncronize) Reset() {
	s.chArr = s.chArr[:0]
	s.inArr = s.inArr[:0]
	s.out = pipeline.StepMakeChan(s.base)
	s.index = 0
}

func (s *syncronize) doSyncronize(ctx context.Context, idx int, data store.Store) error {
	out := store.NewWithName(s.base.Name)
	s.mutex.Lock()
	defer s.mutex.Unlock()

	timestamp, err := data.Int64(store.TIMESTAMP)
	if err != nil {
		return nil
	}

	s.inArr[idx].Add(data)

	bestI := make([]int, len(s.inArr))
	bestCount := 0
	for i, buf := range s.inArr {
		bi, bd := buf.FindClosest(timestamp)
		if bd <= int64(s.SyncOptions.tolerance) {
			bestI[i] = bi
			bestCount++
		}
	}

	if bestCount != len(s.inArr) {
		return nil
	}

	dataList := make([]pipeline.Data, len(s.inArr))
	for i, buf := range s.inArr {
		dataList[i] = buf[bestI[i]]
		s.inArr[i] = buf.Prune(bestI[i])
	}
	out.SetIndex(int64(s.index))
	out.SetTimestamp(timestamp)
	out.Set(store.SYNC_DATA, dataList)
	s.index++

	return pipeline.StepSend(ctx, s.base, s.out, out)
}
