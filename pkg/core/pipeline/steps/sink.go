package steps

import (
	"context"
	"errors"

	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

const SINK_NAME = "sink"

type SinkFunc func(data store.Store) error

type Sink interface {
	Init() error
	Close()
	Reset()
	Sink(data store.Store) error
}

type DefaultSink struct {
	sink  SinkFunc
	init  func() error
	reset func()
	close func()
}

func (p DefaultSink) Init() error {
	if p.init == nil {
		return nil
	}
	return p.init()
}
func (p DefaultSink) Reset() {
	if p.reset == nil {
		return
	}
	p.reset()
}
func (p DefaultSink) Close() {
	if p.close == nil {
		return
	}
	p.close()
}
func (p DefaultSink) Sink(data pipeline.Data) error {
	if p.sink == nil {
		return nil
	}
	return p.sink(data)
}

type SinkOptions struct {
	base plugin.Options
	sink Sink
}

func WithSinkProcessor(pr Sink) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SinkOptions); ok {
			opt.sink = pr
			if named, ok := pr.(pipeline.NamedStep); ok {
				opt.base.Name = named.Name()
			}
		}
	}
}
func WithSinkFunc(f SinkFunc) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SinkOptions); ok {
			if pr, ok := opt.sink.(*DefaultSink); ok {
				pr.sink = f
			}
		}
	}
}
func WithNamedSinkFunc(name string, f SinkFunc) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SinkOptions); ok {
			if pr, ok := opt.sink.(*DefaultSink); ok {
				pr.sink = f
				opt.base.Name = name
			}
		}
	}
}
func WithSinkInitFunc(f func() error) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SinkOptions); ok {
			if pr, ok := opt.sink.(*DefaultSink); ok {
				pr.init = f
			}
		}
	}
}
func WithSinkResetFunc(f func()) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SinkOptions); ok {
			if pr, ok := opt.sink.(*DefaultSink); ok {
				pr.reset = f
			}
		}
	}
}
func WithSinkCloseFunc(f func()) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SinkOptions); ok {
			if pr, ok := opt.sink.(*DefaultSink); ok {
				pr.close = f
			}
		}
	}
}

type sink struct {
	SinkOptions
	ch <-chan pipeline.Data
}

func init() {
	pipeline.Register(SINK_NAME, NewSink)
}

func NewSink(opts ...plugin.Option) (pipeline.Step, error) {
	step := &sink{
		SinkOptions: SinkOptions{
			base: plugin.DefaultOptions(),
			sink: &DefaultSink{},
		},
	}
	step.base.Name = SINK_NAME
	plugin.ApplyOptions(&step.SinkOptions, opts...)
	plugin.ApplyOptions(&step.SinkOptions.base, opts...)
	step.Reset()
	return step, nil
}

func (s *sink) In(ch <-chan pipeline.Data) {
	s.ch = ch
}

func (s *sink) Out() <-chan pipeline.Data {
	return nil
}

func (s *sink) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")

	if err := s.sink.Init(); err != nil {
		return
	}
	defer s.sink.Close()

	for {
		data, err := pipeline.StepReceive(ctx, s.base, s.ch)
		if err != nil {
			if !errors.Is(err, pipeline.ErrEOS) {
				Log.Error().Err(err).Str("name", s.base.Name).Msg("Receive")
			}
			return
		}

		if !(s.base.Close && s.base.Enabled) {
			continue
		}

		if err := s.sink.Sink(data); err != nil {
			Log.Error().Err(err).Str("name", s.base.Name).Msg("Sink")
		}
	}
}

func (s *sink) Reset() {
	s.sink.Reset()
}
