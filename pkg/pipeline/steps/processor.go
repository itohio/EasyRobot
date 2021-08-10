package steps

import (
	"context"
	"errors"

	. "github.com/foxis/EasyRobot/pkg/logger"
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

const PROCESS_NAME = "process"

type ProcessFunc func(src, dst store.Store) error

type Processor interface {
	Init() error
	Close()
	Reset()
	Process(src, dst store.Store) error
}

type DefaultProcessor struct {
	process ProcessFunc
	init    func() error
	reset   func()
	close   func()
}

func (p DefaultProcessor) Init() error {
	if p.init == nil {
		return nil
	}
	return p.init()
}
func (p DefaultProcessor) Reset() {
	if p.reset == nil {
		return
	}
	p.reset()
}
func (p DefaultProcessor) Close() {
	if p.close == nil {
		return
	}
	p.close()
}
func (p DefaultProcessor) Process(src, dst pipeline.Data) error {
	if p.process == nil {
		return nil
	}
	return p.process(src, dst)
}

type ProcessorOptions struct {
	base      plugin.Options
	processor Processor
	fields    store.Store
}

type processor struct {
	ProcessorOptions
	ch  <-chan pipeline.Data
	out chan pipeline.Data
}

func WithFields(fields store.Store) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			opt.fields = fields
		}
	}
}
func WithProcessor(pr Processor) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			opt.processor = pr
			if named, ok := pr.(pipeline.NamedStep); ok {
				opt.base.Name = named.Name()
			}
		}
	}
}
func WithProcessorFunc(f ProcessFunc) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			if pr, ok := opt.processor.(*DefaultProcessor); ok {
				pr.process = f
			}
		}
	}
}
func WithNamedProcessorFunc(name string, f ProcessFunc) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			if pr, ok := opt.processor.(*DefaultProcessor); ok {
				pr.process = f
				opt.base.Name = name
			}
		}
	}
}
func WithInitFunc(f func() error) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			if pr, ok := opt.processor.(*DefaultProcessor); ok {
				pr.init = f
			}
		}
	}
}
func WithResetFunc(f func()) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			if pr, ok := opt.processor.(*DefaultProcessor); ok {
				pr.reset = f
			}
		}
	}
}
func WithCloseFunc(f func()) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*ProcessorOptions); ok {
			if pr, ok := opt.processor.(*DefaultProcessor); ok {
				pr.close = f
			}
		}
	}
}

func init() {
	pipeline.Register(PROCESS_NAME, NewProcessor)
}

func NewProcessor(opts ...plugin.Option) (pipeline.Step, error) {
	step := &processor{
		ProcessorOptions: ProcessorOptions{
			base:      plugin.DefaultOptions(),
			processor: &DefaultProcessor{},
		},
	}
	step.base.Name = PROCESS_NAME
	plugin.ApplyOptions(&step.ProcessorOptions, opts...)
	plugin.ApplyOptions(&step.ProcessorOptions.base, opts...)
	step.Reset()
	return step, nil
}

func (s *processor) In(ch <-chan pipeline.Data) {
	s.ch = ch
}

func (s *processor) Out() <-chan pipeline.Data {
	return s.out
}

func (s *processor) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")
	if err := s.processor.Init(); err != nil {
		Log.Error().Err(err).Str("name", s.base.Name).Msg("Init")
		return
	}
	defer close(s.out)
	defer s.processor.Close()

	for {
		data, err := pipeline.StepReceive(ctx, s.base, s.ch)
		if err != nil {
			if !errors.Is(err, pipeline.ErrEOS) {
				Log.Error().Err(err).Str("name", s.base.Name).Msg("Receive")
			}
			return
		}

		out := store.NewWithName(s.base.Name)
		if s.fields != nil {
			out.CopyFrom(s.fields)
		}
		out.CopyFrom(data)

		if s.base.Enabled {
			err = s.processor.Process(data, out)
			if err != nil {
				Log.Error().Err(err).Str("name", s.base.Name).Msg("Process")
				continue
			}
		}

		err = pipeline.StepSend(ctx, s.base, s.out, out)
		if err != nil && !errors.Is(err, pipeline.ErrDrop) {
			Log.Error().Err(err).Str("name", s.base.Name).Msg("Send")
			return
		}
		if errors.Is(err, pipeline.ErrDrop) {
			Log.Debug().Str("name", s.base.Name).Str("name", s.base.Name).Msg("drop")
		}
	}
}

func (s *processor) Reset() {
	s.out = pipeline.StepMakeChan(s.base)
	s.processor.Reset()
}
