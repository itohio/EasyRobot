package steps

import (
	"context"
	"errors"

	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

const SOURCE_NAME = "source"

type SourceOptions struct {
	base   plugin.Options
	Repeat bool
	reader SourceReader
	dst    store.FQDNType
}

type SourceReader interface {
	Read(o SourceOptions) (img interface{}, path string, index int, timestamp int64, err error)
	Reset()
	Open() error
	Close()
}

type NamedSourceReader interface {
	Name() string
}

func WithSourceReader(reader SourceReader) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SourceOptions); ok {
			opt.reader = reader
			if named, ok := reader.(NamedSourceReader); ok {
				opt.base.Name = named.Name()
			}
		}
	}
}

func WithRepeat() plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SourceOptions); ok {
			opt.Repeat = true
		}
	}
}

func WithKey(dst store.FQDNType) plugin.Option {
	return func(o interface{}) {
		if opt, ok := o.(*SourceOptions); ok {
			opt.dst = dst
		}
	}
}

type readerImpl struct {
	SourceOptions
	out chan pipeline.Data
}

func init() {
	pipeline.Register(SOURCE_NAME, NewReader)
}

func NewReader(opts ...plugin.Option) (pipeline.Step, error) {
	step := &readerImpl{
		SourceOptions: SourceOptions{
			base: plugin.DefaultOptions(),
			dst:  store.IMAGE,
		},
	}
	plugin.ApplyOptions(&step.SourceOptions, opts...)
	plugin.ApplyOptions(&step.SourceOptions.base, opts...)
	step.Reset()
	return step, nil
}

func (s *readerImpl) In(ch <-chan pipeline.Data) {
}

func (s *readerImpl) Out() <-chan pipeline.Data {
	return s.out
}

func (s *readerImpl) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")
	defer close(s.out)

	if s.reader == nil {
		Log.Error().Str("name", s.base.Name).Msg("Reader is nil")
		return
	}
	if err := s.reader.Open(); err != nil {
		Log.Error().Err(err).Str("name", s.base.Name).Msg("Reader failed")
		return
	}
	defer s.reader.Close()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			img, path, index, timestamp, err := s.reader.Read(s.SourceOptions)
			if err != nil && !s.base.IgnoreErrors {
				if s.base.NoEOS {
					continue
				}
				return
			}
			if img == nil {
				continue
			}

			data := store.NewWithName(s.base.Name)
			data.SetIndex(int64(index))
			data.SetTimestamp(timestamp)
			data.Set(s.dst, img)

			if path != "" {
				data.Set(store.PATH, path)
			}

			err = pipeline.StepSend(ctx, s.SourceOptions.base, s.out, data)
			if err != nil && !errors.Is(err, pipeline.ErrDrop) {
				return
			}
		}
	}
}

func (s *readerImpl) Reset() {
	s.out = pipeline.StepMakeChan(s.SourceOptions.base)
	if s.reader == nil {
		Log.Error().Str("name", s.base.Name).Msg("Reader is nil")
		return
	}
	s.reader.Reset()
}
