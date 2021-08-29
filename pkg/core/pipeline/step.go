package pipeline

import (
	"context"
	"errors"

	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

var ErrEOS = errors.New("end of stream")
var ErrDrop = errors.New("dropped data")

type Data = store.Store

type Step interface {
	In(<-chan Data)
	Out() <-chan Data
	Run(ctx context.Context)
	Reset()
}

type NamedStep interface {
	Name() string
}

type MarshallableStep interface {
	Step
	Size() int
	MarshalToSizedBuffer([]byte) (int, error)
	Unmarshal([]byte) (Step, error)
}

type StepConfigurator interface {
	Config() interface{}
	SetConfig(opts ...options.Option)
}

func StepReceive(ctx context.Context, o plugin.Options, in <-chan Data) (Data, error) {
	select {
	case <-ctx.Done():
		return nil, ErrEOS
	case data, ok := <-in:
		if !ok {
			return nil, ErrEOS
		}

		return data, nil
	}
}

func StepSend(ctx context.Context, o plugin.Options, out chan Data, data Data) error {
	if o.Blocking {
		select {
		case out <- data:
		case <-ctx.Done():
			return ErrEOS
		}
	} else {
		select {
		case out <- data:
		case <-ctx.Done():
			return ErrEOS
		default:
			return ErrDrop
		}
	}
	return nil
}

func StepMakeChan(o plugin.Options) chan Data {
	if o.BufferSize > 0 {
		return make(chan Data)
	} else {
		return make(chan Data, o.BufferSize)
	}
}
