package bridge

import (
	"context"

	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

const BRIDGE_SEND_NAME = "w"

type bridgeSend struct {
	options Options
	ch      <-chan pipeline.Data
}

func init() {
	pipeline.Register(BRIDGE_SEND_NAME, NewBridgeSender)
}

func NewBridgeSender(opts ...plugin.Option) (pipeline.Step, error) {
	step := &bridgeSend{
		options: Options{
			base: plugin.DefaultOptions(),
		},
	}
	plugin.ApplyOptions(&step.options, opts...)
	plugin.ApplyOptions(&step.options.base, opts...)
	step.Reset()
	return step, nil
}

func (s *bridgeSend) In(ch <-chan pipeline.Data) {
	s.ch = ch
}

func (s *bridgeSend) Out() <-chan pipeline.Data {
	return nil
}

func (s *bridgeSend) Run(ctx context.Context) {
	s.options.Transport.Dial(s.options.Network, s.options.Address)
	defer s.options.Transport.Close()

	for {
		data, err := pipeline.StepReceive(ctx, s.options.base, s.ch)
		if err != nil {
			return
		}
		// FIXME pass ctx and handle errors
		s.options.Transport.Send(data)
	}
}

func (s *bridgeSend) Reset() {
}
