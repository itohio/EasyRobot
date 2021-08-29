package bridge

import (
	"context"

	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

const BRIDGE_RECEIVE_NAME = "r"

type bridgeReceive struct {
	options Options
	ch      chan pipeline.Data
}

func init() {
	pipeline.Register(BRIDGE_RECEIVE_NAME, NewBridgeReceiver)
}

func NewBridgeReceiver(opts ...options.Option) (pipeline.Step, error) {
	step := &bridgeReceive{
		options: Options{
			base: plugin.DefaultOptions(),
		},
	}
	options.ApplyOptions(&step.options, opts...)
	options.ApplyOptions(&step.options.base, opts...)
	step.Reset()
	return step, nil
}

func (s *bridgeReceive) In(ch <-chan pipeline.Data) {
}

func (s *bridgeReceive) Out() <-chan pipeline.Data {
	return s.ch
}

func (s *bridgeReceive) Run(ctx context.Context) {
	ch := s.options.Transport.Listen(s.options.Network, s.options.Address)
	defer s.options.Transport.Close()

	for {
		data, err := pipeline.StepReceive(ctx, s.options.base, ch)
		if err != nil {
			return
		}
		err = pipeline.StepSend(ctx, s.options.base, s.ch, data)
		if err != nil {
			return
		}
	}
}

func (s *bridgeReceive) Reset() {
	s.ch = pipeline.StepMakeChan(s.options.base)
}
