package nats

import (
	"context"
	"fmt"
	"time"

	. "github.com/itohio/EasyRobot/pkg/core/logger"
	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/store"

	natsgo "github.com/nats-io/nats.go"
)

type nats struct {
	Options
	ch  <-chan pipeline.Data
	out chan pipeline.Data
}

func init() {
	pipeline.Register(NAME+"GoCV", NewGoCV)
}

func NewGoCV(opts ...options.Option) (pipeline.Step, error) {
	step := &nats{
		Options: Options{
			base: plugin.DefaultOptions(),
		},
	}
	options.ApplyOptions(&step.Options, opts...)
	options.ApplyOptions(&step.Options.base, plugin.WithName(NAME+"GoCV"))
	options.ApplyOptions(&step.Options.base, opts...)
	step.Reset()
	return step, nil
}

func (s *nats) In(ch <-chan pipeline.Data) {
	s.ch = ch
}

func (s *nats) Out() <-chan pipeline.Data {
	return s.out
}

func (s *nats) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")

	nc, err := s.connect()
	if err != nil {
		return
	}
	defer nc.Close()

	err = s.subscribeTo(ctx, nc, s.out)
	if err != nil {
		return
	}

	for {
		in, err := pipeline.StepReceive(ctx, s.base, s.ch)
		if err != nil {
			return
		}

		err = s.publishData(in, nc)
		if err != nil {
			return
		}

		if s.keys == nil {
			in.Close(store.ANY)
		} else {
			for _, key := range s.keys {
				in.Close(key)
			}
		}
	}
}

func (s *nats) connect() (*natsgo.Conn, error) {
	totalWait := 10 * time.Minute
	reconnectDelay := time.Second

	opts := []natsgo.Option{natsgo.Name(s.base.Name)}
	opts = append(opts, natsgo.ReconnectWait(reconnectDelay))
	opts = append(opts, natsgo.MaxReconnects(int(totalWait/reconnectDelay)))
	opts = append(opts, natsgo.DisconnectHandler(func(nc *natsgo.Conn) {
		Log.Debug().Str("name", s.base.Name).Msg(fmt.Sprintf("Disconnected: will attempt reconnects for %.0fm", totalWait.Minutes()))
	}))
	opts = append(opts, natsgo.ReconnectHandler(func(nc *natsgo.Conn) {
		Log.Debug().Str("name", s.base.Name).Msg(fmt.Sprintf("Reconnected [%s]", nc.ConnectedUrl()))
	}))
	opts = append(opts, natsgo.ClosedHandler(func(nc *natsgo.Conn) {
		Log.Fatal().Str("name", s.base.Name).Msg(fmt.Sprintf("Exiting: %v", nc.LastError()))
	}))

	// Use UserCredentials
	if s.creds != "" {
		opts = append(opts, natsgo.UserCredentials(s.creds))
	}

	// Connect to NATS
	nc, err := natsgo.Connect(s.urls, opts...)
	if err != nil {
		Log.Fatal().Str("name", s.base.Name).Err(err)
		return nil, err
	}

	return nc, nil
}

func (s *nats) publishData(in store.Store, nc *natsgo.Conn) error {
	if s.topicPub == "" {
		return nil
	}

	data, err := s.encode(in)
	if err != nil {
		Log.Fatal().Str("name", s.base.Name).Err(err)
		return err
	}

	nc.Publish(s.topicPub, data)

	return nil
}

func (s *nats) subscribeTo(ctx context.Context, nc *natsgo.Conn, ch chan pipeline.Data) error {
	if s.topicSub == "" {
		return nil
	}

	nc.Subscribe(s.topicSub, func(msg *natsgo.Msg) {
		data, err := s.decode(msg.Data)
		if err == nil {
			pipeline.StepSend(ctx, s.base, ch, data)
		} else {
			Log.Fatal().Str("name", s.base.Name).Err(err)
		}
	})

	return nil
}

func (s *nats) Reset() {
	s.out = pipeline.StepMakeChan(s.base)
}
