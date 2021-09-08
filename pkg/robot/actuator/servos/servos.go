package servos

//go:generate protoc -I=./ -I=${GOPATH}/pkg/mod/ -I=${GOPATH}/src --proto_path=../../ --gogofaster_out=./ pb/types.proto
//go:generate go run ../../../../cmd/codegen -i pb/types.pb.go -c ../../proto/proto.json -m re

import (
	"errors"
	"io"

	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/robot/actuator"
	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos/pb"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

const (
	// Device ID for robot.transport
	ID = 0x00000100
)

type Motor = pb.Motor
type Config = pb.Config
type State = kinematics.State

type servosClient struct {
	wr io.Writer
}

var (
	errNotSupported = errors.New("not supported")
)

func NewMotorConfig(opts ...options.Option) Motor {
	cfg := Motor{
		Pin:     0,
		Min:     -90,
		Max:     90,
		Default: 0,
		Scale:   500 / 90.0,
		Offset:  1500,
	}
	options.ApplyOptions(&cfg, opts...)
	return cfg
}

func New(writer io.Writer) actuator.Actuator {
	return &servosClient{wr: writer}
}

func (s *servosClient) Configure(opts ...actuator.ConfigureOption) error {
	for _, opt := range opts {
		dataType, data := opt()
		if err := transport.WritePacket(ID, dataType, s.wr, data); err != nil {
			return err
		}
	}

	return nil
}

func (s *servosClient) Get() ([]float32, error) {
	return nil, errNotSupported
}

func (s *servosClient) Set(params []float32) error {
	cfg := State{
		Params: params,
	}
	data, err := cfg.Marshal()
	if err != nil {
		return err
	}
	return transport.WritePacket(ID, transport.PacketSetState, s.wr, data)
}
