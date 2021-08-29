package servos

import (
	"errors"
	io "io"

	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

type servosClient struct {
	wr io.Writer
}

var (
	errNotSupported = errors.New("not supported")
)

func (s *servosClient) Configure(motors []Motor) error {
	cfg := Config{
		Motors: make([]*Motor, len(motors)),
	}
	for i, motor := range motors {
		cfg.Motors[i] = &motor
	}
	data, err := cfg.Marshal()
	if err != nil {
		return err
	}
	return transport.WritePacket(ID, transport.PacketMotorConfig, s.wr, data)
}

func (s *servosClient) ConfigureKinematics(dhArr []kinematics.DenavitHartenberg) error {
	cfg := kinematics.Config{
		DH: make([]*kinematics.DenavitHartenberg, len(dhArr)),
	}
	for i, dh := range dhArr {
		cfg.DH[i] = &dh
	}
	data, err := cfg.Marshal()
	if err != nil {
		return err
	}
	return transport.WritePacket(ID, transport.PacketKinematicsConfig, s.wr, data)
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
