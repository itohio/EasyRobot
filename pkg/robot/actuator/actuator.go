package actuator

import (
	motors "github.com/foxis/EasyRobot/pkg/robot/actuator/motors/pb"
	servos "github.com/foxis/EasyRobot/pkg/robot/actuator/servos/pb"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

type ConfigureOption func() (packetType uint32, packet []byte)

type Actuator interface {
	Configure(...ConfigureOption) error
	Get() ([]float32, error)
	Set(params []float32) error
}

func WithMotorConfig(cfg []motors.Motor) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := motors.Config{
			Motors: make([]*motors.Motor, len(cfg)),
		}
		for i, motor := range cfg {
			data.Motors[i] = &motor
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketMotorConfig
		return
	}
}

func WithServoConfig(cfg []servos.Motor) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := servos.Config{
			Motors: make([]*servos.Motor, len(cfg)),
		}
		for i, motor := range cfg {
			data.Motors[i] = &motor
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketMotorConfig
		return
	}
}

func WithDHKinematics(cfg []kinematics.DenavitHartenberg) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := kinematics.Config{
			DH: make([]*kinematics.DenavitHartenberg, len(cfg)),
		}
		for i, dh := range cfg {
			data.DH[i] = &dh
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketKinematicsConfig
		return
	}
}

func WithPlanarKinematics(cfg []kinematics.PlanarJoint) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := kinematics.Config{
			Planar: make([]*kinematics.PlanarJoint, len(cfg)),
		}
		for i, joint := range cfg {
			data.Planar[i] = &joint
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketKinematicsConfig
		return
	}
}

func WithPIDConfig(cfg []kinematics.PID) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := kinematics.Config{
			PID: make([]*kinematics.PID, len(cfg)),
		}
		for i, pid := range cfg {
			data.PID[i] = &pid
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketKinematicsConfig
		return
	}
}

func WithMotionConfig(cfg []kinematics.Motion) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := kinematics.Config{
			Motion: make([]*kinematics.Motion, len(cfg)),
		}
		for i, motion := range cfg {
			data.Motion[i] = &motion
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketKinematicsConfig
		return
	}
}

func WithWheelKinematics(cfg []kinematics.Wheel) ConfigureOption {
	return func() (packetType uint32, packet []byte) {
		data := kinematics.Config{
			Wheels: make([]*kinematics.Wheel, len(cfg)),
		}
		for i, wheel := range cfg {
			data.Wheels[i] = &wheel
		}
		packet, err := data.Marshal()
		if err != nil {
			return
		}
		packetType = transport.PacketKinematicsConfig
		return
	}
}
