package main

//go:generate tinygo flash -target=xiao -tags logless

import (
	"context"
	"machine"
	"time"

	"github.com/foxis/EasyRobot/pkg/core/math/filter/vaj"
	servos "github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
	fw "github.com/foxis/EasyRobot/pkg/robot/actuator/servos/fw"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

var (
	manipulatorConfig = []servos.Motor{
		servos.NewMotorConfig(servos.WithPin(uint32(machine.D8))),
		servos.NewMotorConfig(servos.WithPin(uint32(machine.D9))),
		servos.NewMotorConfig(servos.WithPin(uint32(machine.D10))),
	}

	manipulator fw.Actuator
	motion      []vaj.VAJ1D
)

func blink(led machine.Pin, t time.Duration) {
	for {
		time.Sleep(t)
		led.Set(!led.Get())
	}
}

func main() {
	led := machine.LED
	led.Configure(machine.PinConfig{Mode: machine.PinOutput})
	uart.Configure(machine.UARTConfig{TX: tx, RX: rx})

	defer blink(led, time.Millisecond*1500)

	var err error
	manipulator, err = fw.New(manipulatorConfig, timerMapping)
	if err != nil {
		return
	}
	ctx := context.Background()
	buffer := make([]byte, 128)
	var (
		n    int
		data transport.PacketData
	)
	for {
		//ch := transport.ReadPackets(ctx, servos.ID, uart)

		for {
			buffer, n, data, err = transport.ReadPacketFromReliableStream(ctx, servos.ID, uart, buffer)
			if err != nil {
				println("read error", err)
				continue
			}

			if n == 0 {
				continue
			}

			switch data.Type {
			case transport.PacketMotorConfig:
				println("motors config")
				configMotors(data)
			case transport.PacketKinematicsConfig:
				println("kinematics")
				configKinematics(data)
			case transport.PacketSetState:
				println("set State")
				setState(data)
			}

			led.Set(!led.Get())
		}
		println("reader failed")
	}
	blink(led, time.Millisecond*50)
}

func configMotors(packet transport.PacketData) {
	var config servos.Config
	err := config.Unmarshal(packet.Data)
	if err != nil {
		return
	}
	if len(config.Motors) != len(manipulatorConfig) {
		return
	}

	motors := make([]servos.Motor, len(config.Motors))
	for i, m := range config.Motors {
		motors[i] = *m
	}

	if err := manipulator.Configure(motors); err != nil {
		motion = NewMotion(len(motors))
	}
}

func NewMotion(N int) []vaj.VAJ1D {
	m := make([]vaj.VAJ1D, N)
	for i := range m {
		m[i] = vaj.New1D(100, 100, 100)
	}
	return m
}

func configMotionKinematics(packet transport.PacketData) {
	var cfg kinematics.Config
	err := cfg.Unmarshal(packet.Data)
	if err != nil {
		return
	}
	if cfg.Motion == nil || len(cfg.Motion) != len(manipulatorConfig) {
		return
	}

	for i := range motion {
		if cfg.Motion[i] == nil {
			continue
		}
		motion[i] = vaj.New1D(cfg.Motion[i].Velocity, cfg.Motion[i].Acceleration, cfg.Motion[i].Jerk)
	}
}
