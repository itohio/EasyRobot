// +build sam,xiao

package main

//go:generate tinygo flash -target=xiao

import (
	"context"
	"machine"
	"time"

	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

var (
	uart = machine.Serial
	tx   = machine.UART_TX_PIN
	rx   = machine.UART_RX_PIN

	manipulatorConfig = []servos.Motor{
		servos.NewDefaultConfig(uint32(machine.D8)),
		servos.NewDefaultConfig(uint32(machine.D9)),
		servos.NewDefaultConfig(uint32(machine.D10)),
	}

	manipulator servos.Actuator
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
	manipulator, err = servos.New(manipulatorConfig)
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

		println("before for loop")
		for {
			buffer, n, data, err = transport.ReadPacketFromReliableStream(ctx, servos.ID, uart, buffer)
			if err != nil {
				println("read error", err)
				continue
			}

			if n == 0 {
				continue
			}

			println("inside for loop", data.Type)
			switch data.Type {
			case transport.PacketMotorConfig:
				println("motors")
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

	motors := []servos.Motor{
		*config.Motors[0],
		*config.Motors[1],
		*config.Motors[2],
	}

	manipulator.Configure(motors)
}
