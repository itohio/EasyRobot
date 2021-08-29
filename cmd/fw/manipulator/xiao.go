// +build sam,xiao

package main

//go:generate tinygo flash -target=xiao -tags logless

import (
	"machine"

	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
)

var (
	uart = machine.Serial
	tx   = machine.UART_TX_PIN
	rx   = machine.UART_RX_PIN

	manipulatorConfig = []servos.Motor{
		servos.NewDefaultConfig(servos.WithPin(uint32(machine.D8))),
		servos.NewDefaultConfig(servos.WithPin(uint32(machine.D9))),
		servos.NewDefaultConfig(servos.WithPin(uint32(machine.D10))),
	}

	manipulator servos.Actuator
)
