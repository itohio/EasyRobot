// +build sam,xiao

package main

import (
	"machine"

	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
	fw "github.com/foxis/EasyRobot/pkg/robot/actuator/servos/fw"
)

var (
	uart = machine.Serial
	tx   = machine.UART_TX_PIN
	rx   = machine.UART_RX_PIN
)

func timerMapping(pin machine.Pin) (*machine.TCC, bool) {
	switch pin {
	case machine.D9:
		return machine.TCC0, true
	case machine.D8:
		return machine.TCC1, true
	case machine.D10:
		return machine.TCC1, true
	default:
		return nil, false
	}
}
