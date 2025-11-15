//go:build sam && xiao

package main

import (
	"machine"

	xiao "github.com/itohio/EasyRobot/x/devices/xiao"
)

var (
	uart = machine.Serial
	tx   = machine.UART_TX_PIN
	rx   = machine.UART_RX_PIN
)

// timerMapping provides backward compatibility for existing servo code.
// Maps XIAO pins to their TCC peripherals.
// TODO: Migrate servo controls to use xiao.NewPWMDevice() directly instead of this function
func timerMapping(pin machine.Pin) (*machine.TCC, bool) {
	// Use XIAO PWM device to get the mapping
	// Create a temporary device instance to access the mapping
	pwmDevice := xiao.NewPWMDevice().(*xiao.XIAOPWMDevice)
	mapping, ok := pwmDevice.GetMapping(pin)
	if !ok {
		return nil, false
	}
	return mapping.TCC, true
}
