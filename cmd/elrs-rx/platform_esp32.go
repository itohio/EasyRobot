//go:build esp32

package main

import (
	"io"
	"machine"

	"github.com/itohio/EasyRobot/x/devices/crsf"
)

const (
	crsfBaud       = 420000
	numPWMChannels = 6
)

var pwmPins = [numPWMChannels]machine.Pin{
	machine.IO25, machine.IO26, machine.IO27,
	machine.IO14, machine.IO12, machine.IO13,
}

func initHardware() {
	machine.LED.Configure(machine.PinConfig{Mode: machine.PinOutput})
	machine.UART1.Configure(machine.UARTConfig{
		BaudRate: crsfBaud,
		TX:       machine.UART1_TX_PIN,
		RX:       machine.UART1_RX_PIN,
	})
	for i := range numPWMChannels {
		pwmPins[i].Configure(machine.PinConfig{Mode: machine.PinOutput})
	}
}

func getStream() (io.Reader, io.Writer) {
	return machine.UART1, machine.UART1
}

func onChannels(ch *crsf.Channels) {
	for i := range numPWMChannels {
		setPWMMicroseconds(pwmPins[i], uint32(ch.Microseconds(i)))
	}
}

func onLinkStats(ls *crsf.LinkStatistics) {
	_ = ls
}

func heartbeat() {
	machine.LED.Set(!machine.LED.Get())
}

func readBatteryVoltage() float32 {
	return 11.1 // TODO: read ADC
}

func setPWMMicroseconds(pin machine.Pin, us uint32) {
	// TODO: implement via ESP32 LEDC PWM
	_ = pin
	_ = us
}
