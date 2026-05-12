//go:build rp2040

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
	machine.GP2, machine.GP3, machine.GP4,
	machine.GP5, machine.GP6, machine.GP7,
}

func initHardware() {
	machine.LED.Configure(machine.PinConfig{Mode: machine.PinOutput})
	machine.UART0.Configure(machine.UARTConfig{
		BaudRate: crsfBaud,
		TX:       machine.UART0_TX_PIN,
		RX:       machine.UART0_RX_PIN,
	})
	for i := range numPWMChannels {
		pwmPins[i].Configure(machine.PinConfig{Mode: machine.PinOutput})
	}
}

func getStream() (io.Reader, io.Writer) {
	return machine.UART0, machine.UART0
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
	// TODO: implement via machine.PWM peripherals
	_ = pin
	_ = us
}
