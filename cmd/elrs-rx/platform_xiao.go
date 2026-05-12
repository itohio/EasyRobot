//go:build sam && xiao

package main

import (
	"io"
	"machine"

	"github.com/itohio/EasyRobot/x/devices/crsf"
)

const (
	crsfBaud       = 420000
	numPWMChannels = 5
)

var pwmPins = [numPWMChannels]machine.Pin{
	machine.D2, machine.D3, machine.D4,
	machine.D8, machine.D9,
}

func initHardware() {
	machine.LED.Configure(machine.PinConfig{Mode: machine.PinOutput})
	machine.Serial.Configure(machine.UARTConfig{
		BaudRate: crsfBaud,
		TX:       machine.UART_TX_PIN,
		RX:       machine.UART_RX_PIN,
	})
	for i := range numPWMChannels {
		pwmPins[i].Configure(machine.PinConfig{Mode: machine.PinOutput})
	}
}

func getStream() (io.Reader, io.Writer) {
	return machine.Serial, machine.Serial
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
	return 11.1 // TODO: read ADC via machine.ADC
}

func setPWMMicroseconds(pin machine.Pin, us uint32) {
	// TODO: implement via XIAO TCC PWM
	_ = pin
	_ = us
}
