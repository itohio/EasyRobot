//go:build unitv

package unitv

import (
	"machine"

	"github.com/itohio/EasyRobot/drivers/camera"
)

// UnitV pins (M5Stack UnitV pin map).
const (
	// WS2812 is the data line of the single WS2812 RGB LED. It is not a plain
	// on/off LED: driving it needs WS2812 timing.
	WS2812 = machine.P08

	ButtonA = machine.P19
	ButtonB = machine.P18

	// Grove (HY2.0-4P) signal pins; usable as I²C, UART or GPIO via FPIOA.
	GroveP34 = machine.P34
	GroveP35 = machine.P35

	// UARTHS on the K210 ISP pins, wired to the FTDI USB-serial bridge.
	// machine.UART0 uses these by default.
	UARTRX = machine.P04
	UARTTX = machine.P05
)

// CameraPins are the camera control pins: the Sipeed/MaixPy K210 board
// mapping IO40–IO47, which UnitV shares (verified by reading the sensor id).
var CameraPins = camera.Pins{
	SDA:   machine.P40,
	SCL:   machine.P41,
	RST:   machine.P42,
	VSYNC: machine.P43,
	PWDN:  machine.P44,
	HREF:  machine.P45,
	XCLK:  machine.P46,
	PCLK:  machine.P47,
}
