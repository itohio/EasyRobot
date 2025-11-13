//go:build !tinygo && linux

package encoder

import "github.com/itohio/EasyRobot/x/devices"

func configurePins(pinA, pinB devices.Pin) {
	// No-op
}
