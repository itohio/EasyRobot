//go:build !tinygo && linux

package encoder

import "github.com/itohio/EasyRobot/x/devices"

func configurePins(pinA, pinB devices.Pin) error {
	// No-op for Linux (pins configured externally)
	return nil
}
