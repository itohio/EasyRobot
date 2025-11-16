//go:build tinygo

package encoder

import "github.com/itohio/EasyRobot/x/devices"

func configurePins(pinA, pinB devices.Pin) error {
	// Configure pins as inputs with pull-up (encoders typically have pull-ups)
	if err := pinA.Configure(devices.PinConfig{
		Mode: devices.PinInputPullup,
	}); err != nil {
		return err
	}
	if err := pinB.Configure(devices.PinConfig{
		Mode: devices.PinInputPullup,
	}); err != nil {
		return err
	}
	return nil
}
