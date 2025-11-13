//go:build tinygo

package encoder

func configurePins(pinA, pinB devices.Pin) {
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
}
