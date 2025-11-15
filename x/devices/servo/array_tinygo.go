//go:build tinygo

package servo

import (
	"fmt"

	"machine"

	"github.com/itohio/EasyRobot/x/devices"
)

// pinNumberToPin converts a pin number to a devices.Pin for TinyGo.
// For TinyGo, machine.Pin directly implements devices.Pin.
func pinNumberToPin(pinNum uint32) devices.Pin {
	return machine.Pin(pinNum)
}

// newServoArray creates a new servo array for TinyGo.
func newServoArray(pwm devices.PWMDevice, motors []Motor) (*ServoArray, error) {
	if pwm == nil {
		return nil, fmt.Errorf("PWM device is required")
	}

	// Configure PWM for servo frequency (50Hz typical)
	frequency := uint32(50)
	if err := pwm.Configure(frequency); err != nil {
		return nil, fmt.Errorf("failed to configure PWM: %w", err)
	}

	array := &ServoArray{
		pwm:       pwm,
		motors:    make([]Motor, len(motors)),
		channels:  make([]devices.PWM, len(motors)),
		frequency: frequency,
	}
	copy(array.motors, motors)

	// Get PWM channels for each motor in the array
	for i, motor := range motors {
		// Convert pin number to devices.Pin
		pin := pinNumberToPin(motor.Pin)

		// Get PWM channel from PWM device
		ch, err := pwm.Channel(pin)
		if err != nil {
			return nil, fmt.Errorf("failed to get PWM channel for pin %d: %w", motor.Pin, err)
		}

		array.channels[i] = ch

		// Set to default position
		defaultUs := uint32(motor.DefaultUs)
		if err := ch.SetMicroseconds(defaultUs); err != nil {
			return nil, fmt.Errorf("failed to set default position for motor %d: %w", i, err)
		}
	}

	return array, nil
}
