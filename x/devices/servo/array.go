package servo

import (
	"fmt"
	"sync"

	"github.com/itohio/EasyRobot/x/devices"
)

// ServoArray implements a multi-channel servo actuator that controls an array of servo motors using PWM.
// This array manages multiple servos simultaneously, with each servo controlled via PWM channels.
type ServoArray struct {
	mu        sync.Mutex
	pwm       devices.PWMDevice
	motors    []Motor
	channels  []devices.PWM
	frequency uint32
}

// NewServoArray creates a new servo array for controlling multiple servo motors.
// The PWM device will be configured for servo frequency (50Hz typical).
// Each motor in the array will be configured with its own PWM channel based on the motor's pin number.
func NewServoArray(pwm devices.PWMDevice, motors []Motor) (*ServoArray, error) {
	return newServoArray(pwm, motors)
}

// Configure configures the array with new motor configurations.
// The number of motors must match the originally configured count.
func (a *ServoArray) Configure(motors []Motor) error {
	if len(motors) != len(a.motors) {
		return fmt.Errorf("motor count mismatch: got %d, expected %d", len(motors), len(a.motors))
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Update motor configurations
	a.motors = make([]Motor, len(motors))
	copy(a.motors, motors)

	return nil
}

// Set sets the servo positions for all channels in the array.
// values is an array of angles in degrees (0 to MaxAngle for each motor).
// The length must match the number of configured motors.
func (a *ServoArray) Set(values []float32) error {
	if len(values) != len(a.channels) {
		return fmt.Errorf("value count mismatch: got %d, expected %d", len(values), len(a.channels))
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	for i, value := range values {
		if i >= len(a.motors) || i >= len(a.channels) {
			continue
		}

		motor := a.motors[i]
		ch := a.channels[i]

		// Convert angle to microseconds
		us := motor.AngleToMicroseconds(value)

		// Set PWM pulse width
		if err := ch.SetMicroseconds(uint32(us)); err != nil {
			return fmt.Errorf("failed to set motor %d: %w", i, err)
		}
	}

	return nil
}

// Stop stops all servo outputs (sets to default positions).
func (a *ServoArray) Stop() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	values := make([]float32, len(a.motors))
	for i := range a.motors {
		values[i] = a.motors[i].Default
	}

	return a.Set(values)
}

