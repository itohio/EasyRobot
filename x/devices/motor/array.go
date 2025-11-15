package motor

import (
	"fmt"
	"sync"

	"github.com/itohio/EasyRobot/x/devices"
)

// MotorArray implements a multi-channel motor actuator that controls an array of motors using PID speed control.
// This array manages multiple motors simultaneously, with each motor controlled via PWM and encoder feedback.
type MotorArray struct {
	mu     sync.Mutex
	pwm    devices.PWMDevice
	motors []*Motor
}

// NewMotorArray creates a new motor array for controlling multiple motors.
// Each motor in the array will be created with its own configuration.
func NewMotorArray(pwm devices.PWMDevice, configs []Config) (*MotorArray, error) {
	if pwm == nil {
		return nil, fmt.Errorf("PWM device is required")
	}

	if len(configs) == 0 {
		return nil, fmt.Errorf("at least one motor configuration is required")
	}

	array := &MotorArray{
		pwm:    pwm,
		motors: make([]*Motor, len(configs)),
	}

	// Create each motor
	for i, config := range configs {
		motor, err := New(pwm, config)
		if err != nil {
			return nil, fmt.Errorf("failed to create motor %d: %w", i, err)
		}
		array.motors[i] = motor
	}

	return array, nil
}

// Enable enables all motors in the array.
func (a *MotorArray) Enable() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for i, motor := range a.motors {
		if err := motor.Enable(); err != nil {
			return fmt.Errorf("failed to enable motor %d: %w", i, err)
		}
	}

	return nil
}

// Disable disables all motors in the array.
func (a *MotorArray) Disable() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for i, motor := range a.motors {
		if err := motor.Disable(); err != nil {
			return fmt.Errorf("failed to disable motor %d: %w", i, err)
		}
	}

	return nil
}

// SetSpeeds sets the target speeds for all motors in the array.
// speeds is an array of target speeds in RPM (positive = forward, negative = reverse).
// The length must match the number of configured motors.
func (a *MotorArray) SetSpeeds(speeds []float32) error {
	if len(speeds) != len(a.motors) {
		return fmt.Errorf("speed count mismatch: got %d, expected %d", len(speeds), len(a.motors))
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	for i, speed := range speeds {
		if i >= len(a.motors) {
			continue
		}

		if err := a.motors[i].SetSpeed(speed); err != nil {
			return fmt.Errorf("failed to set speed for motor %d: %w", i, err)
		}
	}

	return nil
}

// Speeds returns the current speeds for all motors in the array.
func (a *MotorArray) Speeds() []float32 {
	a.mu.Lock()
	defer a.mu.Unlock()

	speeds := make([]float32, len(a.motors))
	for i, motor := range a.motors {
		speeds[i] = motor.Speed()
	}

	return speeds
}

// TargetSpeeds returns the target speeds for all motors in the array.
func (a *MotorArray) TargetSpeeds() []float32 {
	a.mu.Lock()
	defer a.mu.Unlock()

	speeds := make([]float32, len(a.motors))
	for i, motor := range a.motors {
		speeds[i] = motor.TargetSpeed()
	}

	return speeds
}

// Close stops all motors and cleans up resources.
func (a *MotorArray) Close() error {
	return a.Disable()
}

