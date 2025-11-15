package motor

import (
	"github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/encoder"
)

// Type represents the motor driver type (how the motor is connected).
type Type int

const (
	// TypeDirPWM uses one direction pin and one PWM pin.
	// Direction is controlled by the dir pin (high/low).
	// Speed is controlled by PWM duty cycle on the pwm pin.
	TypeDirPWM Type = iota

	// TypeABPWM uses two pins (A and B) both with PWM.
	// Speed and direction are controlled by the relative PWM duty cycles:
	// - Forward: A=PWM, B=0
	// - Reverse: A=0, B=PWM
	// - Stop: A=0, B=0
	TypeABPWM

	// TypeABDirPWM uses two pins (A and B) with direction control and PWM.
	// Direction is controlled by setting A high/low (B is opposite).
	// Speed is controlled by PWM duty cycle on the pwm pin.
	// Note: This is similar to TypeDirPWM but uses A/B pins instead of dir/pwm.
	TypeABDirPWM
)

// Config holds configuration for a motor.
type Config struct {
	// Motor driver type
	Type Type

	// Pins configuration (depends on Type)
	// For TypeDirPWM: Dir and PWM are used
	// For TypeABPWM: PinA and PinB are used (both PWM)
	// For TypeABDirPWM: PinA, PinB, and PWM are used
	Dir  devices.Pin // Direction pin (TypeDirPWM)
	PWM  devices.Pin // PWM pin (TypeDirPWM, TypeABDirPWM)
	PinA devices.Pin // Pin A (TypeABPWM, TypeABDirPWM)
	PinB devices.Pin // Pin B (TypeABPWM, TypeABDirPWM)

	// Encoder for feedback
	Encoder *encoder.Encoder

	// PID gains for speed control
	PIDGains struct {
		P float32 // Proportional gain
		I float32 // Integral gain
		D float32 // Derivative gain
	}

	// PID output limits (clamped to [-MaxOutput, MaxOutput])
	MaxOutput float32 // Maximum PID output (default: 1.0)

	// Control parameters
	SamplePeriod float32 // PID update period in seconds (default: 0.01 = 10ms)

	// Max speed in RPM (for scaling PID output)
	MaxRPM float32 // Maximum motor speed in RPM (default: 100)
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		Type:        TypeDirPWM,
		PIDGains:    struct{ P, I, D float32 }{P: 1.0, I: 0.1, D: 0.01},
		MaxOutput:   1.0,
		SamplePeriod: 0.01, // 10ms
		MaxRPM:      100,
	}
}

