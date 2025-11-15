package devices

// PWM represents a PWM (Pulse Width Modulation) channel.
// It allows setting duty cycle and frequency for PWM output.
type PWM interface {
	// Set sets the duty cycle for this PWM channel.
	// duty is in range 0.0 to 1.0, where 0.0 = 0% and 1.0 = 100%
	Set(duty float32) error

	// SetMicroseconds sets the pulse width in microseconds.
	// This is commonly used for servo control (typically 500-2500 microseconds).
	SetMicroseconds(us uint32) error

	// Stop stops the PWM output (sets duty to 0).
	Stop() error
}

// PWMDevice represents a PWM controller device that can provide PWM channels.
// Different platforms may have different PWM controllers (TCC on SAM, etc.)
type PWMDevice interface {
	// Channel returns a PWM channel for the specified pin.
	// Returns an error if the pin does not support PWM or is already in use.
	Channel(pin Pin) (PWM, error)

	// Configure configures the PWM device with the specified frequency.
	// frequency is in Hz (typically 50Hz for servos, higher for LEDs, etc.)
	Configure(frequency uint32) error

	// SetFrequency changes the PWM frequency for all channels.
	// Note: Some platforms may require stopping PWM before changing frequency.
	SetFrequency(frequency uint32) error
}
