package servo

// Motor represents a single servo motor configuration.
type Motor struct {
	Pin       uint32  // GPIO pin number
	MinUs     float32 // Minimum pulse width in microseconds
	MaxUs     float32 // Maximum pulse width in microseconds
	DefaultUs float32 // Default/center pulse width in microseconds
	MaxAngle  float32 // Maximum angle in degrees
	Default   float32 // Default angle in degrees (computed from DefaultUs)
}

// MotorOption configures a Motor.
type MotorOption func(*Motor)

// NewMotorConfig creates a new motor configuration with the specified options.
func NewMotorConfig(opts ...MotorOption) Motor {
	m := Motor{
		MinUs:     500,
		MaxUs:     2500,
		DefaultUs: 1500,
		MaxAngle:  180,
		Default:   90,
	}
	for _, opt := range opts {
		opt(&m)
	}
	// Compute Default angle from DefaultUs if not explicitly set
	if m.Default == 90 && m.DefaultUs != 1500 {
		// Linear mapping: (DefaultUs - MinUs) / (MaxUs - MinUs) * MaxAngle
		rangeUs := m.MaxUs - m.MinUs
		if rangeUs > 0 {
			normalized := (m.DefaultUs - m.MinUs) / rangeUs
			m.Default = normalized * m.MaxAngle
		}
	}
	return m
}

// WithPin sets the GPIO pin number.
func WithPin(pin uint32) MotorOption {
	return func(m *Motor) {
		m.Pin = pin
	}
}

// WithMicroseconds sets the pulse width parameters in microseconds.
// minUs: minimum pulse width (e.g., 500 for most servos)
// maxUs: maximum pulse width (e.g., 2500 for most servos)
// defaultUs: default/center pulse width (e.g., 1500 for most servos)
// maxAngle: maximum angle in degrees (e.g., 180)
func WithMicroseconds(minUs, maxUs, defaultUs, maxAngle float32) MotorOption {
	return func(m *Motor) {
		m.MinUs = minUs
		m.MaxUs = maxUs
		m.DefaultUs = defaultUs
		m.MaxAngle = maxAngle
		// Compute default angle
		rangeUs := maxUs - minUs
		if rangeUs > 0 {
			normalized := (defaultUs - minUs) / rangeUs
			m.Default = normalized * maxAngle
		}
	}
}

// AngleToMicroseconds converts an angle (degrees) to pulse width (microseconds).
func (m *Motor) AngleToMicroseconds(angle float32) float32 {
	// Clamp angle to valid range
	if angle < 0 {
		angle = 0
	}
	if angle > m.MaxAngle {
		angle = m.MaxAngle
	}

	// Linear mapping: angle / MaxAngle -> (0..1) -> (MinUs..MaxUs)
	normalized := angle / m.MaxAngle
	return m.MinUs + normalized*(m.MaxUs-m.MinUs)
}

// MicrosecondsToAngle converts pulse width (microseconds) to angle (degrees).
func (m *Motor) MicrosecondsToAngle(us float32) float32 {
	// Clamp microseconds to valid range
	if us < m.MinUs {
		us = m.MinUs
	}
	if us > m.MaxUs {
		us = m.MaxUs
	}

	// Linear mapping: (us - MinUs) / (MaxUs - MinUs) -> (0..1) -> (0..MaxAngle)
	rangeUs := m.MaxUs - m.MinUs
	if rangeUs <= 0 {
		return 0
	}
	normalized := (us - m.MinUs) / rangeUs
	return normalized * m.MaxAngle
}

