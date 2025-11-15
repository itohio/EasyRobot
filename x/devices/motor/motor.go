package motor

import (
	"fmt"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/math/control/pid"
)

// Motor represents a motor with PID speed control.
// It reads encoder feedback and controls PWM and direction.
type Motor struct {
	mu sync.Mutex

	config Config
	pwm    devices.PWMDevice

	// PWM channels (depends on motor type)
	pwmChannel  devices.PWM // For TypeDirPWM and TypeABDirPWM
	pwmChannelA devices.PWM // For TypeABPWM
	pwmChannelB devices.PWM // For TypeABPWM

	// PID controller for speed control
	pid pid.PID1D

	// Current state
	targetSpeed  float32 // Target speed in RPM (can be negative for reverse)
	currentSpeed float32 // Current speed in RPM (from encoder)
	direction    int     // Current direction: 1 = forward, -1 = reverse, 0 = stopped

	// Control
	enabled bool
	ticker  *time.Ticker
	stopCh  chan struct{}
}

// New creates a new motor with the specified configuration.
func New(pwm devices.PWMDevice, config Config) (*Motor, error) {
	if pwm == nil {
		return nil, fmt.Errorf("PWM device is required")
	}

	// Validate configuration
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Configure PWM for motor frequency (typically 20kHz or higher for motors)
	frequency := uint32(20000) // 20kHz typical for motor control
	if err := pwm.Configure(frequency); err != nil {
		return nil, fmt.Errorf("failed to configure PWM: %w", err)
	}

	m := &Motor{
		config: config,
		pwm:    pwm,
		stopCh: make(chan struct{}),
	}

	// Initialize PID controller
	m.pid = pid.New1D(
		config.PIDGains.P,
		config.PIDGains.I,
		config.PIDGains.D,
		-config.MaxOutput,
		config.MaxOutput,
	)
	m.pid.Reset()

	// Get PWM channels based on motor type
	if err := m.setupPWMChannels(); err != nil {
		return nil, fmt.Errorf("failed to setup PWM channels: %w", err)
	}

	return m, nil
}

// validateConfig validates the motor configuration.
func validateConfig(config Config) error {
	switch config.Type {
	case TypeDirPWM:
		if config.Dir == nil {
			return fmt.Errorf("Dir pin is required for TypeDirPWM")
		}
		if config.PWM == nil {
			return fmt.Errorf("PWM pin is required for TypeDirPWM")
		}
	case TypeABPWM:
		if config.PinA == nil {
			return fmt.Errorf("PinA is required for TypeABPWM")
		}
		if config.PinB == nil {
			return fmt.Errorf("PinB is required for TypeABPWM")
		}
	case TypeABDirPWM:
		if config.PinA == nil {
			return fmt.Errorf("PinA is required for TypeABDirPWM")
		}
		if config.PinB == nil {
			return fmt.Errorf("PinB is required for TypeABDirPWM")
		}
		if config.PWM == nil {
			return fmt.Errorf("PWM pin is required for TypeABDirPWM")
		}
	default:
		return fmt.Errorf("invalid motor type: %d", config.Type)
	}

	if config.Encoder == nil {
		return fmt.Errorf("encoder is required")
	}

	if config.SamplePeriod <= 0 {
		return fmt.Errorf("sample period must be positive")
	}

	if config.MaxRPM <= 0 {
		return fmt.Errorf("max RPM must be positive")
	}

	return nil
}

// setupPWMChannels sets up PWM channels based on motor type.
func (m *Motor) setupPWMChannels() error {
	var err error

	switch m.config.Type {
	case TypeDirPWM:
		// One PWM channel for speed
		m.pwmChannel, err = m.pwm.Channel(m.config.PWM)
		if err != nil {
			return fmt.Errorf("failed to get PWM channel: %w", err)
		}

	case TypeABPWM:
		// Two PWM channels for A and B
		m.pwmChannelA, err = m.pwm.Channel(m.config.PinA)
		if err != nil {
			return fmt.Errorf("failed to get PWM channel A: %w", err)
		}
		m.pwmChannelB, err = m.pwm.Channel(m.config.PinB)
		if err != nil {
			return fmt.Errorf("failed to get PWM channel B: %w", err)
		}

	case TypeABDirPWM:
		// One PWM channel for speed
		m.pwmChannel, err = m.pwm.Channel(m.config.PWM)
		if err != nil {
			return fmt.Errorf("failed to get PWM channel: %w", err)
		}
		// Direction pins are digital (already provided in config)
	}

	return nil
}

// SetSpeed sets the target speed in RPM.
// Positive values = forward, negative values = reverse.
func (m *Motor) SetSpeed(rpm float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Clamp to max RPM
	if rpm > m.config.MaxRPM {
		rpm = m.config.MaxRPM
	} else if rpm < -m.config.MaxRPM {
		rpm = -m.config.MaxRPM
	}

	m.targetSpeed = rpm
	m.pid.Target = rpm

	// Determine direction
	if rpm > 0.1 {
		m.direction = 1
	} else if rpm < -0.1 {
		m.direction = -1
	} else {
		m.direction = 0
	}

	return nil
}

// Speed returns the current speed in RPM.
func (m *Motor) Speed() float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.currentSpeed
}

// TargetSpeed returns the target speed in RPM.
func (m *Motor) TargetSpeed() float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.targetSpeed
}

// Enable starts the motor control loop.
func (m *Motor) Enable() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.enabled {
		return nil // Already enabled
	}

	m.enabled = true
	m.pid.Reset()

	// Start control loop
	period := time.Duration(m.config.SamplePeriod * float32(time.Second))
	m.ticker = time.NewTicker(period)
	go m.controlLoop()

	return nil
}

// Disable stops the motor control loop and stops the motor.
func (m *Motor) Disable() error {
	m.mu.Lock()

	if !m.enabled {
		m.mu.Unlock()
		return nil // Already disabled
	}

	m.enabled = false

	// Stop ticker
	if m.ticker != nil {
		m.ticker.Stop()
		m.ticker = nil
	}

	// Stop motor
	m.setPWM(0, 0)

	m.targetSpeed = 0
	m.direction = 0

	// Signal control loop to stop (unlock before closing channel)
	stopCh := m.stopCh
	m.stopCh = make(chan struct{})
	m.mu.Unlock()

	close(stopCh)

	return nil
}

// controlLoop runs the PID control loop.
func (m *Motor) controlLoop() {
	for {
		select {
		case <-m.stopCh:
			return
		case <-m.ticker.C:
			m.update()
		}
	}
}

// update performs one PID update cycle.
func (m *Motor) update() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.enabled {
		return
	}

	// Read current speed from encoder
	m.currentSpeed = float32(m.config.Encoder.RPM())

	// Update PID controller
	m.pid.Input = m.currentSpeed
	m.pid.Update(m.config.SamplePeriod)

	// Get PID output (normalized to [-1, 1])
	pidOutput := m.pid.Output

	// Convert to PWM duty cycle (0 to 1)
	pwmDuty := abs(pidOutput)

	// Set direction and PWM based on motor type
	if m.config.Type == TypeABPWM {
		// TypeABPWM: direction controlled by which PWM channel is active
		if pidOutput > 0 {
			// Forward: A=PWM, B=0
			m.pwmChannelA.Set(pwmDuty)
			m.pwmChannelB.Set(0)
		} else if pidOutput < 0 {
			// Reverse: A=0, B=PWM
			m.pwmChannelA.Set(0)
			m.pwmChannelB.Set(pwmDuty)
		} else {
			// Stop: A=0, B=0
			m.pwmChannelA.Set(0)
			m.pwmChannelB.Set(0)
		}
	} else {
		// TypeDirPWM or TypeABDirPWM: direction controlled by pin, speed by PWM
		dir := pidOutput >= 0

		if m.config.Type == TypeDirPWM {
			// Set direction pin
			m.config.Dir.Set(dir)
			// Set PWM
			m.pwmChannel.Set(pwmDuty)
		} else { // TypeABDirPWM
			// Set direction pins (A high = forward, B high = reverse)
			m.config.PinA.Set(dir)
			m.config.PinB.Set(!dir)
			// Set PWM
			m.pwmChannel.Set(pwmDuty)
		}
	}
}

// setPWM sets PWM and direction directly (for manual control).
func (m *Motor) setPWM(duty float32, direction int) {
	switch m.config.Type {
	case TypeDirPWM:
		m.config.Dir.Set(direction > 0)
		if m.pwmChannel != nil {
			m.pwmChannel.Set(abs(duty))
		}
	case TypeABPWM:
		if m.pwmChannelA != nil && m.pwmChannelB != nil {
			if direction > 0 {
				m.pwmChannelA.Set(abs(duty))
				m.pwmChannelB.Set(0)
			} else if direction < 0 {
				m.pwmChannelA.Set(0)
				m.pwmChannelB.Set(abs(duty))
			} else {
				m.pwmChannelA.Set(0)
				m.pwmChannelB.Set(0)
			}
		}
	case TypeABDirPWM:
		m.config.PinA.Set(direction > 0)
		m.config.PinB.Set(direction < 0)
		if m.pwmChannel != nil {
			m.pwmChannel.Set(abs(duty))
		}
	}
}

// abs returns the absolute value of a float32.
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Close stops the motor and cleans up resources.
func (m *Motor) Close() error {
	return m.Disable()
}
