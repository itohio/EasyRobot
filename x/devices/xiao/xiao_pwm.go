//go:build sam && xiao

// Package xiao provides XIAO-specific device implementations.
// XIAO is based on SAM D21 microcontroller which uses TCC (Timer/Counter Controller) for PWM.
package xiao

import (
	"errors"
	"machine"
	"sync"

	"github.com/itohio/EasyRobot/x/devices"
)

var (
	ErrPinNotSupported = errors.New("pin does not support PWM")
	ErrInvalidChannel  = errors.New("invalid PWM channel")
)

// PinTCCMapping maps XIAO pins to their TCC peripheral and channel.
// Based on SAM D21 pinout for Seeed Studio XIAO.
// TCC peripherals: TCC0, TCC1, TCC2
// Each TCC has multiple channels (usually 2-4 channels per TCC)
type PinTCCMapping struct {
	TCC     *machine.TCC
	Channel uint8
}

// XIAOPWMDevice provides PWM functionality for XIAO board using TCC peripherals.
type XIAOPWMDevice struct {
	mu        sync.Mutex
	frequency uint32
	channels  map[machine.Pin]*XIAOPWMChannel
	tccMap    map[machine.Pin]PinTCCMapping
}

// XIAOPWMChannel represents a single PWM channel on XIAO.
type XIAOPWMChannel struct {
	device  *XIAOPWMDevice
	pin     machine.Pin
	mapping PinTCCMapping
	period  uint32 // Period in TCC counts
}

// NewPWMDevice creates a new PWM device for XIAO board.
func NewPWMDevice() devices.PWMDevice {
	return &XIAOPWMDevice{
		channels: make(map[machine.Pin]*XIAOPWMChannel),
		tccMap:   getXIAOTCCMapping(),
	}
}

// GetMapping returns the TCC mapping for a pin (for backward compatibility).
// This is exported for use by legacy code that needs direct TCC access.
func (d *XIAOPWMDevice) GetMapping(pin machine.Pin) (PinTCCMapping, bool) {
	mapping, ok := d.tccMap[pin]
	return mapping, ok
}

// getXIAOTCCMapping returns the mapping of XIAO pins to TCC peripherals and channels.
// Based on SAM D21 pinout for Seeed Studio XIAO.
//
// NOTE: Pin mappings should be verified against actual XIAO board pinout.
// The mapping below is based on typical SAM D21 pin assignments.
// Some pins may not be available or may be assigned differently on XIAO.
//
// Verified pins (from existing manipulator code):
// - D8: TCC1 Channel 0
// - D9: TCC0 Channel 0
// - D10: TCC1 Channel 1
func getXIAOTCCMapping() map[machine.Pin]PinTCCMapping {
	return map[machine.Pin]PinTCCMapping{
		// TCC0 channels (WO0-WO7)
		// Verified:
		machine.D9: {TCC: machine.TCC0, Channel: 0}, // WO[0] - Verified
		// Additional TCC0 pins (verify against XIAO pinout):
		machine.D3: {TCC: machine.TCC0, Channel: 1}, // WO[1]
		machine.D4: {TCC: machine.TCC0, Channel: 2}, // WO[2]
		machine.D5: {TCC: machine.TCC0, Channel: 3}, // WO[3]
		machine.D6: {TCC: machine.TCC0, Channel: 4}, // WO[4]
		machine.D7: {TCC: machine.TCC0, Channel: 5}, // WO[5]

		// TCC1 channels (WO0-WO5)
		// Verified:
		machine.D8:  {TCC: machine.TCC1, Channel: 0}, // WO[0] - Verified
		machine.D10: {TCC: machine.TCC1, Channel: 1}, // WO[1] - Verified
		// Additional TCC1 pins (verify against XIAO pinout):
		machine.D2: {TCC: machine.TCC1, Channel: 2}, // WO[2]
		machine.A0: {TCC: machine.TCC1, Channel: 3}, // WO[3]
		machine.A1: {TCC: machine.TCC1, Channel: 4}, // WO[4]
		machine.A2: {TCC: machine.TCC1, Channel: 5}, // WO[5]

		// TCC2 channels (WO0-WO3)
		// TCC2 has fewer channels - add pins as verified
		// machine.D1: {TCC: machine.TCC2, Channel: 0}, // WO[0]
		// machine.D0: {TCC: machine.TCC2, Channel: 1}, // WO[1]
	}
}

// Configure initializes the PWM device with the specified frequency.
// Default frequency is 50Hz (common for servos).
func (d *XIAOPWMDevice) Configure(frequency uint32) error {
	if frequency == 0 {
		frequency = 50 // Default: 50Hz for servos
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	d.frequency = frequency

	// Configure all TCC peripherals that are in use
	tccConfigured := make(map[*machine.TCC]bool)

	// Configure TCCs for all pins in the mapping (even if not yet used)
	for _, mapping := range d.tccMap {
		if !tccConfigured[mapping.TCC] {
			err := d.configureTCC(mapping.TCC, frequency)
			if err != nil {
				return err
			}
			tccConfigured[mapping.TCC] = true
		}
	}

	return nil
}

// configureTCC configures a TCC peripheral for PWM operation.
func (d *XIAOPWMDevice) configureTCC(tcc *machine.TCC, frequency uint32) error {
	// Configure TCC for PWM mode
	// Period = CPU frequency / prescaler / frequency
	// For SAM D21: typically 48MHz CPU, but may vary
	// Using prescaler of 64 gives good range for servo frequencies
	prescaler := uint8(64)
	cpuFreq := uint32(48000000) // 48MHz typical for SAM D21

	// Calculate period: period = (CPU_FREQ / prescaler) / frequency
	period := (cpuFreq / uint32(prescaler)) / frequency

	// Configure TCC
	err := tcc.Configure(machine.PWMConfig{
		Period: period,
	})
	if err != nil {
		return err
	}

	return nil
}

// SetFrequency changes the PWM frequency for all channels.
// Note: This may require reconfiguring TCC peripherals.
func (d *XIAOPWMDevice) SetFrequency(frequency uint32) error {
	if frequency == 0 {
		return devices.ErrInvalidValue
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.frequency == frequency {
		return nil // Already at this frequency
	}

	oldFreq := d.frequency
	d.frequency = frequency

	// Reconfigure all TCCs
	tccConfigured := make(map[*machine.TCC]bool)
	for pin, channel := range d.channels {
		mapping := d.tccMap[pin]
		if !tccConfigured[mapping.TCC] {
			err := d.configureTCC(mapping.TCC, frequency)
			if err != nil {
				// Rollback on error
				d.frequency = oldFreq
				return err
			}
			tccConfigured[mapping.TCC] = true
		}

		// Update channel period
		cpuFreq := uint32(48000000)
		prescaler := uint32(64)
		channel.period = (cpuFreq / prescaler) / frequency
	}

	return nil
}

// Channel returns a PWM channel for the specified pin.
func (d *XIAOPWMDevice) Channel(pin devices.Pin) (devices.PWM, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Convert devices.Pin to machine.Pin
	machinePin, ok := pin.(machine.Pin)
	if !ok {
		return nil, ErrInvalidChannel
	}

	// Check if pin supports PWM
	mapping, ok := d.tccMap[machinePin]
	if !ok {
		return nil, ErrPinNotSupported
	}

	// Check if channel already exists
	if ch, exists := d.channels[machinePin]; exists {
		return ch, nil
	}

	// Configure pin for PWM output
	machinePin.Configure(machine.PinConfig{Mode: machine.PinPWM})

	// Calculate period for this frequency
	cpuFreq := uint32(48000000)
	prescaler := uint32(64)
	period := (cpuFreq / prescaler) / d.frequency

	// Create channel
	channel := &XIAOPWMChannel{
		device:  d,
		pin:     machinePin,
		mapping: mapping,
		period:  period,
	}

	// Store channel
	d.channels[machinePin] = channel

	// Configure TCC if not already configured
	if d.frequency == 0 {
		d.frequency = 50 // Default frequency
	}
	err := d.configureTCC(mapping.TCC, d.frequency)
	if err != nil {
		delete(d.channels, machinePin)
		return nil, err
	}

	return channel, nil
}

// Set sets the duty cycle (0.0 to 1.0).
func (ch *XIAOPWMChannel) Set(duty float32) error {
	if duty < 0.0 || duty > 1.0 {
		return devices.ErrInvalidValue
	}

	// Calculate PWM value: value = duty * period
	value := uint32(float32(ch.period) * duty)

	// TCC.Set(channel, value) sets the compare value for the channel
	// This controls the pulse width (duty cycle)
	err := ch.mapping.TCC.Set(ch.mapping.Channel, value)
	if err != nil {
		return err
	}

	return nil
}

// SetMicroseconds sets the pulse width in microseconds.
// Commonly used for servo control (typically 500-2500 microseconds for servos).
func (ch *XIAOPWMChannel) SetMicroseconds(us uint32) error {
	if ch.device.frequency == 0 {
		return devices.ErrInvalidState
	}

	// Calculate duty cycle from microseconds
	// Period in microseconds = 1,000,000 / frequency
	periodUs := uint32(1000000 / ch.device.frequency)

	if us > periodUs {
		return devices.ErrInvalidValue
	}

	// Duty = us / periodUs
	duty := float32(us) / float32(periodUs)

	return ch.Set(duty)
}

// Stop stops the PWM output (sets duty to 0).
func (ch *XIAOPWMChannel) Stop() error {
	return ch.Set(0.0)
}
