//go:build !tinygo && linux

// Package devices provides Raspberry Pi hardware interface implementations.
// This file implements PWM (Pulse Width Modulation) using Linux sysfs PWM interface.
package devices

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"
)

var (
	// ErrPinNotSupported is returned when a pin does not support PWM
	ErrPinNotSupported = fmt.Errorf("pin does not support PWM")
	// ErrPWMNotConfigured is returned when PWM is used before configuration
	ErrPWMNotConfigured = fmt.Errorf("PWM device not configured")
)

// pinToPWMChannel maps GPIO pin numbers to PWM chip and channel.
// Raspberry Pi hardware PWM pins (BCM numbering):
// - GPIO 12, 18 -> PWM0 (channel 0) - Hardware PWM
// - GPIO 13, 19 -> PWM1 (channel 1) - Hardware PWM
// Note: GPIO 12/13 and 18/19 are alternate functions on the same PWM channels.
// GPIO 18 and 19 are more commonly available and recommended.
// PWM must be enabled in device tree (typically enabled by default on newer Raspberry Pi OS).
var pinToPWMChannel = map[int]struct {
	chip    int // PWM chip number (usually 0)
	channel int // PWM channel number (0 or 1)
}{
	12: {chip: 0, channel: 0}, // PWM0
	18: {chip: 0, channel: 0}, // PWM0 (alternate)
	13: {chip: 0, channel: 1}, // PWM1
	19: {chip: 0, channel: 1}, // PWM1 (alternate)
}

// LinuxPWMDevice provides PWM functionality for Linux (Raspberry Pi) using sysfs PWM interface.
type LinuxPWMDevice struct {
	mu         sync.Mutex
	frequency  uint32 // Frequency in Hz
	channels   map[*LinuxPin]*LinuxPWMChannel
	configured bool
}

// LinuxPWMChannel represents a single PWM channel on Linux.
type LinuxPWMChannel struct {
	device   *LinuxPWMDevice
	pin      *LinuxPin
	chip     int
	channel  int
	periodNs int64 // Period in nanoseconds
	enabled  bool
}

// NewPWMDevice creates a new PWM device for Linux (Raspberry Pi).
func NewPWMDevice() PWMDevice {
	return &LinuxPWMDevice{
		channels: make(map[*LinuxPin]*LinuxPWMChannel),
	}
}

// Configure configures the PWM device with the specified frequency.
func (d *LinuxPWMDevice) Configure(frequency uint32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if frequency == 0 {
		return fmt.Errorf("frequency must be greater than 0")
	}

	d.frequency = frequency
	d.configured = true

	// Update all existing channels with the new frequency
	for _, ch := range d.channels {
		periodNs := int64(1e9 / float64(frequency))
		if err := ch.setPeriod(periodNs); err != nil {
			return fmt.Errorf("failed to update channel period: %w", err)
		}
		ch.periodNs = periodNs
	}

	return nil
}

// SetFrequency changes the PWM frequency for all channels.
func (d *LinuxPWMDevice) SetFrequency(frequency uint32) error {
	return d.Configure(frequency)
}

// Channel returns a PWM channel for the specified pin.
func (d *LinuxPWMDevice) Channel(pin Pin) (PWM, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.configured {
		return nil, ErrPWMNotConfigured
	}

	linuxPin, ok := pin.(*LinuxPin)
	if !ok {
		return nil, fmt.Errorf("pin must be a LinuxPin")
	}

	// Check if channel already exists
	if ch, exists := d.channels[linuxPin]; exists {
		return ch, nil
	}

	// Get GPIO pin number
	pinNum := linuxPin.PinNum()

	// Check if pin supports PWM
	pwmInfo, ok := pinToPWMChannel[pinNum]
	if !ok {
		return nil, fmt.Errorf("GPIO %d does not support hardware PWM: %w", pinNum, ErrPinNotSupported)
	}

	// Create channel
	channel := &LinuxPWMChannel{
		device:   d,
		pin:      linuxPin,
		chip:     pwmInfo.chip,
		channel:  pwmInfo.channel,
		periodNs: int64(1e9 / float64(d.frequency)),
	}

	// Export PWM channel
	if err := channel.export(); err != nil {
		return nil, fmt.Errorf("failed to export PWM channel: %w", err)
	}

	// Set period
	if err := channel.setPeriod(channel.periodNs); err != nil {
		channel.unexport()
		return nil, fmt.Errorf("failed to set PWM period: %w", err)
	}

	// Store channel
	d.channels[linuxPin] = channel

	return channel, nil
}

// export exports the PWM channel via sysfs.
func (ch *LinuxPWMChannel) export() error {
	exportPath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/export", ch.chip)

	// Check if already exported
	pwmPath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/pwm%d", ch.chip, ch.channel)
	if _, err := os.Stat(pwmPath); err == nil {
		// Already exported
		return nil
	}

	// Export the channel
	if err := os.WriteFile(exportPath, []byte(strconv.Itoa(ch.channel)), 0); err != nil {
		return fmt.Errorf("failed to export PWM channel %d: %w", ch.channel, err)
	}

	// Wait for sysfs to create the directory
	time.Sleep(100 * time.Millisecond)

	return nil
}

// unexport unexports the PWM channel via sysfs.
func (ch *LinuxPWMChannel) unexport() error {
	unexportPath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/unexport", ch.chip)
	return os.WriteFile(unexportPath, []byte(strconv.Itoa(ch.channel)), 0)
}

// setPeriod sets the PWM period in nanoseconds.
func (ch *LinuxPWMChannel) setPeriod(periodNs int64) error {
	periodPath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/pwm%d/period", ch.chip, ch.channel)
	return os.WriteFile(periodPath, []byte(strconv.FormatInt(periodNs, 10)), 0)
}

// setDutyCycle sets the PWM duty cycle in nanoseconds.
func (ch *LinuxPWMChannel) setDutyCycle(dutyNs int64) error {
	if dutyNs < 0 {
		dutyNs = 0
	}
	if dutyNs > ch.periodNs {
		dutyNs = ch.periodNs
	}

	dutyPath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/pwm%d/duty_cycle", ch.chip, ch.channel)
	return os.WriteFile(dutyPath, []byte(strconv.FormatInt(dutyNs, 10)), 0)
}

// enable enables the PWM channel.
func (ch *LinuxPWMChannel) enable() error {
	if ch.enabled {
		return nil
	}

	enablePath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/pwm%d/enable", ch.chip, ch.channel)
	if err := os.WriteFile(enablePath, []byte("1"), 0); err != nil {
		return fmt.Errorf("failed to enable PWM channel: %w", err)
	}

	ch.enabled = true
	return nil
}

// disable disables the PWM channel.
func (ch *LinuxPWMChannel) disable() error {
	if !ch.enabled {
		return nil
	}

	enablePath := fmt.Sprintf("/sys/class/pwm/pwmchip%d/pwm%d/enable", ch.chip, ch.channel)
	if err := os.WriteFile(enablePath, []byte("0"), 0); err != nil {
		return fmt.Errorf("failed to disable PWM channel: %w", err)
	}

	ch.enabled = false
	return nil
}

// Set sets the duty cycle (0.0 to 1.0).
func (ch *LinuxPWMChannel) Set(duty float32) error {
	if duty < 0.0 {
		duty = 0.0
	}
	if duty > 1.0 {
		duty = 1.0
	}

	dutyNs := int64(float64(ch.periodNs) * float64(duty))

	if err := ch.setDutyCycle(dutyNs); err != nil {
		return err
	}

	return ch.enable()
}

// SetMicroseconds sets the pulse width in microseconds.
func (ch *LinuxPWMChannel) SetMicroseconds(us uint32) error {
	dutyNs := int64(us) * 1000 // Convert microseconds to nanoseconds

	if err := ch.setDutyCycle(dutyNs); err != nil {
		return err
	}

	return ch.enable()
}

// Stop stops the PWM output (sets duty to 0).
func (ch *LinuxPWMChannel) Stop() error {
	return ch.disable()
}

// Close closes the PWM channel and unexports it.
func (ch *LinuxPWMChannel) Close() error {
	ch.device.mu.Lock()
	defer ch.device.mu.Unlock()

	if err := ch.disable(); err != nil {
		return err
	}

	if err := ch.unexport(); err != nil {
		return err
	}

	// Remove from device's channel map
	delete(ch.device.channels, ch.pin)

	return nil
}
