//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
)

// PinChange represents one or more trigger events that can happen on a given GPIO pin
// on the RP2040. ORed PinChanges are valid input to most IRQ functions.
type PinChange uint8

// Pin change interrupt constants for SetInterrupt.
const (
	// Edge falling
	PinFalling PinChange = 4 << iota
	// Edge rising
	PinRising

	PinToggle = PinFalling | PinRising
)

// LinuxPin implements Pin interface using Linux sysfs GPIO.
type LinuxPin struct {
	pinNum int
	value  *os.File
}

// NewLinuxPin creates a new Pin interface for Linux GPIO.
// pinNum is the GPIO pin number (e.g., 18 for GPIO18).
// The pin must be exported first (e.g., echo 18 > /sys/class/gpio/export).
func NewPin(pinNum int) (*LinuxPin, error) {
	valuePath := fmt.Sprintf("/sys/class/gpio/gpio%d/value", pinNum)
	value, err := os.OpenFile(valuePath, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open GPIO pin %d: %w (ensure pin is exported)", pinNum, err)
	}

	return &LinuxPin{
		pinNum: pinNum,
		value:  value,
	}, nil
}

// Get returns the current pin state.
func (p *LinuxPin) Get() bool {
	buf := make([]byte, 1)
	_, err := p.value.ReadAt(buf, 0)
	if err != nil {
		return false
	}
	return buf[0] == '1'
}

// Set sets the pin state.
func (p *LinuxPin) Set(value bool) {
	var b byte = '0'
	if value {
		b = '1'
	}
	p.value.WriteAt([]byte{b}, 0)
}

// High sets the pin to high.
func (p *LinuxPin) High() {
	p.value.WriteAt([]byte{'1'}, 0)
}

// Low sets the pin to low.
func (p *LinuxPin) Low() {
	p.value.WriteAt([]byte{'0'}, 0)
}

// Close closes the GPIO pin file.
func (p *LinuxPin) Close() error {
	return p.value.Close()
}

// PinNum returns the GPIO pin number.
func (p *LinuxPin) PinNum() int {
	return p.pinNum
}
