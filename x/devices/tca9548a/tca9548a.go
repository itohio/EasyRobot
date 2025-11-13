// Package tca9548a provides a driver for the TCA9548A 8-channel I2C multiplexer.
//
// The package provides two ways to use the TCA9548A:
//  1. Direct device control via Device struct (manual channel switching)
//  2. Automatic routing via Router (provides I2C wrappers for each channel)
//
// The Router is recommended for most use cases as it automatically handles
// channel switching, allowing I2C device drivers to work transparently.
//
// This package works with both TinyGo (machine.I2C) and standard Go on Linux
// (Raspberry Pi) through the i2c.Bus abstraction.
//
// Datasheet: https://www.ti.com/lit/ds/symlink/tca9548a.pdf
package tca9548a

import (
	"errors"

	"github.com/itohio/EasyRobot/x/devices"
)

// DefaultAddress is the default I2C address for the TCA9548A
const DefaultAddress = 0x70

var (
	// ErrInvalidChannel is returned when an invalid channel number is provided.
	ErrInvalidChannel = errors.New("invalid channel number (must be 0-7)")
)

// Device wraps an I2C connection to a TCA9548A device.
type Device struct {
	bus     devices.I2C
	address uint8
}

// New creates a new TCA9548A connection. The I2C bus must already be configured.
// The bus can be either a TinyGo machine.I2C (wrapped with devices.NewTinyGoI2C)
// or a Linux I2C bus (created with devices.NewLinuxI2C).
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{
		bus:     bus,
		address: address,
	}
}

// Configure initializes the device. If init is true, resets all channels.
func (d *Device) Configure(init bool) error {
	if init {
		return d.Reset()
	}
	return nil
}

// SetChannel selects a single channel (0-7).
func (d *Device) SetChannel(channel uint8) error {
	if channel > 7 {
		return ErrInvalidChannel
	}
	return d.SetChannelMask(1 << channel)
}

// SetChannelMask sets multiple channels using a bitmask (bit 0 = channel 0, etc.).
func (d *Device) SetChannelMask(mask uint8) error {
	return d.bus.Tx(uint16(d.address), []byte{mask}, nil)
}

// Reset disables all channels.
func (d *Device) Reset() error {
	return d.bus.Tx(uint16(d.address), []byte{0x00}, nil)
}

