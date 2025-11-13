// Package pcf8574 provides a driver for the PCF8574 8-bit I2C GPIO expander.
//
// Datasheet: https://www.ti.com/lit/ds/symlink/pcf8574.pdf
package pcf8574

import (
	"github.com/itohio/EasyRobot/x/devices"
)

// DefaultAddress is the default I2C address for the PCF8574
const DefaultAddress = 0x20

// Device wraps an I2C connection to a PCF8574 device.
type Device struct {
	bus     devices.I2C
	address uint8
	ioData  uint8
}

// New creates a new PCF8574 connection. The I2C bus must already be configured.
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{
		bus:     bus,
		address: address,
	}
}

// Configure initializes the device. If init is true, reads current pin states.
func (d *Device) Configure(init bool) error {
	if init {
		return d.ReadAll()
	}
	return nil
}

// WriteAll writes all 8 pins at once.
func (d *Device) WriteAll(data uint8) error {
	d.ioData = data
	return d.bus.Tx(uint16(d.address), []byte{d.ioData}, nil)
}

// ReadAll reads all 8 pins at once.
func (d *Device) ReadAll() error {
	data := make([]byte, 1)
	if err := d.bus.Tx(uint16(d.address), nil, data); err != nil {
		return err
	}
	d.ioData = data[0]
	return nil
}

// SetPin sets a single pin (0-7) to the given value.
func (d *Device) SetPin(pin uint8, value bool) error {
	if pin > 7 {
		return devices.ErrInvalidInputPin
	}

	if value {
		d.ioData |= (1 << pin)
	} else {
		d.ioData &^= (1 << pin)
	}
	return d.WriteAll(d.ioData)
}

// GetPin reads a single pin (0-7).
func (d *Device) GetPin(pin uint8) (bool, error) {
	if pin > 7 {
		return false, devices.ErrInvalidInputPin
	}

	if err := d.ReadAll(); err != nil {
		return false, err
	}
	return (d.ioData & (1 << pin)) != 0, nil
}

// GetData returns the last read pin state.
func (d *Device) GetData() uint8 {
	return d.ioData
}
