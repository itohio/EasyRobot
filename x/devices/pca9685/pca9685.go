// Package pca9685 provides a driver for the PCA9685 16-channel 12-bit PWM driver.
//
// Datasheet: https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf
package pca9685

import (
	"math"

	"github.com/itohio/EasyRobot/x/devices"
)

// I2C addresses
const (
	DefaultAddress = 0x40
	SubAdr1        = 0x02
	SubAdr2        = 0x03
	SubAdr3        = 0x04
)

// Register addresses
const (
	Mode1      = 0x00
	Prescale   = 0xFE
	LED0OnL    = 0x06
	LED0OnH    = 0x07
	LED0OffL   = 0x08
	LED0OffH   = 0x09
	AllLEDOnL  = 0xFA
	AllLEDOnH  = 0xFB
	AllLEDOffL = 0xFC
	AllLEDOffH = 0xFD
)

// Device wraps an I2C connection to a PCA9685 device.
type Device struct {
	bus     devices.I2C
	address uint8
}

// New creates a new PCA9685 connection. The I2C bus must already be configured.
// The bus can be either a TinyGo machine.I2C (wrapped with devices.NewI2C)
// or a Linux I2C bus (created with devices.NewI2C).
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{
		bus:     bus,
		address: address,
	}
}

// Configure initializes the device. If init is true, resets the device.
func (d *Device) Configure(init bool) error {
	if init {
		return d.Reset()
	}
	return nil
}

// Reset resets the device to default state.
func (d *Device) Reset() error {
	return d.write8(Mode1, 0x00)
}

// SetFrequency sets the PWM frequency in Hz (typically 24-1526 Hz).
func (d *Device) SetFrequency(freq float32) error {
	// Calculate prescale value: prescale = round(25000000 / (4096 * freq)) - 1
	prescale := uint8(math.Round(float64(25000000/(4096*freq))) - 1)

	// Read current mode
	oldMode, err := d.read8(Mode1)
	if err != nil {
		return err
	}

	// Put device to sleep
	newMode := (oldMode & 0x7F) | 0x10
	if err := d.write8(Mode1, newMode); err != nil {
		return err
	}

	// Set prescaler
	if err := d.write8(Prescale, prescale); err != nil {
		return err
	}

	// Restore old mode
	if err := d.write8(Mode1, oldMode); err != nil {
		return err
	}

	// Wait for oscillator to stabilize
	// Note: In TinyGo, use machine.Delay. In standard Go, use time.Sleep.
	// This is a simplified delay - implementations should handle this appropriately.
	// For now, we rely on the caller to ensure proper timing.

	// Enable auto-increment
	if err := d.write8(Mode1, oldMode|0xa1); err != nil {
		return err
	}

	return nil
}

// SetPWM sets the PWM value for a channel (0-15).
// value is in range 0.0 to 1.0, or use SetPWMRaw for raw 12-bit values.
func (d *Device) SetPWM(channel uint8, value float32, invert bool) error {
	rawValue := uint16(value * 4095)
	return d.SetPWMRaw(channel, rawValue, invert)
}

// SetPWMRaw sets the raw 12-bit PWM value for a channel (0-15).
func (d *Device) SetPWMRaw(channel uint8, value uint16, invert bool) error {
	if channel > 15 {
		return devices.ErrInvalidInputPin
	}

	var on, off uint16
	if invert {
		off = 4095 - value
	} else {
		off = value
	}

	// Write 32-bit value: off in upper 16 bits, on in lower 16 bits
	data := []byte{
		byte(off & 0xFF),        // LEDx_OFF_L
		byte((off >> 8) & 0x0F), // LEDx_OFF_H (only lower 4 bits)
		byte(on & 0xFF),         // LEDx_ON_L
		byte((on >> 8) & 0x0F),  // LEDx_ON_H (only lower 4 bits)
	}

	reg := LED0OnL + 4*channel
	return d.bus.Tx(uint16(d.address), append([]byte{reg}, data...), nil)
}

func (d *Device) write8(reg uint8, value uint8) error {
	return d.bus.Tx(uint16(d.address), []byte{reg, value}, nil)
}

func (d *Device) read8(reg uint8) (uint8, error) {
	data := make([]byte, 1)
	err := d.bus.Tx(uint16(d.address), []byte{reg}, data)
	return data[0], err
}
