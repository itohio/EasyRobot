// Package as734x provides a driver for the AS7341 and AS7343 spectral sensors.
//
// The AS734x devices are multi-channel spectral sensors that can measure
// light across multiple wavelength bands. The driver supports both AS7341
// (10 channels) and AS7343 (18 channels) variants with automatic detection.
//
// Datasheets:
//   - AS7341: https://ams.com/as7341
//   - AS7343: https://ams.com/as7343
package as734x

import (
	"errors"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
)

var (
	errUnknownDevice   = errors.New("as734x: unknown device")
	errUnsupportedGain = errors.New("as734x: gain not supported by variant")
)

// Device wraps an I2C connection to an AS734x device.
type Device struct {
	bus     devices.I2C
	address uint8
	variant Variant
	cfg     Config
	bank    uint8
}

// New creates a new AS734x device connection. The I2C bus must already be configured.
// If address is 0, DefaultAddress will be used.
func New(bus devices.I2C, address uint8) *Device {
	if address == 0 {
		address = DefaultAddress
	}
	return &Device{
		bus:     bus,
		address: address,
		bank:    0,
	}
}

// Configure initializes the device and detects the variant.
func (d *Device) Configure(cfg Config) error {
	if cfg.Address == 0 {
		cfg.Address = DefaultAddress
	}
	d.address = cfg.Address

	if err := d.powerOn(); err != nil {
		return err
	}

	variant, err := d.detectVariant()
	if err != nil {
		return err
	}
	d.variant = variant
	d.cfg = cfg

	switch d.variant {
	case VariantAS7341:
		return d.configureAS7341()
	case VariantAS7343:
		return d.configureAS7343()
	default:
		return errUnknownDevice
	}
}

// Variant returns the detected device variant.
func (d *Device) Variant() Variant {
	return d.variant
}

// Read performs a spectral measurement and returns the raw data.
func (d *Device) Read() (RawMeasurement, error) {
	if d.variant == VariantAS7341 {
		return d.readAS7341()
	}
	if d.variant == VariantAS7343 {
		return d.readAS7343()
	}
	return RawMeasurement{}, errUnknownDevice
}

// Connected checks if the device is connected by attempting to detect the variant.
func (d *Device) Connected() bool {
	variant, err := d.detectVariant()
	return err == nil && variant != VariantUnknown
}

func (d *Device) powerOn() error {
	return d.writeRegRaw(regEnable, 0x01)
}

func (d *Device) detectVariant() (Variant, error) {
	if id, err := d.readRegRaw(regWhoAmIAS7341); err == nil {
		if (id & 0xFC) == (as7341ChipID << 2) {
			return VariantAS7341, nil
		}
	}

	if id, err := d.readAS7343ID(); err == nil {
		if id == as7343ChipID {
			return VariantAS7343, nil
		}
	}

	return VariantUnknown, errUnknownDevice
}

func (d *Device) readAS7343ID() (byte, error) {
	id, err := d.readRegRaw(regIDAS7343)
	if err == nil && id == as7343ChipID {
		return id, nil
	}
	if err := d.forceBankAS7343(1); err != nil {
		return 0, err
	}
	return d.readRegRaw(regIDAS7343)
}

func (d *Device) integrationTimeUs() uint32 {
	return integrationTimeUs(d.cfg.ATime, d.cfg.AStep)
}

func integrationTimeUs(atime uint8, astep uint16) uint32 {
	steps := uint32(atime) + 1
	delta := uint32(astep) + 1
	// 2.78 microseconds per step.
	return (steps * delta * 2780) / 1000
}

func (d *Device) waitForDataReady(reg byte, mask byte, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		b, err := d.readReg(reg)
		if err != nil {
			return err
		}
		if b&mask != 0 {
			return nil
		}
		if time.Now().After(deadline) {
			return errors.New("as734x: data ready timeout")
		}
		time.Sleep(time.Millisecond * 5)
	}
}

func (d *Device) enableSpectral(enable bool) error {
	reg, err := d.readReg(regEnable)
	if err != nil {
		return err
	}
	if enable {
		reg |= regSpectralEnable
	} else {
		reg &^= regSpectralEnable
	}
	return d.writeReg(regEnable, reg)
}

