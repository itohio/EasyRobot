//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"path/filepath"
)

// RaspberryPi provides helper functions for common Raspberry Pi device paths and configuration.
// These functions help locate standard Raspberry Pi hardware interfaces.
type RaspberryPi struct{}

// DefaultRaspberryPi returns a RaspberryPi helper instance.
func DefaultRaspberryPi() *RaspberryPi {
	return &RaspberryPi{}
}

// I2CDevice returns the standard I2C device path for the given bus number.
// Common Raspberry Pi I2C buses:
//   - Bus 1: /dev/i2c-1 (primary I2C, GPIO 2/3) - most common
//   - Bus 0: /dev/i2c-0 (secondary I2C, if available)
func (rpi *RaspberryPi) I2CDevice(bus int) string {
	return fmt.Sprintf("/dev/i2c-%d", bus)
}

// SPIDevice returns the standard SPI device path for the given bus and chip select.
// Common Raspberry Pi SPI devices:
//   - /dev/spidev0.0 (SPI0, CS0) - Primary SPI
//   - /dev/spidev0.1 (SPI0, CS1) - Primary SPI, alternate CS
//   - /dev/spidev1.0 (SPI1, CS0) - Auxiliary SPI (if available)
func (rpi *RaspberryPi) SPIDevice(bus, cs int) string {
	return fmt.Sprintf("/dev/spidev%d.%d", bus, cs)
}

// SerialDevice returns common serial device paths.
// Returns the primary serial device, checking symlinks first.
// Common paths:
//   - /dev/serial0 - Symlink to primary UART (recommended)
//   - /dev/ttyAMA0 - Primary UART (GPIO 14/15)
//   - /dev/ttyS0 - Mini UART (GPIO 14/15)
func (rpi *RaspberryPi) SerialDevice() (string, error) {
	// Try symlink first (recommended)
	devices := []string{
		"/dev/serial0",
		"/dev/ttyAMA0",
		"/dev/ttyS0",
	}

	for _, dev := range devices {
		if _, err := os.Stat(dev); err == nil {
			// Resolve symlink if it's a symlink
			if resolved, err := os.Readlink(dev); err == nil {
				if filepath.IsAbs(resolved) {
					return resolved, nil
				}
				return filepath.Join(filepath.Dir(dev), resolved), nil
			}
			return dev, nil
		}
	}

	return "", fmt.Errorf("no serial device found (checked: %v)", devices)
}

// PWMChip returns the standard PWM chip path.
// Raspberry Pi typically uses pwmchip0 for hardware PWM.
func (rpi *RaspberryPi) PWMChip(chip int) string {
	return fmt.Sprintf("/sys/class/pwm/pwmchip%d", chip)
}

// GPIOExportPath returns the GPIO export path for sysfs GPIO.
func (rpi *RaspberryPi) GPIOExportPath() string {
	return "/sys/class/gpio/export"
}

// GPIOBasePath returns the base path for a GPIO pin in sysfs.
func (rpi *RaspberryPi) GPIOBasePath(pin int) string {
	return fmt.Sprintf("/sys/class/gpio/gpio%d", pin)
}

// CheckI2CEnabled checks if I2C is enabled on the Raspberry Pi.
// Returns true if /dev/i2c-1 exists (primary I2C bus).
func (rpi *RaspberryPi) CheckI2CEnabled() bool {
	_, err := os.Stat("/dev/i2c-1")
	return err == nil
}

// CheckSPIEnabled checks if SPI is enabled on the Raspberry Pi.
// Returns true if /dev/spidev0.0 exists (primary SPI device).
func (rpi *RaspberryPi) CheckSPIEnabled() bool {
	_, err := os.Stat("/dev/spidev0.0")
	return err == nil
}

// CheckSerialEnabled checks if serial is enabled on the Raspberry Pi.
// Returns true if a serial device is available.
func (rpi *RaspberryPi) CheckSerialEnabled() bool {
	_, err := rpi.SerialDevice()
	return err == nil
}

// CheckPWMEnabled checks if PWM is enabled on the Raspberry Pi.
// Returns true if pwmchip0 exists.
func (rpi *RaspberryPi) CheckPWMEnabled() bool {
	_, err := os.Stat("/sys/class/pwm/pwmchip0")
	return err == nil
}

// HardwareInfo returns information about available Raspberry Pi hardware interfaces.
type HardwareInfo struct {
	I2CAvailable bool
	SPIAvailable bool
	SerialAvailable bool
	PWMAvailable bool
	I2CDevices   []string
	SPIDevices   []string
	SerialDevice string
}

// GetHardwareInfo returns information about available hardware interfaces.
func (rpi *RaspberryPi) GetHardwareInfo() HardwareInfo {
	info := HardwareInfo{
		I2CAvailable:  rpi.CheckI2CEnabled(),
		SPIAvailable:  rpi.CheckSPIEnabled(),
		SerialAvailable: rpi.CheckSerialEnabled(),
		PWMAvailable:  rpi.CheckPWMEnabled(),
	}

	// Find I2C devices
	for i := 0; i < 10; i++ {
		dev := rpi.I2CDevice(i)
		if _, err := os.Stat(dev); err == nil {
			info.I2CDevices = append(info.I2CDevices, dev)
		}
	}

	// Find SPI devices
	for bus := 0; bus < 2; bus++ {
		for cs := 0; cs < 2; cs++ {
			dev := rpi.SPIDevice(bus, cs)
			if _, err := os.Stat(dev); err == nil {
				info.SPIDevices = append(info.SPIDevices, dev)
			}
		}
	}

	// Get serial device
	if info.SerialAvailable {
		info.SerialDevice, _ = rpi.SerialDevice()
	}

	return info
}

