//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"syscall"
)

// I2CTargetEvent represents an I2C target mode event.
type I2CTargetEvent uint8

const (
	I2CTargetEventStart I2CTargetEvent = iota
	I2CTargetEventStop
	I2CTargetEventAddress
	I2CTargetEventData
	I2CTargetEventError
)

// LinuxI2C implements I2C bus using Linux i2c-dev interface.
// Note: Linux I2C only supports controller mode, not target mode.
type LinuxI2C struct {
	fd   *os.File
	bus  int
	addr uint8
}

// Ensure LinuxI2C implements I2C interface
var _ I2C = (*LinuxI2C)(nil)

// NewI2C creates a new I2C bus for Linux (Raspberry Pi).
// device should be like "/dev/i2c-1" for I2C bus 1.
func NewI2C(device string) (*LinuxI2C, error) {
	fd, err := os.OpenFile(device, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open I2C device %s: %w", device, err)
	}

	// Extract bus number from device path
	var bus int
	_, err = fmt.Sscanf(device, "/dev/i2c-%d", &bus)
	if err != nil {
		fd.Close()
		return nil, fmt.Errorf("invalid I2C device path: %s", device)
	}

	return &LinuxI2C{
		fd:  fd,
		bus: bus,
	}, nil
}

// ReadRegister reads a register value from the device.
func (b *LinuxI2C) ReadRegister(addr uint8, r uint8, buf []byte) error {
	// Set slave address if it changed
	if b.addr != addr {
		if err := b.setAddr(addr); err != nil {
			return err
		}
		b.addr = addr
	}

	// Write register address
	if _, err := b.fd.Write([]byte{r}); err != nil {
		return fmt.Errorf("I2C write register failed: %w", err)
	}

	// Read data
	if len(buf) > 0 {
		n, err := b.fd.Read(buf)
		if err != nil {
			return fmt.Errorf("I2C read register failed: %w", err)
		}
		if n != len(buf) {
			return fmt.Errorf("I2C read register incomplete: read %d of %d bytes", n, len(buf))
		}
	}

	return nil
}

// WriteRegister writes a register value to the device.
func (b *LinuxI2C) WriteRegister(addr uint8, r uint8, buf []byte) error {
	// Set slave address if it changed
	if b.addr != addr {
		if err := b.setAddr(addr); err != nil {
			return err
		}
		b.addr = addr
	}

	// Write register address + data
	data := append([]byte{r}, buf...)
	n, err := b.fd.Write(data)
	if err != nil {
		return fmt.Errorf("I2C write register failed: %w", err)
	}
	if n != len(data) {
		return fmt.Errorf("I2C write register incomplete: wrote %d of %d bytes", n, len(data))
	}

	return nil
}

// Tx performs a generic I2C transaction.
func (b *LinuxI2C) Tx(addr uint16, w, r []byte) error {
	// Set slave address if it changed
	if b.addr != uint8(addr) {
		if err := b.setAddr(uint8(addr)); err != nil {
			return err
		}
		b.addr = uint8(addr)
	}

	// Write data if provided
	if len(w) > 0 {
		n, err := b.fd.Write(w)
		if err != nil {
			return fmt.Errorf("I2C write failed: %w", err)
		}
		if n != len(w) {
			return fmt.Errorf("I2C write incomplete: wrote %d of %d bytes", n, len(w))
		}
	}

	// Read data if requested
	if len(r) > 0 {
		n, err := b.fd.Read(r)
		if err != nil {
			return fmt.Errorf("I2C read failed: %w", err)
		}
		if n != len(r) {
			return fmt.Errorf("I2C read incomplete: read %d of %d bytes", n, len(r))
		}
	}

	return nil
}

// setAddr sets the I2C slave address using ioctl.
func (b *LinuxI2C) setAddr(addr uint8) error {
	// I2C_SLAVE ioctl
	const I2C_SLAVE = 0x0703
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, b.fd.Fd(), I2C_SLAVE, uintptr(addr))
	if errno != 0 {
		return fmt.Errorf("failed to set I2C slave address: %v", errno)
	}
	return nil
}

// Close closes the I2C bus.
func (b *LinuxI2C) Close() error {
	return b.fd.Close()
}

// Device returns the device path.
func (b *LinuxI2C) Device() string {
	return fmt.Sprintf("/dev/i2c-%d", b.bus)
}

// Listen is not supported on Linux I2C (controller mode only).
func (b *LinuxI2C) Listen(addr uint16) error {
	return ErrNotSupported
}

// WaitForEvent is not supported on Linux I2C (controller mode only).
func (b *LinuxI2C) WaitForEvent(buf []byte) (I2CTargetEvent, int, error) {
	return I2CTargetEventError, 0, ErrNotSupported
}

// Reply is not supported on Linux I2C (controller mode only).
func (b *LinuxI2C) Reply(buf []byte) error {
	return ErrNotSupported
}
