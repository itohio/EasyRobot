//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// LinuxSerial implements Serial interface using Linux serial port.
type LinuxSerial struct {
	file   *os.File
	config SerialConfig
}

// NewSerial creates a new Serial interface for Linux.
// Common Linux serial devices:
//   - /dev/ttyUSB0, /dev/ttyUSB1, etc. - USB-to-serial adapters
//   - /dev/ttyAMA0 - Primary UART (on Raspberry Pi, GPIO 14/15)
//   - /dev/ttyS0 - Serial port (on Raspberry Pi, Mini UART)
//   - /dev/serial0 - Symlink to primary UART (on Raspberry Pi)
func NewSerial(device string) (*LinuxSerial, error) {
	return NewSerialWithConfig(device, DefaultSerialConfig())
}

// NewSerialWithConfig creates a new Serial interface with custom configuration.
func NewSerialWithConfig(device string, config SerialConfig) (*LinuxSerial, error) {
	file, err := os.OpenFile(device, os.O_RDWR|syscall.O_NOCTTY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open serial device %s: %w (ensure user is in dialout group)", device, err)
	}

	// Set to blocking mode (clear O_NONBLOCK flag)
	flags, _, errno := syscall.Syscall(syscall.SYS_FCNTL, file.Fd(), syscall.F_GETFL, 0)
	if errno != 0 {
		file.Close()
		return nil, fmt.Errorf("failed to get file flags: %v", errno)
	}
	flags &^= syscall.O_NONBLOCK
	_, _, errno = syscall.Syscall(syscall.SYS_FCNTL, file.Fd(), syscall.F_SETFL, flags)
	if errno != 0 {
		file.Close()
		return nil, fmt.Errorf("failed to set blocking mode: %v", errno)
	}

	return &LinuxSerial{
		file:   file,
		config: config,
	}, nil
}

// Read reads data from the serial port.
func (s *LinuxSerial) Read(p []byte) (n int, err error) {
	return s.file.Read(p)
}

// Write writes data to the serial port.
func (s *LinuxSerial) Write(p []byte) (n int, err error) {
	return s.file.Write(p)
}

// Buffered returns the number of bytes currently in the receive buffer.
// Only works if EnableBuffering was set to true in the configuration.
// Returns 0 if buffering is disabled (default) to avoid issues with devices
// like Arduino that expect immediate, unbuffered communication.
func (s *LinuxSerial) Buffered() int {
	if !s.config.EnableBuffering {
		return 0
	}

	const FIONREAD = 0x541B
	var n int32
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, s.file.Fd(), FIONREAD, uintptr(unsafe.Pointer(&n)))
	if errno != 0 {
		return 0
	}
	return int(n)
}

// Close closes the serial port.
func (s *LinuxSerial) Close() error {
	return s.file.Close()
}

// File returns the underlying *os.File.
func (s *LinuxSerial) File() *os.File {
	return s.file
}
